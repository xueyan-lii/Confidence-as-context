# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time
import copy
import os
import json
from datetime import datetime

from typing import Dict, List, Tuple, Union, Optional

import torch
from tqdm import tqdm

from lm_eval.evaluator import evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import get_task_dict, TaskManager
from lm_eval.utils import make_table
from lm_eval.api.instance import Instance
from omegaconf import DictConfig
from lm_eval.models.utils import (
    Collator,
    handle_stop_sequences,
)
from torchtune import config, training, utils
from torchtune.data import (
    left_pad_sequence,
)

from torchtune.generation import generate as standard_generate, sample
from torchtune.modules import TransformerDecoder
from torchtune.modules.common_utils import local_kv_cache
from torchtune.modules.transforms.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform
from torchtune.recipe_interfaces import EvalRecipeInterface
from torchtune.data import InputOutputToMessages
from torchtune.training import FullModelTorchTuneCheckpointer
from entropy_guided_generation.generation import (
    generate_full_interleaved,
    generate_answer_interleaved,
    generate_answer_interleaved_marker,
)


class _LLMEvalWrapper(HFLM):
    """An EvalWrapper for EleutherAI's eval harness based on gpt-fast's
    EvalWrapper: https://github.com/pytorch-labs/gpt-fast/blob/main/eval.py.

    Note:
        This is for text-only decoder models.

    Args:
        model (TransformerDecoder): The model to evaluate.
        tokenizer (ModelTokenizer): Tokenizer associated with the model being evaluated.
            This should be the same tokenizer used when fine-tuning the model.
        device (torch.device): The device to use.
        max_seq_length (int): The maximum sequence length to use.
        batch_size (int): The batch size per GPU to use.
        dtype (torch.dtype): dtype for the model caches during generation.
        enable_kv_cache (bool): Whether to enable KV cache for generation.
    """

    def __init__(
        self,
        model: TransformerDecoder,
        tokenizer: ModelTokenizer,
        *,
        device: torch.device,
        max_seq_length: int = 4096,
        batch_size: int = 8,
        dtype: torch.dtype = torch.float32,
        enable_kv_cache: bool = True,
        new_system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        greedy_threshold: Optional[float] = None,
        dynamic_top_k: Optional[List[int]] = None,
        prob_threshold: Optional[float] = None,
        min_p: Optional[float] = None,
        top_p: Optional[float] = None,
        edt_var: Optional[List[float]] = None,
        hewitt_epsilon: Optional[float] = None,
        low_conf_log_path: Optional[str] = None,
        generate_fn=None,
    ):
        # TODO (@joecummings): Remove this init function so we don't load in extraneous stuff
        super().__init__(pretrained="gpt2", device=str(device))
        self._model = model
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._batch_size = batch_size
        self._dtype = dtype
        self._enable_kv_cache = enable_kv_cache
        self._device = device
        self.new_system_prompt = new_system_prompt
        self.temperature = temperature
        self.top_k = top_k
        self.greedy_threshold = greedy_threshold
        self.dynamic_top_k = dynamic_top_k
        self.prob_threshold = prob_threshold
        self.min_p = min_p
        self.top_p = top_p
        self.edt_var = edt_var
        self.hewitt_epsilon = hewitt_epsilon
        # Low-confidence logging state
        self._low_conf_log_path = low_conf_log_path
        self._low_conf_threshold = 0.1
        # generation function (selected by config)
        self._generate_fn = generate_fn or standard_generate

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_id

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    @property
    def enable_kv_cache(self):
        return self._enable_kv_cache

    def tok_encode(self, text: str, **kwargs) -> List[int]:
        # Note on add_bos flag: setting to False as this gives better results, for example
        # +1% on truthfulqa_mc2 with a LoRA finetune. lit-gpt also sets this to False,
        # see https://github.com/Lightning-AI/lit-gpt/blob/main/eval/lm_eval_harness.py#L66,
        # though notably fast-gpt does the opposite
        # https://github.com/pytorch-labs/gpt-fast/blob/main/eval.py#L123.
        sample = {
            "input": text,   # The user question
            "output": ""         # Empty assistant message
        }
        #do this if you want to add system prompt
        #transform = InputOutputToMessages(masking_strategy='train_on_assistant', new_system_prompt=self.new_system_prompt)
        #result = transform(sample)
        #remove the last assistant message in order to match with lm_eval's apply chat template format
        #tokens, _ = self._tokenizer.tokenize_messages(result["messages"][:-1], add_end_tokens=False)
        
        tokens = self._tokenizer.encode(text=text, add_bos=False, add_eos=False)
        return tokens

    def tok_batch_encode(
        self, text: List[str], left_truncate_len: int = None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokenized_text = [self.tok_encode(x) for x in text]
        
        # pad left
        x = left_pad_sequence(
            [torch.tensor(x) for x in tokenized_text],
            batch_first=True,
            padding_value=self._tokenizer.pad_id,
        )
        #print(f"[tok_batch_encode] Padded tensor shape: {x.shape}")

        # the harness will use left_truncate_len to indicate that the current batch
        # needs to be truncated to self.max_seq_len - self.max_gen_toks
        if left_truncate_len is not None:
            x = x[:, -left_truncate_len:]
        return x, torch.ones_like(x)  # return 'mask' b/c it's expected by the harness

    def tok_decode(self, tokens: Union[List[int], int], **kwargs) -> str:
        if isinstance(tokens, int):
            tokens = [tokens]
        #print(self._tokenizer.decode(tokens, skip_special_tokens=False))
        return self._tokenizer.decode(tokens)

    def _model_call(self, inps: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._model(inps)

    @torch.inference_mode()
    def _model_generate(
        self, context: torch.Tensor, **generation_kwargs
    ) -> torch.Tensor:
        bsz, seq_len = context.shape

        do_sample = generation_kwargs.get("do_sample", False)
        
        # if we've recieved fewer than self._batch_size samples in the current
        # batch we need to pad the batch out. here we're padding the end of the
        # current batch to the correct length. this is because when we use static
        # KV-caches, the model will expect a fixed batch size for all samples.
        maybe_padded_context = torch.nn.functional.pad(
            context,
            (0, 0, 0, self._batch_size - bsz),
            value=self._tokenizer.eos_id,  # pad with one of the tokenizer's stop tokens so generation can stop early
        )
        
        with local_kv_cache(
            self.model,
            batch_size=self.batch_size,
            device=self.device,
            dtype=self._dtype,
            decoder_max_seq_len=self.max_length,
        ):
            original_bsz = bsz
            # Collect low-confidence ranks per sequence (only for true batch rows)
            low_conf_ranks_per_seq: List[List[int]] = [[] for _ in range(bsz)]
            # Collect positions (kth generated token) per sequence
            low_conf_positions_per_seq: List[List[int]] = [[] for _ in range(bsz)]
            # Collect detokenized sampled tokens per sequence
            low_conf_tokens_per_seq: List[List[str]] = [[] for _ in range(bsz)]
            # Generation step index (0-based)
            gen_step_idx: int = 0
            # Stop-tracking control per sequence and stop token sets
            tracking_active: List[bool] = [True for _ in range(bsz)]
            #user_stop_ids = {14924, 271, 198, 382, 627} #llama3.2 1 and 3b
            user_stop_ids = {14582, 271, 198, 382, 627} #qwen2.5 7b instruct
            tokenizer_stop_ids = set(getattr(self._tokenizer, "stop_tokens", []) or [])
            stop_token_ids = user_stop_ids.union(tokenizer_stop_ids)
            # Track last 4 sampled tokens per sequence for pattern detection
            last_tokens_per_seq: List[List[int]] = [[] for _ in range(bsz)]
            # After matching a pattern, keep tracking for next 3 tokens, then disable
            post_pattern_remaining: List[int] = [-1 for _ in range(bsz)]
            #pattern_a = [791, 4320, 374, 220] #llama3.2 1 and 3b
            #pattern_b = [1820, 4320, 374, 220] #llama3.2 1 and 3b
            pattern_a = [785, 4226, 374, 220] #qwen2.5 7b instruct
            pattern_b = [1782, 4226, 374, 220] #qwen2.5 7b instruct

            def _custom_generate_next_token(
                model: TransformerDecoder,
                input_pos: torch.Tensor,
                x: torch.Tensor,
                q: Optional[torch.Tensor] = None,
                *,
                mask: Optional[torch.Tensor] = None,
                temperature: float = 1.0,
                top_k: Optional[int] = None,
                dynamic_top_k: Optional[List[int]] = None,
                greedy_threshold: Optional[float] = None,
                prob_threshold: Optional[float] = None,
                min_p: Optional[float] = None,
                top_p: Optional[float] = None,
                edt_var: Optional[List[float]] = None,
                hewitt_epsilon: Optional[float] = None,
            ):
                nonlocal gen_step_idx
                out = model(
                    x,
                    input_pos=input_pos,
                    mask=mask,
                )
                last_logits = out[:, -1]

                token = sample(
                    last_logits,
                    temperature=temperature,
                    top_k=top_k,
                    dynamic_top_k=dynamic_top_k,
                    q=q,
                    greedy_threshold=greedy_threshold,
                    prob_threshold=prob_threshold,
                    min_p=min_p,
                    top_p=top_p,
                    edt_var=edt_var,
                    hewitt_epsilon=hewitt_epsilon,
                )

                # Collect low-confidence info for active sequences only; stop further tracking after stop tokens
                with torch.no_grad():
                    probs = torch.softmax(last_logits, dim=-1)
                    max_vals, _ = torch.max(probs, dim=-1)
                    for i in range(original_bsz):
                        tok_id = token[i].item()
                        # Maintain last 4 tokens for pattern detection
                        seq_list = last_tokens_per_seq[i]
                        seq_list.append(tok_id)
                        if len(seq_list) > 4:
                            del seq_list[0]
                        # Pattern detection: if matched and not already counting down, start 3-step countdown
                        if tracking_active[i] and post_pattern_remaining[i] < 0 and len(seq_list) == 4:
                            if seq_list == pattern_a or seq_list == pattern_b:
                                post_pattern_remaining[i] = 3
                        if tracking_active[i]:
                            if max_vals[i].item() < self._low_conf_threshold:
                                sampled_logit = last_logits[i, tok_id]
                                rank = int((last_logits[i] > sampled_logit).sum().item()) + 1
                                low_conf_ranks_per_seq[i].append(rank)
                                low_conf_positions_per_seq[i].append(gen_step_idx)
                                # Detokenize the sampled token
                                try:
                                    detok = self._tokenizer.decode([tok_id])
                                except Exception:
                                    detok = ""
                                low_conf_tokens_per_seq[i].append(detok)
                        # If current token is a stop token, disable future tracking for this sequence immediately
                        if tok_id in stop_token_ids:
                            tracking_active[i] = False
                            post_pattern_remaining[i] = -1
                        # Handle post-pattern countdown: after 3 more tokens, disable tracking
                        if tracking_active[i] and post_pattern_remaining[i] >= 0:
                            post_pattern_remaining[i] -= 1
                            if post_pattern_remaining[i] == 0:
                                tracking_active[i] = False
                                post_pattern_remaining[i] = -1
                # Advance generation step index (applies to all sequences)
                gen_step_idx += 1

                return token, last_logits.unsqueeze(1)

            toks, _ = self._generate_fn(
                self.model,
                maybe_padded_context,
                max_generated_tokens=self.max_gen_toks,
                temperature=self.temperature,
                top_k=self.top_k,
                pad_id=self._tokenizer.pad_id,
                greedy_threshold=self.greedy_threshold,
                dynamic_top_k=self.dynamic_top_k,
                stop_tokens=self._tokenizer.stop_tokens,
                custom_generate_next_token=_custom_generate_next_token,
                prob_threshold=self.prob_threshold,
                min_p=self.min_p,
                top_p=self.top_p,
                edt_var=self.edt_var,
                hewitt_epsilon=self.hewitt_epsilon,
            )
        # After generation, log per-sequence low-confidence ranks
        # Decode generated tokens for actual (unpadded) sequences
        generated_texts: List[str] = []
        try:
            # Determine per-sequence prompt lengths (non-pad tokens) from the original context
            pad_id_val = self._tokenizer.pad_id
            input_lengths = (context != pad_id_val).sum(dim=1).tolist()
            for i in range(original_bsz):
                seq_ids_full = toks[i].tolist() if isinstance(toks, torch.Tensor) else toks[i]
                start_idx = int(input_lengths[i]) if i < len(input_lengths) else 0
                seq_ids = seq_ids_full[start_idx:] if start_idx < len(seq_ids_full) else seq_ids_full
                generated_texts.append(self._tokenizer.decode(seq_ids))
        except Exception:
            generated_texts = [""] * original_bsz
        # Write logs only for the actual (unpadded) sequences
        if self._low_conf_log_path is not None:
            try:
                with open(self._low_conf_log_path, "a", encoding="utf-8") as f:
                    for i in range(original_bsz):
                        f.write(json.dumps(low_conf_ranks_per_seq[i], ensure_ascii=False) + "\n")
            except Exception:
                pass
        return toks[:bsz]

class EleutherEvalRecipeDirect(EvalRecipeInterface):
    """
    This recipe runs evaluation on a trained model using EleutherAI's eval harness.
    This assumes the user has the EleutherAI eval harness installed. See
    https://github.com/EleutherAI/lm-evaluation-harness for more details.

    Features:
        - Single GPU evaluation. Multi-GPU evaluation is currently not supported.
        - Quantization (for text-only models) is supported.
        - Any task from the EleutherAI eval harness

    We recommend launching evaluation using the tune CLI::

        tune run eleuther_eval --config eleuther_evaluation \
            tasks=["truthfulqa_mc2","hellaswag"] \
            limit=50 \
    """

    def __init__(self, cfg: DictConfig) -> None:
        # Double check we have the right Eval Harness version
        from importlib.metadata import version

        if version("lm-eval") < "0.4.5":
            raise RuntimeError(
                "This recipe requires EleutherAI Eval Harness v0.4.5 or higher. "
                "Please install with `pip install lm-eval>=0.4.5`"
            )

        # General variable initialization
        self.device = utils.get_device(device=cfg.device)
        self.dtype = training.get_dtype(dtype=cfg.dtype, device=self.device)
        self.logger = utils.get_logger(cfg.get("log_level", "info"))
        training.set_seed(seed=cfg.seed)

        # Eval specific variables
        self.limit = cfg.limit
        self.tasks = list(cfg.tasks)
        self.batch_size = cfg.batch_size
        self.enable_kv_cache = cfg.get("enable_kv_cache", True)
        self.include_path = cfg.get("include_path", None)
        # Interleaving selection
        interleave = cfg.get("interleaving", {}) or {}
        self.interleaving_mode = interleave.get("mode", "none")
        self.temperature = cfg.get("temperature", None)
        self.top_k = cfg.get("top_k", None)
        self.greedy_threshold = cfg.get("greedy_threshold", None)
        self.dynamic_top_k = cfg.get("dynamic_top_k", None)
        self.prob_threshold = cfg.get("prob_threshold", None)
        self.min_p = cfg.get("min_p", None)
        self.top_p = cfg.get("top_p", None)
        self.edt_var = cfg.get("edt_var", None)
        self.hewitt_epsilon = cfg.get("hewitt_epsilon", None)
        # Persist output directory and model name for saving results
        self.output_dir = cfg.get("output_dir", ".")
        try:
            self.model_name = str(cfg.model._component_).split(".")[-1]
        except Exception:
            self.model_name = "model"

    def setup(self, cfg: DictConfig) -> None:
# Initialize quantizer and quantization mode
        quantizer = config.instantiate(cfg.quantizer)
        quantization_mode = training.get_quantizer_mode(quantizer)

        # Load checkpoint
        checkpointer = config.instantiate(cfg.checkpointer)

        # Initialize model
        with training.set_default_dtype(self.dtype), self.device:
            model = config.instantiate(cfg.model)

        # Quantize model if requested
        if quantization_mode is not None:
            if not isinstance(checkpointer, FullModelTorchTuneCheckpointer):
                raise ValueError(
                    "Quantization is only supported for models quantized and saved with the "
                    "FullModelTorchTuneCheckpointer - please ensure you have quantized your "
                    "model and are using the quantized weights!"
                )
            if "qat" in quantization_mode:
                raise ValueError(
                    "You have specified a quantizer with 'QAT' - "
                    "QAT quantizers should only be used during quantization aware training "
                    "and when quantizing models. Please use the corresponding post-training "
                    "quantizer e.g. Int8DynActInt4WeightQuantizer for Int8DynActInt4WeightQATQuantizer."
                )
            model = quantizer.quantize(model)
            model = model.to(device=self.device, dtype=self.dtype)
            ckpt_dict = checkpointer.load_checkpoint(weights_only=False)[
                training.MODEL_KEY
            ]
            for k, v in ckpt_dict.items():
                ckpt_dict[k] = v.to(self.device)
            model.load_state_dict(ckpt_dict, assign=True)
        else:
            ckpt_dict = checkpointer.load_checkpoint()[training.MODEL_KEY]
            model.load_state_dict(ckpt_dict)

        # Load model weights into initialized model
        self.logger.info(f"Model is initialized with precision {self.dtype}.")

        # Put model in eval mode.
        # Note: This will not disable the dropout applied in SDPA,
        # see https://github.com/pytorch/pytorch/issues/124464
        model.eval()
        if self.temperature is not None:
            print("Using temperature: ", self.temperature)
        if self.top_k is not None:
            print("Using top_k: ", self.top_k)
        if self.greedy_threshold is not None:
            print("Using greedy threshold: ", self.greedy_threshold)
        if self.dynamic_top_k is not None:
            print("Using dynamic top-k: ", self.dynamic_top_k)
            if len(self.dynamic_top_k) != 10:
                raise ValueError("Dynamic top-k must be a list of 10 integers")
        if self.edt_var is not None:
            print("Using edt_var: ", self.edt_var)
            if not (len(self.edt_var) == 3):
                raise ValueError("edt_var must be a list/tuple of three floats: [N, theta, T0]")
        if self.prob_threshold is not None:
            print("Using prob_threshold: ", self.prob_threshold)
        if self.min_p is not None:
            print("Using min_p: ", self.min_p)
        if self.top_p is not None:
            print("Using top_p: ", self.top_p)
        if self.hewitt_epsilon is not None:
            print("Using hewitt_epsilon: ", self.hewitt_epsilon)

        # Initialize tokenizer/transform
        model_transform = config.instantiate(cfg.tokenizer)

        if isinstance(model, TransformerDecoder):
            eleuther_model_wrapper = _LLMEvalWrapper
        # Prepare low-confidence log path (timestamped per run)
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception:
            pass
        _now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        low_conf_log_path = os.path.join(self.output_dir, f"low_conf_ranks_{_now_str}.jsonl")
        self.logger.info(f"Low-confidence log path: {low_conf_log_path}")
        # Select generation function based on interleaving.mode
        gen_mode = (self.interleaving_mode or "none").lower()
        if gen_mode == "full":
            selected_generate = generate_full_interleaved
        elif gen_mode == "answer":
            selected_generate = generate_answer_interleaved
        elif gen_mode == "marker":
            selected_generate = generate_answer_interleaved_marker
        else:
            selected_generate = standard_generate

        self.eleuther_model_wrapper = eleuther_model_wrapper(
            model,
            model_transform,
            device=self.device,
            max_seq_length=cfg.max_seq_length,
            batch_size=self.batch_size,
            dtype=self.dtype,
            enable_kv_cache=self.enable_kv_cache,
            new_system_prompt=cfg.get('new_system_prompt', None),
            temperature=self.temperature,
            top_k=self.top_k,
            greedy_threshold=self.greedy_threshold,
            dynamic_top_k=self.dynamic_top_k,
            prob_threshold=self.prob_threshold,
            min_p=self.min_p,
            top_p=self.top_p,
            edt_var=self.edt_var,
            hewitt_epsilon=self.hewitt_epsilon,
            low_conf_log_path=low_conf_log_path,
            generate_fn=selected_generate,
        )
       

    def evaluate(self) -> Dict:
        # Initialize tasks for the harness
        task_manager = TaskManager(include_path=self.include_path)
        task_dict = get_task_dict(self.tasks, task_manager)

        # Run evaluation
        t0 = time.time()
        self.logger.info(f"Running evaluation on the following tasks: {self.tasks}")
        output = evaluate(
            self.eleuther_model_wrapper,
            task_dict,
            limit=self.limit,
            write_out=True,
            confirm_run_unsafe_code=True #for coding tasks
        )
        t1 = time.time() - t0

        # Log metrics
        self.logger.info(f"Eval completed in {t1:.02f} seconds.")
        self.logger.info(
            f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB"
        )
        formatted_output = make_table(output)
        self.logger.info(f"\n\n{formatted_output}\n")
        # Save raw evaluation output to JSON with model name and timestamp
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            output_path = os.path.join(self.output_dir, f"{self.model_name}_{now_str}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False, default=str)
            self.logger.info(f"Saved evaluation output to {output_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save evaluation output: {e}")
        
        return output


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """Entry point for the recipe."""
    config.log_config(recipe_name="EleutherEvalRecipeDirect", cfg=cfg)
    recipe = EleutherEvalRecipeDirect(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.evaluate()


if __name__ == "__main__":
    sys.exit(recipe_main())
