## Confidence-as-Context: \\Persistent Uncertainty Signals for Autoregressive Generation

Overview

- This repo contains minimal add-ons for interleaving confidence tokens during training and inference bootstrapped to Torchtune.
- You train with provided finetune recipes and decode with matching generation functions.

Requirements

- Python 3.10+
- PyTorch compatible with your CUDA setup
- torchtune installed (pip or from source)

Install

- Ensure torchtune is installed and importable.
- Place this `entropy_guided_generation` folder anywhere on your PYTHONPATH or install it as an editable package.

Folder structure (relevant)

- generation/
  - __init__.py: exports three decode variants
  - common.py: shared discretization and helper utilities
  - _generation_full_intl.py: full interleaving generator
  - _generation_ans_intl.py: answer-only interleaving generator
  - _generation_ans_intl_mistake.py: marker-based reasoning interleave variant
- full_finetune_distributed_*.py: distributed training recipes matching the three variants
- full_finetune_single_device_ans_intl.py: single-device variant (answer-only)
- entropy_intl/*.yaml and bigbench_mistake/*.yaml: example configs

Which variant to use

- Full interleaving (train and infer with alternating entropy tokens everywhere):
  - Train: full_finetune_distributed_full_intl.py
  - Infer: generation.generate_full_interleaved

- Answer-only interleaving (question untouched, answer tokens interleaved with confidence):
  - Train: full_finetune_distributed_ans_intl.py (or single-device variant)
  - Infer: generation.generate_answer_interleaved

- Marker-based reasoning interleave + plain answer:
  - Train: full_finetune_distributed_ans_intl_mistake.py
  - Infer: generation.generate_answer_interleaved_marker

Training (examples)

- Multi-GPU (FSDP/TP depends on torchtune config):
  - tune run --nproc_per_node 4 entropy_guided_generation/full_finetune_distributed_ans_intl.py --config entropy_guided_generation/entropy_intl/8B_distributed_intl.yaml

- Single GPU (answer-only):
  - python -m entropy_guided_generation.full_finetune_single_device_ans_intl --config entropy_guided_generation/entropy_intl/1B_full_single_device_intl.yaml

- Configs point to HF checkpoints and tokenizer paths; adjust paths for your environment.

Inference (with Eleuther eval)

```python
import torch
from torchtune import config, training

from entropy_guided_generation.eleuther_eval import EleutherEvalRecipeDirect

# YAML example: set interleaving mode in your eval config
# interleaving:
#   mode: full   # options: none | full | answer | marker

cfg = config.load_yaml("entropy_guided_generation/entropy_intl/gsm8k_eval.yaml")
recipe = EleutherEvalRecipeDirect(cfg)
recipe.setup(cfg)
recipe.evaluate()
```

Interleaving selection via YAML

- In your eval YAML (e.g., `entropy_intl/gsm8k_eval.yaml`), set:

```yaml
interleaving:
  mode: answer  # none | full | answer | marker
```

- The evaluation script will automatically select:
  - `none`   -> standard Torchtune generation
  - `full`   -> `entropy_guided_generation.generation.generate_full_interleaved`
  - `answer` -> `entropy_guided_generation.generation.generate_answer_interleaved`
  - `marker` -> `entropy_guided_generation.generation.generate_answer_interleaved_marker`


