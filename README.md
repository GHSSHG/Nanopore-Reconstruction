# NanoRecon

NanoRecon is a Flax/JAX implementation of the trained SimVQ nanopore signal reconstruction model used in this repository. The public repository is intentionally reduced to the model code, one training configuration, and one training entrypoint.

Trained inference weights are available on Hugging Face:

<https://huggingface.co/GHSSHG/NanoRecon>

## What Is Included

```text
codec/
  data/       POD5 loading, normalization, and prefetching
  jaxlayers/  Small Flax/JAX layer helpers
  models/     Encoder, decoder, quantizer, and model factory
  train/      Reconstruction losses, train state, train step, and loop
  utils/      File discovery, random seeds, and optional W&B logging
configs/
  offline-high.json
train.py
requirements.txt
```

Only `configs/offline-high.json` is the retained training configuration. Earlier experimental configs, validation scripts, post-training scripts, and internal analysis outputs are not part of the public reproduction path.

## Environment

Create an environment with Python 3.11 or 3.12, then install the requirements:

```bash
pip install -r requirements.txt
```

`requirements.txt` specifies the CUDA 12 GPU JAX package (`jax[cuda12]`). If your CUDA/JAX setup is different, adjust that line according to the official JAX installation instructions:

<https://docs.jax.dev/en/latest/installation.html>

Weights & Biases support is optional. The public config disables W&B by default.
Install `wandb` separately only if you enable W&B logging.

## Dataset

The training config expects POD5 data from Oxford Nanopore's hereditary cancer example dataset, arranged as:

```text
hereditary_cancer_2025.09/
  raw/
    FC01/
      pod5/
        *.pod5
```

Dataset source:

<https://epi2me.nanoporetech.com/hereditary_cancer_2025.09/>

The committed config uses an environment-variable placeholder:

```json
"root": "${DATA_PATH}/hereditary_cancer_2025.09/raw"
```

Set `DATA_PATH` to the directory that contains `hereditary_cancer_2025.09`, then run training:

```bash
export DATA_PATH=/path/to/datasets
python train.py --config configs/offline-high.json
```

For example, the loader will read POD5 files from `${DATA_PATH}/hereditary_cancer_2025.09/raw/FC01/pod5`. The actual dataset subpath `FC01/pod5` is preserved because it is part of the dataset layout.

The dataset files themselves are not included in this repository.

## Download The Released Model

Download the inference-ready weights from the Hugging Face model page:

<https://huggingface.co/GHSSHG/NanoRecon/tree/main>

Place the downloaded files under `checkpoints/NanoRecon/` if you want to run the inference example:

```text
checkpoints/
  NanoRecon/
    config.json
    flax_model.msgpack
```

Alternatively, if you already have the Hugging Face CLI installed, you can run:

```bash
huggingface-cli download GHSSHG/NanoRecon --local-dir checkpoints/NanoRecon
```

The model repository contains:

- `flax_model.msgpack`: Flax variables for inference.
- `config.json`: model architecture and input signal metadata.

The released model expects normalized signal chunks with shape `(batch, samples)`.

Key settings:

- sample rate: `5000 Hz`
- segment length: `8192` samples
- segment duration: `1.6384 s`
- model variant: `foundation_v1`
- codebook size: `65536`

## Minimal Inference Example

```python
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from flax.core import freeze
from flax.serialization import msgpack_restore

from codec.models import build_audio_model

repo_dir = Path("checkpoints/NanoRecon")
cfg = json.loads((repo_dir / "config.json").read_text())

model = build_audio_model(cfg["model"])
variables = freeze(msgpack_restore((repo_dir / "flax_model.msgpack").read_bytes()))

x = jnp.zeros((1, cfg["input_signal"]["segment_samples"]), dtype=jnp.float32)
rng = jax.random.PRNGKey(0)

out = model.apply(
    variables,
    x,
    train=False,
    offset=0,
    rng=rng,
    collect_codebook_stats=False,
)

reconstructed = out["wave_hat"]
print(reconstructed.shape)
```

In real use, replace the zero input with normalized nanopore signal chunks prepared with the same normalization mode used for training.

## Training

Run training with the retained config:

```bash
python train.py --config configs/offline-high.json
```

Short smoke test:

```bash
python train.py \
  --config configs/offline-high.json \
  --max-steps 100 \
  --max-steps-per-epoch 100
```

Common overrides:

```bash
python train.py \
  --config configs/offline-high.json \
  --batch-size 64 \
  --lr 5e-6 \
  --ckpt-dir checkpoints/test-run \
  --log-every-steps 50
```

Checkpoints are written under `checkpoints/` by default and are ignored by Git.

## Configuration Notes

- The public config defaults to `logging.wandb.enabled=false`.
- If you use W&B, enable it explicitly with `--wandb` or edit the config locally.
- Do not commit credentials or machine-specific dataset roots.
- Multi-GPU behavior is controlled by `train.data_parallel`, `train.per_device_batch_size`, and the device-count scaling keys in the config.
- The retained model was trained with 8192-sample chunks at 5000 Hz.

## Citation

Citation metadata has not been provided yet. If you use NanoRecon in a paper or report, cite the model repository:

```text
GHSSHG/NanoRecon. Hugging Face model repository.
https://huggingface.co/GHSSHG/NanoRecon
```
