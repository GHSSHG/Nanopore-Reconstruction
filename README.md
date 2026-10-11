# NanoRecon

NanoRecon is a Flax/JAX implementation of the SimVQ nanopore signal reconstruction model. The public repository is intentionally reduced to the model code, one training configuration, and one training entrypoint.

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

## Quantization

The quantizer always outputs the nearest projected codeword (exact nearest-neighbour search,
the same fused GPU kernel as the NanoRecon app). Training uses deterministic DiVeQ gradients
without noise, so training and inference run the same forward pass; there is no `train` flag.

Models trained by this code are run with the NanoRecon app. The weights released before this
change (<https://huggingface.co/GHSSHG/NanoRecon>) were trained with DiVeQ noise
(`diveq_sigma2=1e-3`) and their `config.json` is not accepted by this code.

## Model Interface

```python
import jax
import jax.numpy as jnp

from codec.models import build_audio_model

model = build_audio_model(cfg["model"])
variables = {"params": ..., "vq": ...}  # from a training checkpoint
x = jnp.zeros((batch, 8192), dtype=jnp.float32)  # chunks normalized per chunk to [-1, 1]
out = model.apply(variables, x)
out["wave_hat"], out["enc"]["indices"]
```

The codebook search is a Pallas/Triton kernel and needs a CUDA GPU.

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

Resume with `--resume checkpoints/<run>/periodic/checkpoint_<step>`.

Checkpoints are written under `checkpoint.root_dir/<run name>` (default `checkpoints/offline-high-hard`),
which is ignored by Git. Training refuses to start in a non-empty checkpoint directory unless it resumes.

## Configuration Notes

- `train.batch_size` is the global batch, split evenly over all visible GPUs (data parallel);
  choose GPUs with `CUDA_VISIBLE_DEVICES`.
- Data: files are visited in a seeded random order each epoch, `data.files_in_flight` files at a
  time, and windows pass through a shuffle buffer of `data.shuffle_buffer` chunks; one background
  thread prepares the batches.
- The public config defaults to `logging.wandb.enabled=false`; enable W&B with `--wandb`.
- Do not commit credentials or machine-specific dataset roots.
- The model is trained on 8192-sample chunks at 5000 Hz.

## Citation

Citation metadata has not been provided yet. If you use NanoRecon in a paper or report, cite the model repository:

```text
GHSSHG/NanoRecon. Hugging Face model repository.
https://huggingface.co/GHSSHG/NanoRecon
```
