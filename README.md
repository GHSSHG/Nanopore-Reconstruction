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
  utils/      File discovery, random seeds, and W&B logging
configs/
  offline-high.json
train.py
validate.py
requirements.txt
```

Only `configs/offline-high.json` is the retained training configuration. Earlier experimental configs, post-training scripts, and internal analysis outputs are not part of the public reproduction path.

## Environment

Create an environment with Python 3.11 or 3.12, then install the requirements:

```bash
pip install -r requirements.txt
```

`requirements.txt` specifies the CUDA 12 GPU JAX package (`jax[cuda12]`). If your CUDA/JAX setup is different, adjust that line according to the official JAX installation instructions:

<https://docs.jax.dev/en/latest/installation.html>

The config enables Weights & Biases logging (`logging.wandb.enabled`). Install `wandb` and log in
(`wandb login`), or set `logging.wandb.enabled` to `false`.

## Dataset

The training config expects POD5 data from Oxford Nanopore's hereditary cancer example dataset, arranged as:

```text
hereditary_cancer_2025.09/
  raw/
    FC01/
      pod5/
        *.pod5
    FC02/
      pod5/
    FC03/
      pod5/
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

The loader reads the POD5 files of `FC01/pod5`, `FC02/pod5` and `FC03/pod5` under `${DATA_PATH}/hereditary_cancer_2025.09/raw` (one epoch over all three flow cells, files in a seeded random order).

The dataset files themselves are not included in this repository.

## Quantization

The quantizer always outputs the nearest projected codeword (exact nearest-neighbour search,
the same fused GPU kernel as the NanoRecon app), so training and inference run the same
forward pass; there is no `train` flag. It is trained as a classic VQ-VAE: gradients pass
straight through the quantizer to the encoder, the SimVQ projection (`W`, `proj_bias`; the base
codebook stays frozen) is trained by the codebook loss `||q - sg(z)||^2`, and the commitment loss
`||z - sg(q)||^2` keeps the encoder output near its codewords (weights `train.loss_weights.vq_codebook`
and `vq_commitment`, 1.0 and 0.25).

Models trained by this code are run with the NanoRecon app. The weights released before this
change (<https://huggingface.co/GHSSHG/NanoRecon>) were trained with noisy DiVeQ gradients
(`diveq_sigma2=1e-3`) and their `config.json` is not accepted by this code.

## Decoder

The decoder is Vocos-style: a ConvNeXt backbone (with a small attention block in front) predicts
the complex spectrum of each latent frame (`istft_n_fft // 2 + 1` bins, real and imaginary parts),
and an inverse STFT with `istft_hop_length` = 16 samples, one latent frame, turns it into the
waveform (`istft_n_fft` 64, a 4x overlap).

## Losses

`train.loss_weights` (reconstruction losses on chunks normalized to [-1, 1]):

- `pa_l1`: L1 in picoamps (error × chunk half-range / `train.loss.pa_l1_scale`); mostly the low
  frequencies, which carry almost all of the signal energy.
- `diff1_l1`, `diff2_l1`: L1 of the first and second differences (edges and their shape; weighted
  toward high frequencies).
- `band_l1_500_1000hz`, `band_l1_1000_2500hz`: time-domain L1 of the 500-1000 Hz and 1000-2500 Hz
  components, so the high-frequency waveform has to line up with the original.
- `stft_logmag_500_1000hz`, `stft_logmag_1000_2500hz`: L1 of log(1 + |STFT|) over the bins of the same
  two bands, averaged over the `train.stft_loss.scales`; keeps the high-frequency energy at the
  original level instead of shrinking it.
- `vq_codebook`, `vq_commitment`: see Quantization.

## Validation

With `validation.enabled`, every checkpoint is validated in the background on GPU
`validation.gpu` (shared with training): `validate.py` reconstructs the held-out POD5
`validation.source_pod5` from codewords, basecalls it with Dorado and reports the identity and
empirical Q against `validation.truth_fastq` (overall and per flow cell), the ADC MAE, and per
frequency band the amplitude of the correctly reconstructed part and the energy of the wrong part.
Results go to `<checkpoint dir>/validation/step_<N>/summary.json`, one line per checkpoint in
`<checkpoint dir>/validation/summary.jsonl`. It takes about 5-8 minutes. Run it by hand with

```bash
python validate.py --config configs/offline-high.json \
  --checkpoint checkpoints/<run>/periodic/checkpoint_<step> --out /tmp/val
```

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
  --lr 1e-4 \
  --ckpt-dir checkpoints/test-run \
  --log-every-steps 50
```

Resume with `--resume checkpoints/<run>/periodic/checkpoint_<step>`.

Checkpoints are written under `checkpoint.root_dir/<run name>` (default `checkpoints/offline-high-vocos-pa1`),
which is ignored by Git. Training refuses to start in a non-empty checkpoint directory unless it resumes.

## Configuration Notes

- `train.batch_size` is the global batch, split evenly over all visible GPUs (data parallel);
  choose GPUs with `CUDA_VISIBLE_DEVICES`.
- Learning rate: `train.learning_rate` (1e-4, as in SimVQ) is reached by a linear warmup over
  `train.warmup_steps` steps and then held constant.
- Every `train.log_every_steps` steps the log reports the window means of the losses, the
  quantization distance `q_z_dist`, the mean latent and codeword norms (`z_norm`, `code_norm`,
  which should be of similar size), and the codebook perplexity and utilization over all codes
  chosen in the window.
- Data: files are visited in a seeded random order each epoch, `data.files_in_flight` files at a
  time, and windows pass through a shuffle buffer of `data.shuffle_buffer` chunks; one background
  thread prepares the batches.
- W&B logging is on in the config (`logging.wandb`); the run name also names the checkpoint directory.
- Do not commit credentials or machine-specific dataset roots.
- The model is trained on 8192-sample chunks at 5000 Hz.

## Citation

Citation metadata has not been provided yet. If you use NanoRecon in a paper or report, cite the model repository:

```text
GHSSHG/NanoRecon. Hugging Face model repository.
https://huggingface.co/GHSSHG/NanoRecon
```
