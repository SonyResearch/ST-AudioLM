# Testing

## Unit tests

These tests use temporary files and small tensors. No model downloads, GPUs,
or evaluation datasets are required.

```bash
python -m unittest discover -s tests -v
```

## Model smoke test

Given a standalone Encoder bundle, test the real encoder with generated noise:

```bash
python scripts/smoke_test.py \
  --encoder-model ./weights/ST-AudioLM \
  --device cpu
```

Run the complete model on a recording:

```bash
python scripts/smoke_test.py \
  --model ./weights/ST-AudioLM \
  --level full --device cuda \
  --audio example.wav \
  --question "Where is the sound source?"
```

The script verifies tensor shapes, finite values, and non-empty generation.
Omitting `--audio` uses synthetic noise and checks execution only.
Use `--base-model models/olmo2 --local-files-only` with a local base model.

## RIR and reconstruction tests

The unit suite verifies trajectory interpolation, RIR packing, dynamic convolution,
channel conversion, and exact reconstruction against a fixed fixture. Habitat-Sim
itself is optional and is not loaded by unit tests.

For a release candidate, reconstruct the fixed test recording with its released
exact RIR and compare it to the trusted precomputed waveform. Regenerated RIRs are
checked for four channels, finite values, non-zero energy, and the expected number
of keyframes; they are not required to be bit-exact across Habitat-Sim builds.

## Prediction comparison

Prepare a JSONL manifest with one object per line:

```json
{"id":"clip-001","audio":"audio/clip-001.wav","question":"Where is the sound source?","prediction":"left"}
```

The `prediction` field is the saved model output to reproduce, not a
ground-truth answer. Audio paths must be relative to the manifest's directory.
By default, clips are loaded as AmbiX. For an existing precomputed NPY tensor,
set `"input_format":"legacy_model"` and, for a shard, `"npy_index":0`.

```bash
python scripts/regression_test.py \
  --model ./weights/ST-AudioLM \
  --manifest evaluation/samples.jsonl \
  --num-samples 100 --batch-size 16 \
  --device cuda --output comparison.json
```

Generation uses greedy decoding, 24 new tokens, and a 256-token text limit.
Match the original batch size and numerical environment when comparing
predictions. The script reports all differences after trimming leading and
trailing whitespace and exits with a nonzero status if any output differs.
