# ST-AudioQA data and reconstruction

ST-AudioQA is released as annotations and reconstruction assets rather than redistributed source waveforms. The original AudioSet audio and Matterport3D scene files remain with their respective providers.

Install the data utilities with `python -m pip install -e ".[data]"`.

## Benchmark contents

| Split | Recordings | Type A | Type B | Type C | All QA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 40,000 | 216,437 | 300,000 | 217,965 | 734,402 |
| Validation | 4,000 | 21,608 | 29,966 | 21,891 | 73,465 |
| Test | 10,000 | 54,181 | 74,974 | 54,679 | 183,834 |

## Hugging Face configurations

| Configuration | Contents |
| --- | --- |
| `qa` | Questions, answers, types, families, and target metadata |
| `recordings` | One row per virtual recording and links to its sources |
| `sources` | AudioSet IDs, YouTube IDs, time segments, labels, and gains |
| `rir_recipes` | Listener geometry, trajectories, and RIR render parameters |
| `rir_test_index` | Test RIR IDs, shard names, tar member paths, and sizes |

The repository also contains 1,958 deduplicated exact test RIRs in 28 uncompressed tar shards. The complete dataset repository is approximately 58 GB, but a single RIR requires downloading only its indexed shard. A single RIR can be referenced by multiple recordings.

```text
ST-AudioQA/
├── qa/{train,validation,test}.parquet
├── recordings/{train,validation,test}.parquet
├── sources/{train,validation,test}.parquet
├── rir/
│   ├── recipes/{train,validation,test}.parquet
│   └── test/
│       ├── index.parquet
│       └── test-rirs-*.tar
├── release_manifest.json
└── SHA256SUMS
```

## Reconstructing a recording

1. Find an `audio_id` in the `recordings` table.
2. Read its rows from the `sources` table.
3. Acquire the referenced AudioSet segments from their original provider. The reconstruction CLI converts them to mono 32-kHz audio with the original resampling procedure.
4. Obtain each `rir_id`: extract the exact released RIR for test, or render the recipe for train and validation.
5. Run `scripts/reconstruct_audio.py`, passing one `SOURCE_ID=PATH` and `RIR_ID=PATH` argument per source.

For a test recording, obtain each `rir_id` from `sources/test.parquet` and extract it directly from Hugging Face:

```bash
python scripts/extract_test_rir.py 6fbce3a3a725a435ce1090e3 \
  --output rirs/6fbce3a3a725a435ce1090e3.npz
```

The helper downloads `rir/test/index.parquet`, selects the one required tar shard, verifies the indexed member size, and extracts only the requested NPZ. To use an already downloaded dataset, add `--dataset-root ./ST-AudioQA`.

```bash
python scripts/reconstruct_audio.py \
  --dataset-root ./ST-AudioQA \
  --split test \
  --audio-id stqa_test_000000 \
  --source-audio SOURCE_ID=/path/to/source.flac \
  --rir RIR_ID=/path/to/rirs.npz \
  --output example.wav
```

The output is a 10-second, 32-kHz, four-channel float WAV using AmbiX ACN/SN3D `[W, Y, Z, X]`.

## Exact and regenerated RIRs

The exact test archives contain the original RIR arrays used to build the benchmark. They are the reproducible evaluation path.

Train and validation RIRs are distributed as recipes. `scripts/render_rirs.py` implements the original renderer, but Habitat-Sim acoustic output can vary across builds. Regenerated RIRs must therefore pass structural and signal checks, but they are not claimed to be bit-identical to the released test RIRs.

### Pinned renderer recipe

The release recipe records the complete public renderer environment in `assets/rir_environment.json`:

- SoundSpaces commit `287184fd7067a0385558492716355c54875500ee`;
- Habitat-Sim commit `4f61e321477708fa606fbd8f42b4bef41d67c672`;
- RLR Audio Propagation commit `4fd446b4abb5c71fb7a232a083bbddd65f25fc6f`;
- sample rate 32 kHz, four-channel ambisonic output, indirect propagation enabled; and
- SHA-256-pinned MP3D scene and acoustic-material configurations.

The renderer requires Linux and a source build of Habitat-Sim with its audio sensor. The following commands reproduce the pinned simulator checkout used by this release:

```bash
conda create -n staudiolm-rir python=3.9 cmake=3.14.0 -y
conda activate staudiolm-rir

python -m pip install numpy pyarrow

git clone https://github.com/facebookresearch/habitat-sim.git
git -C habitat-sim checkout 4f61e321477708fa606fbd8f42b4bef41d67c672
git -C habitat-sim submodule update --init --recursive
(cd habitat-sim && python setup.py install --headless --audio)
```

This renderer environment is separate from the Python 3.10+ model-inference environment. The renderer scripts run directly from the cloned ST-AudioLM source tree and therefore do not require installing the `st-audiolm` package into the Python 3.9 environment.

The pinned Habitat-Sim checkout includes RLR Audio Propagation commit `4fd446b4abb5c71fb7a232a083bbddd65f25fc6f` as a submodule. The SoundSpaces commit above records the upstream recipe provenance; `scripts/render_rirs.py` calls the Habitat-Sim audio API directly and does not require the SoundSpaces Python package. Obtain the Matterport3D assets under their original terms.

Create the two renderer configurations and verify their hashes with:

```bash
python scripts/setup_rir_renderer.py --output ./rir_renderer_assets
```

First render a single recipe as an installation check:

```bash
python scripts/render_rirs.py \
  --recipes ./ST-AudioQA/rir/recipes/train.parquet \
  --output ./rir_smoke_test \
  --mp3d-root /path/to/mp3d \
  --scene-dataset-config ./rir_renderer_assets/mp3d.scene_dataset_config.json \
  --materials-json ./rir_renderer_assets/mp3d_material_config.json \
  --max-items 1
```

Remove `--max-items 1` and choose a new output directory to render the complete split:

```bash
python scripts/render_rirs.py \
  --recipes ./ST-AudioQA/rir/recipes/train.parquet \
  --output ./rendered_train_rirs \
  --mp3d-root /path/to/mp3d \
  --scene-dataset-config ./rir_renderer_assets/mp3d.scene_dataset_config.json \
  --materials-json ./rir_renderer_assets/mp3d_material_config.json
```

The renderer rejects modified scene or material configs by SHA-256 and refuses to overwrite an existing output directory. Matterport3D assets themselves are not downloaded by the setup script.

## Release boundary

- Distributed: QA annotations, recording/source metadata, AudioSet references, all RIR recipes, exact test RIRs.
- Not distributed: AudioSet/YouTube waveforms, Matterport3D scene assets, precomputed train/validation waveforms, and the full train/validation RIR corpus.

ST-AudioQA annotations and released RIR assets are provided under CC BY-NC-SA 3.0 US and remain subject to the Matterport3D Terms of Use where applicable. Users are also responsible for complying with AudioSet, source-video, Matterport3D, Habitat-Sim, and other upstream terms.

## Limitations

- Referenced AudioSet source videos can become unavailable or change over time.
- Event labels inherit AudioSet ontology coverage and annotation noise.
- Scenes are simulated from Matterport3D environments and do not represent every real acoustic condition.
- Train and validation RIRs regenerated with different simulator builds may not be numerically identical.
