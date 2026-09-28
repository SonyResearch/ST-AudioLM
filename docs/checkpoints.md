# Model files and offline use

ST-Audio Encoder and ST-AudioLM are separate components in one Hugging Face repository. The Encoder remains independently reusable, while the full model loads both subdirectories from the same snapshot.

```text
ST-AudioLM/
├── encoder/
│   ├── config.json
│   └── encoder.safetensors
└── model/
    ├── config.json
    ├── connector.safetensors
    ├── adapter/
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    └── tokenizer/
```

Load only the Encoder from the combined repository:

```python
from staudiolm import STAudioEncoder

encoder = STAudioEncoder.from_pretrained("HBoh/ST-AudioLM", device="cuda")
```

For offline full-model use, download the combined repository and OLMo-2, then pass their local directories:

```python
from staudiolm import STAudioLM

model = STAudioLM.from_pretrained(
    "./weights/ST-AudioLM",
    base_model="./weights/olmo2",
    local_files_only=True,
    device="cuda",
)
```

The optional `encoder_model` argument can still override the bundled Encoder. The Encoder emits 40 temporal tokens followed by one semantic token. The direct connector maps each 768-dimensional token to the LLM's 4096-dimensional hidden space. Loaders validate the release configuration, require strict state-dictionary matches, and read JSON and safetensors only.
