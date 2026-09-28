"""Final-only ST-AudioLM model and generation path."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from safetensors.torch import load_file

from .checkpoint import validate_lm_bundle
from .encoder_model import STAudioEncoder

AUDIO_TOKEN = "<audio>"
NUM_AUDIO_TOKENS = 41


class AudioDirectConnector(nn.Module):
    """Project each encoder token independently into the LLM hidden space."""

    def __init__(self, encoder_dim: int = 768, llm_dim: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(tokens)


def format_chat(tokenizer, user_text: str) -> str:
    """Reproduce the no-thinking inference prompt used for final evaluation."""

    message = [{"role": "user", "content": user_text}]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    return f"User: {user_text}\nAssistant:"


class STAudioLM(nn.Module):
    """Frozen dynamic FOA encoder + direct connector + LoRA-adapted OLMo-2."""

    def __init__(
        self,
        encoder: STAudioEncoder,
        connector: nn.Module,
        llm: nn.Module,
        tokenizer,
    ):
        super().__init__()
        self.encoder = encoder
        self.audio_connector = connector
        self.llm = llm
        self.tokenizer = tokenizer
        if AUDIO_TOKEN not in tokenizer.get_vocab():
            raise RuntimeError(f"Tokenizer is missing {AUDIO_TOKEN!r}")
        self.audio_token_id = int(tokenizer.convert_tokens_to_ids(AUDIO_TOKEN))

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str | Path,
        *,
        encoder_model: str | Path | None = None,
        base_model: str | Path | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        local_files_only: bool = False,
    ) -> "STAudioLM":
        """Load ST-AudioLM, its bundled encoder, and the external base LLM."""
        root, config = validate_lm_bundle(
            model_id_or_path, local_files_only=local_files_only
        )
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers and peft are required for full-model inference") from exc

        model_path = str(base_model or config["base_model"])
        tokenizer = AutoTokenizer.from_pretrained(
            root / "tokenizer", local_files_only=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
        )
        if len(tokenizer) != llm.get_input_embeddings().weight.shape[0]:
            llm.resize_token_embeddings(len(tokenizer))
        llm = PeftModel.from_pretrained(llm, root / "adapter", is_trainable=False)
        hidden_size = int(llm.config.hidden_size)
        if hidden_size != 4096:
            raise RuntimeError(f"Expected OLMo-2 hidden size 4096, got {hidden_size}")
        if encoder_model is not None:
            encoder_source = encoder_model
        else:
            candidate = Path(str(model_id_or_path)).expanduser()
            encoder_source = (
                model_id_or_path
                if not candidate.exists() or (candidate / "encoder").is_dir()
                else config["encoder_model"]
            )
        encoder = STAudioEncoder.from_pretrained(
            encoder_source,
            device="cpu",
            local_files_only=local_files_only,
        )
        connector = AudioDirectConnector(llm_dim=hidden_size)
        connector_state = load_file(str(root / "connector.safetensors"), device="cpu")
        connector.load_state_dict(connector_state, strict=True)
        model = cls(encoder, connector, llm, tokenizer)
        model.encoder.requires_grad_(False).eval()
        model.audio_connector.requires_grad_(False).eval()
        model.llm.requires_grad_(False).eval()
        model.to_device(device)
        model.eval()
        return model

    def to_device(self, device: str | torch.device) -> "STAudioLM":
        device = torch.device(device)
        self.encoder.to_device(device)
        self.audio_connector.to(device=device, dtype=torch.float32)
        self.llm.to(device=device)
        return self

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    @torch.inference_mode()
    def encode_audio(self, foa_waveforms: torch.Tensor) -> torch.Tensor:
        if foa_waveforms.ndim == 2:
            foa_waveforms = foa_waveforms.unsqueeze(0)
        tokens = self.encoder.encode(foa_waveforms)
        if tokens.shape[1:] != (NUM_AUDIO_TOKENS, 768):
            raise RuntimeError(f"Expected encoder tokens [B, 41, 768], got {tuple(tokens.shape)}")
        return self.audio_connector(tokens)

    def _merge_audio_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text = self.llm.get_input_embeddings()(input_ids)
        audio_embeddings = audio_embeddings.to(device=text.device, dtype=text.dtype)
        merged, masks = [], []
        for index in range(input_ids.shape[0]):
            positions = (input_ids[index] == self.audio_token_id).nonzero(as_tuple=False).flatten()
            if positions.numel() != 1:
                raise ValueError(
                    f"Each question must contain exactly one {AUDIO_TOKEN}; sample {index} has {positions.numel()}"
                )
            pos = int(positions.item())
            merged.append(torch.cat((text[index, :pos], audio_embeddings[index], text[index, pos + 1 :]), 0))
            masks.append(
                torch.cat(
                    (
                        attention_mask[index, :pos],
                        attention_mask.new_ones(NUM_AUDIO_TOKENS),
                        attention_mask[index, pos + 1 :],
                    ),
                    0,
                )
            )
        return torch.stack(merged), torch.stack(masks)

    @torch.inference_mode()
    def generate(
        self,
        foa_waveforms: torch.Tensor,
        questions: str | Sequence[str],
        *,
        max_new_tokens: int = 24,
        max_seq_length: int = 256,
    ) -> list[str]:
        if isinstance(questions, str):
            questions = [questions]
        if foa_waveforms.ndim == 2:
            foa_waveforms = foa_waveforms.unsqueeze(0)
        if foa_waveforms.shape[0] != len(questions):
            raise ValueError("Audio batch size and question count differ")
        user_texts = [q if AUDIO_TOKEN in q else f"{AUDIO_TOKEN}\n{q}" for q in questions]
        prompts = [format_chat(self.tokenizer, text) for text in user_texts]
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            batch = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
                add_special_tokens=False,
            )
        finally:
            self.tokenizer.padding_side = old_padding_side
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()
        with autocast:
            audio_embeddings = self.encode_audio(foa_waveforms)
            inputs_embeds, merged_mask = self._merge_audio_embeddings(
                input_ids, attention_mask, audio_embeddings
            )
            generated = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=merged_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return [text.strip() for text in self.tokenizer.batch_decode(generated, skip_special_tokens=True)]
