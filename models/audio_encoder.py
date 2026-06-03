"""
AudioEncoder: raw wav/fbank -> speaker embedding via wespeaker models.

Supports:
  - w2vbert: raw wav -> W2V-BERT SSL frontend -> speaker model -> embedding
  - resnet:  fbank features -> ResNet speaker model -> embedding
             (fbank extraction is done in E2EDataset)
"""

import os
import sys

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# wespeaker import — try package first, then project-relative paths
# ---------------------------------------------------------------------------
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_parent_root = os.path.abspath(os.path.join(_project_root, ".."))
for _p in (_project_root, _parent_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from wespeaker.frontend import frontend_class_dict
    from wespeaker.models.speaker_model import get_speaker_model
    from wespeaker.utils.checkpoint import load_checkpoint as _ws_load_ckpt
    _HAS_WESPEAKER = True
except ImportError:
    _HAS_WESPEAKER = False


# ---------------------------------------------------------------------------
# Default model configs (matching train/conf/*.yaml)
# ---------------------------------------------------------------------------
_W2VBERT_MODEL_ARGS = dict(
    feat_dim=1024,
    embed_dim=256,
    pooling_func="ASP",
    n_mfa_layers=-1,
    adapter_dim=128,
    dropout=0.0,
    num_frontend_hidden_layers=24,
)

_W2VBERT_FRONTEND_ARGS = dict(
    model_name="facebook/w2v-bert-2.0",
    frozen=True,
    use_lora=True,
    lora_config_args=dict(
        r=64,
        lora_alpha=128,
        target_modules=["linear_q", "linear_v"],
        lora_dropout=0.0,
        bias="none",
    ),
)

_RESNET_MODEL_ARGS = dict(
    feat_dim=80,
    embed_dim=256,
    pooling_func="TSTP",
    two_emb_layer=False,
)


# ---------------------------------------------------------------------------
# AudioEncoder
# ---------------------------------------------------------------------------
class AudioEncoder(nn.Module):
    """
    raw wav/fbank -> speaker embedding via wespeaker frontend + speaker model.

    Args:
        frontend_type:  "w2vbert" or "resnet"
        model_name:     wespeaker model class (None = auto from frontend_type)
        model_args:     kwargs for the speaker model (merged with defaults)
        frontend_args:  kwargs for the SSL frontend (w2vbert only, merged with defaults)
        pretrained_path:  path to pretrained wespeaker checkpoint
        freeze_strategy: "none" | "all" | "frontend" | "speaker"
        sample_rate:     target sample rate
    """

    def __init__(
        self,
        frontend_type="w2vbert",
        model_name=None,
        model_args=None,
        frontend_args=None,
        pretrained_path="",
        freeze_strategy="all",
        sample_rate=16000,
    ):
        super().__init__()

        if not _HAS_WESPEAKER:
            raise ImportError(
                "wespeaker is required for AudioEncoder. "
                "Activate the wespeaker conda env or add it to PYTHONPATH."
            )

        assert frontend_type in ("w2vbert", "resnet"), \
            f"Unknown frontend_type: {frontend_type}"

        self.frontend_type = frontend_type
        self.sample_rate = sample_rate

        # ---- speaker model ----
        if frontend_type == "w2vbert":
            _default_name = "W2VBert_Adapter_MFA"
            _default_args = _W2VBERT_MODEL_ARGS
        else:
            _default_name = "ResNet293"
            _default_args = _RESNET_MODEL_ARGS

        _model_name = model_name or _default_name
        _model_args = {**_default_args, **(model_args or {})}
        self._output_dim = _model_args.get("embed_dim", 256)

        self.model = get_speaker_model(_model_name)(**_model_args)

        # ---- SSL frontend (w2vbert only) ----
        if frontend_type == "w2vbert":
            _frontend_args = {**_W2VBERT_FRONTEND_ARGS, **(frontend_args or {})}
            frontend = frontend_class_dict["w2vbert"](
                **_frontend_args,
                sample_rate=sample_rate,
            )
            self.model.add_module("frontend", frontend)

        # ---- pretrained checkpoint ----
        if pretrained_path:
            self._load_pretrained(pretrained_path)

        # ---- freeze ----
        if freeze_strategy == "all":
            self._freeze_all()
        elif freeze_strategy == "frontend":
            self._freeze_frontend()
        elif freeze_strategy == "speaker":
            self._freeze_speaker()
        # "none" does nothing

    # --------------------------------------------------
    # internal helpers
    # --------------------------------------------------
    def _load_pretrained(self, path):
        """Load checkpoint, skipping mismatched keys (e.g. projection)."""
        if not path or not os.path.exists(path):
            return  # skip if no pretrained path (weights loaded from main checkpoint)
        try:
            _ws_load_ckpt(self.model, path)
        except Exception:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
            self.model.load_state_dict(state, strict=False)

    def _freeze_all(self):
        """Freeze every parameter in the audio model."""
        for p in self.model.parameters():
            p.requires_grad = False

    def _freeze_frontend(self):
        """Freeze only the SSL frontend (w2vbert). No-op for resnet."""
        if hasattr(self.model, "frontend"):
            for p in self.model.frontend.parameters():
                p.requires_grad = False

    def _freeze_speaker(self):
        """Freeze the speaker model but keep the frontend trainable."""
        for name, p in self.model.named_parameters():
            if not name.startswith("frontend."):
                p.requires_grad = False

    # --------------------------------------------------
    # public API
    # --------------------------------------------------
    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, audio_input, wav_len=None):
        """
        Args:
            audio_input:
                w2vbert mode — (B, T) or (T,) raw waveform
                resnet mode  — (B, T, F) fbank features (from E2EDataset)
            wav_len: (B,) actual lengths (optional, w2vbert only)
        Returns:
            embed: (B, embed_dim) speaker embedding
        """
        if self.frontend_type == "w2vbert":
            if audio_input.dim() == 1:
                audio_input = audio_input.unsqueeze(0)
            B, T = audio_input.shape
            if wav_len is None:
                wav_len = torch.full((B,), T, dtype=torch.long, device=audio_input.device)
            features, _ = self.model.frontend(audio_input, wav_len)
        else:
            # resnet: audio_input is already fbank (B, T, F) from E2EDataset
            features = audio_input

        # ---- speaker embedding ----
        outputs = self.model(features)
        embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
        return embeds
