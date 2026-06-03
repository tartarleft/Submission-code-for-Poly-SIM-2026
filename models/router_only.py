"""
Router-Only Model: frozen pretrained encoders + lightweight adapters + 2-class Router.

Architecture:
  Audio encoder (frozen) → 256d emb → AudioAdapter → 256d ─┐
                                                             ├→ Router → [w_audio, w_face]
  Face encoder (frozen) → 512d emb → FaceAdapter → 256d ───┘

  audio_logits = cosine(audio_emb, proj_weight) * scale
  face_logits = face_classifier(face_emb)
  final_logits = w_audio * audio_logits + w_face * face_logits

Loss:
  Total = CE(final_logits, label) + λ * KL(router_weights, router_target)
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio_encoder import AudioEncoder
from .face_encoder import load_face_model


class ModalityAdapter(nn.Module):
    """Linear → ReLU → Linear adapter to project modality features to common dim."""

    def __init__(self, input_dim, hidden_dim=256, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class Router2(nn.Module):
    """2-class Router: [audio_adapted; face_adapted] → softmax → [w_audio, w_face]."""

    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, audio_adapted, face_adapted):
        x = torch.cat([audio_adapted, face_adapted], dim=1)
        return F.softmax(self.net(x), dim=1)


class RouterOnlyModel(nn.Module):
    """Router-Only multimodal fusion: frozen encoders + adapters + 2-class router."""

    def __init__(self, config, audio_encoder_type="w2vbert",
                 audio_pretrained_path="", audio_freeze="all",
                 audio_model_name=None, audio_model_args=None,
                 audio_frontend_args=None,
                 face_model_ckpt="best_face_model.pt"):
        super().__init__()

        self.config = config
        num_classes = config.resolved_num_classes  # 70

        # ── Audio encoder (frozen) ──────────────────────────────────
        self.audio_encoder = AudioEncoder(
            frontend_type=audio_encoder_type,
            model_name=audio_model_name,
            model_args=audio_model_args,
            frontend_args=audio_frontend_args,
            pretrained_path=audio_pretrained_path,
            freeze_strategy=audio_freeze,
        )
        audio_dim = self.audio_encoder.output_dim  # 256

        # Load projection weights from avg4_model.pt (frozen, as buffer)
        self.register_buffer("_audio_proj_weight", None)
        self.register_buffer("_audio_proj_scale", None)
        self._load_audio_projection(audio_pretrained_path)

        # ── Face encoder + classifier (frozen) ──────────────────────
        self.face_encoder, face_dim = load_face_model(config)
        self.face_classifier = nn.Linear(face_dim, num_classes)
        self._load_face_model(face_model_ckpt, face_dim, num_classes)

        # Freeze face encoder + classifier
        for p in self.face_encoder.parameters():
            p.requires_grad = False
        for p in self.face_classifier.parameters():
            p.requires_grad = False

        # ── Adapters (trainable) ────────────────────────────────────
        self.audio_adapter = ModalityAdapter(audio_dim, 256, 256)
        self.face_adapter = ModalityAdapter(face_dim, 256, 256)

        # ── Router (trainable) ──────────────────────────────────────
        self.router = Router2(input_dim=512, hidden_dim=256)

    def _load_audio_projection(self, audio_pretrained_path):
        """Load projection weight & scale from wespeaker avg4 checkpoint."""
        # Skip if no pretrained path — weights loaded from main checkpoint
        if not audio_pretrained_path or not os.path.exists(audio_pretrained_path):
            return
        import sys
        # Prevent sys.path pollution from AudioEncoder's imports
        saved_path = sys.path.copy()
        try:
            ckpt = torch.load(audio_pretrained_path, map_location="cpu",
                              weights_only=False)
        finally:
            sys.path = saved_path

        if isinstance(ckpt, dict) and "projection.weight" not in ckpt:
            # Might be wrapped in 'state_dict' or 'model_state'
            for k in ("state_dict", "model_state"):
                if k in ckpt:
                    ckpt = ckpt[k]
                    break

        if "projection.weight" in ckpt:
            self._audio_proj_weight = ckpt["projection.weight"].clone()
        else:
            raise KeyError(f"'projection.weight' not found in {audio_pretrained_path}")

        # Scale from config default (ArcMargin scale=32.0)
        self._audio_proj_scale = torch.tensor(32.0)

    def _load_face_model(self, face_model_ckpt, face_dim, num_classes):
        """Load face encoder + classifier from best_face_model.pt."""
        # Skip if no pretrained path — weights loaded from main checkpoint
        if not face_model_ckpt or not os.path.exists(face_model_ckpt):
            return
        ckpt = torch.load(face_model_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state", ckpt)

        # Load face encoder
        enc_keys = {k: v for k, v in state.items() if k.startswith("face_encoder.")}
        if enc_keys:
            self.face_encoder.load_state_dict(enc_keys, strict=False)

        # Load face classifier
        if "face_classifier.weight" in state and "face_classifier.bias" in state:
            self.face_classifier.weight.data.copy_(state["face_classifier.weight"])
            self.face_classifier.bias.data.copy_(state["face_classifier.bias"])
        else:
            print(f"WARNING: face_classifier not found in {face_model_ckpt}")

    def _compute_audio_logits(self, audio_emb):
        """Cosine similarity against projection weights → logits."""
        cosine = F.linear(
            F.normalize(audio_emb, dim=1),
            F.normalize(self._audio_proj_weight, dim=1),
        )
        return cosine * self._audio_proj_scale

    def forward(self, face, audio):
        # Force frozen encoders' BN layers to eval mode
        # (prevents running_mean/var update during model.train())
        for m in self.audio_encoder.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                m.eval()
        for m in self.face_encoder.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                m.eval()

        # Audio path
        audio_emb = self.audio_encoder(audio)  # (B, 256)
        audio_logits = self._compute_audio_logits(audio_emb)  # (B, 70)

        # Face path
        face_emb = self.face_encoder(face)  # (B, 512)
        face_emb_norm = F.normalize(face_emb, dim=1)
        face_logits = self.face_classifier(face_emb_norm)  # (B, 70)

        # Adapter + Router
        audio_adapted = self.audio_adapter(audio_emb)
        face_adapted = self.face_adapter(face_emb)
        router_weights = self.router(audio_adapted, face_adapted)  # (B, 2)

        # Weighted combination
        w_audio = router_weights[:, 0:1]
        w_face = router_weights[:, 1:2]
        final_logits = w_audio * audio_logits + w_face * face_logits

        return {
            "final_logits": final_logits,
            "audio_logits": audio_logits,
            "face_logits": face_logits,
            "router_weights": router_weights,
        }
