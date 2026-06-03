from dataclasses import dataclass
from typing import Optional
import logging


@dataclass
class ExperimentConfig:
    # ── Core / Training ──────────────────────────────────────────────
    seed: int = 1
    device: str = "cuda"
    lr: float = 3e-5
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 300
    num_workers: int = 0
    embedding_dim: int = 512
    weighted_sampler: bool = True          # oversample minority classes
    debug: bool = False

    # ── Model Architecture ───────────────────────────────────────────
    model_type: str = "e2e_moe"              # "e2e_moe" | "e2e_face" | "fop" | "multibranch"
    loss_audio: float = 1.0
    loss_fusion: float = 1.0

    # ── Only used by non-MoE models ──────────────────────────────────
    fusion: str = "gated"                  # "linear" | "gated" | "concat" (E2EMultiBranchFOP/FOP)
    # alpha: float = 0.0                     # OPL loss weight (E2EMultiBranchFOP/FOP)
    # loss_face: float = 2.0                 # face head loss weight (E2EMultiBranchFOP/FOP)

    # ── Data / Version ───────────────────────────────────────────────
    version: str = "v1"
    seen_lang: str = "English"

    # ── Baseline Pipeline (precomputed .npy features, not used by e2e models) ─
    # home_dir: str = "/feats"

    # ── Face Encoder (e2e models) ────────────────────────────────────
    face_model_config: str = "./face/models/iresnet/configs/v1_ir18.yaml"
    # face_model_ckpt: str = "/opt/nas/p/local/zhuyao/fame/polysim/embeddings/pretrained_face_models/adaface_ir18_webface4m/model.pt"
    face_model_ckpt: str = "./checkpoints/e2e_face_v1_Eng_none_a0.0_w2vbert_0524_095254/model.pt"
    face_model_freeze: bool = True          # True = freeze, False = fine-tune

    # ── Audio Encoder (e2e models only) ──────────────────────────────
    audio_encoder_type: str = "w2vbert"    # "w2vbert" | "resnet"
    # audio_pretrained_path: str = "/opt/nas/p/local/zhuyao/fame/polysim/embeddings/pretrained_audio_models/voxceleb_voxblink2_w2v_bert2_lora_adapterMFA/avg_model.pt"
    audio_pretrained_path: str = "avg4_model.pt"
    # freeze level: "none" (all trainable) | "all" | "frontend" | "speaker"
    audio_freeze: str = "all"
    audio_model_name: str = ""
    audio_model_args: Optional[dict] = None
    audio_frontend_args: Optional[dict] = None

    # ── E2E Dataset ──────────────────────────────────────────────────
    e2e_home_dir: str = "/opt/nas/p/local/zhuyao/fame/polysim/data/train"
    e2e_val_home_dir: str = "/opt/nas/p/local/zhuyao/fame/polysim/data"
    audio_home_dir: str = "data/audio_train"
    face_home_dir: str = "data/faces_train"
    train_csv: str = "preprocess/train_vad_filtered_0523_tts_aug_face.csv"
    val_csv: str = "/opt/nas/p/local/zhuyao/fame/polysim_audio/csv_files/comp/v1_test_English_label.csv"
    unseen_csv: str = "/opt/nas/p/local/zhuyao/fame/polysim_audio/csv_files/comp/v1_test_Urdu_label.csv"
    sample_rate: int = 16000
    num_mel_bins: int = 80                 # fbank bins for resnet frontend
    chunk_wav_len: int = 48040
    eval_batch_size: int = 4

    # ── Face Image Data ──────────────────────────────────────────────
    face_home_dir: str = "/opt/nas/p/local/zhuyao/fame/polysim_audio/data/faces_train"

    # ── Only used by non-MoE models ──────────────────────────────────
    # face_only: bool = False                # only use face branch (E2EMultiBranchFOP)
    # face_drop_prob: float = 0.5            # training: probability to simulate missing face (E2EMultiBranchFOP)

    # ── Test-Time (not used by e2e models) ───────────────────────────
    # test_missing_modality: str = "face"    # "face" | "audio"
    # test_alpha: float = 0.0

    # ── MoE v2 Router ────────────────────────────────────────────────
    router_loss_weight: float = 0.5          # lambda for KL(router, target)
    external_audio_dir: str = ""             # path to external audio files (not from 70 speakers)
    external_face_dir: str = ""              # path to external face images
    sample_ratios: tuple = (0.6, 0.1, 0.1, 0.2)  # [normal, audio_replace, face_replace, no_face]
    infer_mode: str = "fusion"               # "fusion" (blank face) or "audio_only" (bypass fusion)

    # ── Early Stopping ───────────────────────────────────────────────
    early_stop: bool = True
    early_stop_patience: int = 5           # tolerance in epochs
    early_stop_min_delta: float = 0.2      # minimum improvement
    early_stop_metric: str = "unseen"      # "seen" | "unseen"
    val_start_epoch: int = 5              # skip validation before this epoch

    def __post_init__(self):
        if self.audio_pretrained_path and self.audio_frontend_args is None and self.audio_encoder_type == "w2vbert":
            self.audio_frontend_args = {"frozen": False, "use_lora": False, "lora_config_args": None}

    @property
    def log_level(self):
        return logging.DEBUG if self.debug else logging.INFO

    @property
    def resolved_num_classes(self) -> int:
        if self.version == "v1":
            return 70
        elif self.version == "v2":
            return 84
        elif self.version == "v3":
            return 36
        else:
            raise ValueError(f"Unknown version '{self.version}'")

    @property
    def unseen_lang(self) -> str:
        mapping = {
            ("v1", "English"): "Urdu",
            ("v1", "Urdu"):    "English",
            ("v2", "English"): "Hindi",
            ("v2", "Hindi"):   "English",
            ("v3", "English"): "German",
            ("v3", "German"):  "English",
        }
        key = (self.version, self.seen_lang)
        if key not in mapping:
            raise ValueError(f"Invalid version '{self.version}' or seen_lang '{self.seen_lang}'.")
        return mapping[key]
