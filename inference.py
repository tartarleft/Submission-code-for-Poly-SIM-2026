#!/usr/bin/env python
"""
Standalone inference script for Router-Only v2 model submission.

Usage:
    CUDA_VISIBLE_DEVICES=0 python inference.py --test_home /path/to/test_data --output_dir submission
    CUDA_VISIBLE_DEVICES=0 python inference.py --test_home /path/to/test_data --output_dir submission --gpu 0
"""

import os
import sys

# Setup paths: make local modules importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
# Ensure wespeaker package is importable from local copy
_wespeaker_dir = os.path.join(SCRIPT_DIR, "wespeaker")
if os.path.isdir(_wespeaker_dir) and SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Point w2v-bert to local HF cache instead of downloading
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = os.path.join(SCRIPT_DIR, "hf_cache")

import argparse

import numpy as np
import pandas as pd
import torch
import torchaudio
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from config import ExperimentConfig
from models.router_only import RouterOnlyModel

SAMPLE_RATE = 16000
FACE_TRANSFORM = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])



def load_model(checkpoint_path, device):
    """Load model from checkpoint. No external pretrained files needed."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    cfg_dict["device"] = str(device)
    config = ExperimentConfig(**cfg_dict)

    # Skip loading separate pretrained ckpts — all weights come from our checkpoint
    config.audio_pretrained_path = ""   # skip avg4_model.pt
    config.face_model_ckpt = ""          # skip best_face_model.pt
    # Force local relative path (checkpoint config may contain training-time absolute paths)
    config.face_model_config = os.path.join(SCRIPT_DIR, "face/models/iresnet/configs/v1_ir18.yaml")

    # Construct model
    model = RouterOnlyModel(
        config=config,
        audio_encoder_type=config.audio_encoder_type,
        audio_pretrained_path=config.audio_pretrained_path,
        audio_freeze=config.audio_freeze,
        audio_model_name=config.audio_model_name or None,
        audio_model_args=config.audio_model_args,
        audio_frontend_args=config.audio_frontend_args,
        face_model_ckpt="",  # skip, weights loaded below
    )

    # Load all weights from checkpoint
    # Register projection buffer first since _load_audio_projection was skipped
    state = ckpt["model_state"]
    if "_audio_proj_weight" in state:
        model.register_buffer("_audio_proj_weight", state["_audio_proj_weight"])
    if "_audio_proj_scale" in state:
        model.register_buffer("_audio_proj_scale", state["_audio_proj_scale"])
    model.load_state_dict(state)
    model.to(device).eval()

    print(f"Model loaded: epoch={ckpt.get('epoch', '?')}, device={device}")
    return model


def load_audio(path):
    wav, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=SAMPLE_RATE)
    return wav.mean(dim=0) if wav.ndim > 1 else wav.squeeze(0)


def load_face(path):
    img = Image.open(path).convert("RGB")
    return FACE_TRANSFORM(img)


def predict_sample(model, face_tensor, wav_tensor, device):
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        out = model(face_tensor, wav_tensor)
    return {
        "fusion": out["final_logits"].argmax(1).item(),
        "audio": out["audio_logits"].argmax(1).item(),
    }


def run_inference(checkpoint, output_dir, test_home, english_csv, urdu_csv,
                  gpu_id=0, num_gpus=1):
    """Run inference on test set and generate submission CSVs."""
    test_csvs = {
        "English": english_csv,
        "Urdu": urdu_csv,
    }

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, device)

    os.makedirs(output_dir, exist_ok=True)

    # English submission
    print("\n=== Processing English ===")
    _process_language(model, "English", test_csvs["English"], test_home,
                      output_dir, "submission_v1_test_English_English",
                      ["p3", "p4"], device, gpu_id, num_gpus)

    # Urdu submission
    print("\n=== Processing Urdu ===")
    _process_language(model, "Urdu", test_csvs["Urdu"], test_home,
                      output_dir, "submission_v1_test_English_Urdu",
                      ["p5", "p6"], device, gpu_id, num_gpus)


def _process_language(model, lang, csv_path, test_home, output_dir, prefix,
                      col_names, device, gpu_id, num_gpus):
    df = pd.read_csv(csv_path)
    indices = list(range(gpu_id, len(df), num_gpus))

    p_fusion = np.full(len(df), -1, dtype=int)
    p_audio = np.full(len(df), -1, dtype=int)

    for i in tqdm(indices, desc=lang):
        row = df.iloc[i]
        audio_path = os.path.join(test_home, row["voices"])
        face_path = os.path.join(test_home, row["faces"])

        wav = load_audio(audio_path)
        face = load_face(face_path)

        preds = predict_sample(
            model,
            face.unsqueeze(0).to(device),
            wav.unsqueeze(0).float().to(device),
            device,
        )
        p_fusion[i] = preds["fusion"]
        p_audio[i] = preds["audio"]

    if gpu_id == 0 or num_gpus == 1:
        out = pd.DataFrame({"key": df["key"]})
        out[col_names[0]] = p_fusion
        out[col_names[1]] = p_audio
        out_path = os.path.join(output_dir, f"{prefix}.csv")
        out.to_csv(out_path, index=False)
        print(f"  Saved: {out_path} ({len(df)} samples)")


def main():
    parser = argparse.ArgumentParser(description="Router-Only v2 Submission")
    parser.add_argument("--checkpoint", default=os.path.join(SCRIPT_DIR, "epoch_030.pt"),
                        help="Path to model checkpoint")
    parser.add_argument("--output_dir", default="submission",
                        help="Output directory for submission CSVs")
    parser.add_argument("--test_home", required=True,
                        help="Root directory of test data (CSV voices/faces paths are relative to this)")
    parser.add_argument("--english_csv", required=True,
                        help="Path to English test CSV")
    parser.add_argument("--urdu_csv", required=True,
                        help="Path to Urdu test CSV")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (default: 0)")
    args = parser.parse_args()

    # Auto-detect HF cache
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = os.path.join(SCRIPT_DIR, "hf_cache")

    run_inference(args.checkpoint, args.output_dir, args.test_home,
                  args.english_csv, args.urdu_csv, args.gpu, num_gpus=1)


if __name__ == "__main__":
    main()
