import os
import importlib.util
from omegaconf import OmegaConf
import torch

def load_face_model(config):
    """Load iResNet face model from CVLface framework."""
    FACE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "face")
    models_init = os.path.join(FACE_DIR, "models", "__init__.py")
    spec = importlib.util.spec_from_file_location("face.models", models_init, submodule_search_locations=[os.path.join(FACE_DIR, "models")])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    get_model = mod.get_model

    cfg = OmegaConf.load(config.face_model_config)
    cfg.yaml_path = config.face_model_config
    model = get_model(cfg)
    if config.face_model_ckpt and os.path.exists(config.face_model_ckpt):
        model.load_state_dict_from_path(config.face_model_ckpt)
    output_dim = cfg.output_dim  # 512

    if config.face_model_freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model, output_dim


def convert_e2e_ckpt_to_face_weights(e2e_ckpt_path, out_path, prefix="face_encoder."):
    """Extract face encoder weights from an e2e checkpoint and save in adaface format.

    e2e checkpoint:  {"model_state": {"face_encoder.net.*": ...}, ...}
    adaface format:  {"net.*": ...}  (flat state_dict, no wrapping)

    The output can be loaded directly via load_face_model (which calls
    load_state_dict_from_path under the hood).
    """
    ckpt = torch.load(e2e_ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

    face_sd = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    if not face_sd:
        raise ValueError(f"No keys with prefix '{prefix}' found in {e2e_ckpt_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    torch.save(face_sd, out_path)
    print(f"Extracted {len(face_sd)} face encoder weights from {e2e_ckpt_path}")
    print(f"Saved to {out_path}")
    return out_path


def  convert_face_model(input_path):
    out_dir  = os.path.dirname(input_path)
    convert_e2e_ckpt_to_face_weights(input_path, os.path.join(out_dir, "model.pt"))

if __name__ == "__main__":
    convert_face_model('best.pt')