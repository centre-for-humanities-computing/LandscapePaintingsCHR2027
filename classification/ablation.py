"""
Leave-one-out crop ablation attribution. Auto-discovers every classifier
under CLASSIFIER_BASE_DIR and combines all their results into one parquet.

Since predictions.parquet has no "image" column, images are re-merged 
via MERGE_KEY.

"""

import gc
import io
import os
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import polars as pl
import open_clip
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T
from transformers import AutoModel
from joblib import load

# ── CONFIG ────────────────────────────────────────────────────────────────────

ABLATION_MODE = "fold"
# fold: baseline + every leave-one-out variant scored by
# that row's own held-out fold model (fold_models[fold_id])
# full: baseline computed here from final_clf directly for in-sample inquiries

LOCAL_DATA_PATH = "data/classification.parquet"   # must match train-classifier

# Used for key-based merges to attach images and to combine different 
# classifiers' results together.
MERGE_KEY = "filename"

# Auto-discovers subfolder with classifier.joblib and predictions.parquet 
# (created by train-classifier.py)
CLASSIFIER_BASE_DIR = "classifier"

# backbone_key of trained projection heads
HEAD_PATHS = {
    "dinov2base":  ("DINOv2-base",  r"heads/Landscapehead-DINOv2-base.pt"),
    "dinov2giant": ("DINOv2-giant", r"heads/Landscapehead-DINOv2-giant.pt"),
    "eva02":       ("EVA-02",       r"heads/Landscapehead-EVA02-CLIP-L-14-336.pt"),
    "siglip":      ("SigLIP",       r"heads/Landscapehead-SigLIP-SO400M.pt"),
}

OUTPUT_PARQUET = rf"data/attribution_scores_{ABLATION_MODE}.parquet"

GRID_SIZE      = 4
PATCHES_SAMPLE = 64
DEVICE         = "auto"

SLEEP_BETWEEN_IMAGES = 0.5
SLEEP_EVERY_N        = 10
MAX_CPU_THREADS      = 4

# ── infering backbone  ────────────────────────────────────────────────────

BACKBONE_REGISTRY = {
    "dinov2base": {"backend": "dinov2", "model_name": "facebook/dinov2-base", "image_size": 224, "patch_size": 14},
    "dinov2giant": {"backend": "dinov2", "model_name": "facebook/dinov2-giant", "image_size": 224, "patch_size": 14},
    "eva02": {"backend": "open_clip", "model_name": "EVA02-L-14-336", "pretrained": "merged2b_s6b_b61k",
              "image_size": 336, "patch_size": 14},
    "siglip": {"backend": "open_clip", "model_name": "ViT-SO400M-14-SigLIP-384", "pretrained": "webli",
               "image_size": 384, "patch_size": 14},
}


def infer_backbone_key(embedding_col):
    normalized = re.sub(r"[^a-z0-9]", "", embedding_col.lower())
    for key in BACKBONE_REGISTRY:
        if key in normalized:
            return key
    raise ValueError(f"Cannot infer backbone from column '{embedding_col}'. "
                      f"Expected one of {list(BACKBONE_REGISTRY)} to appear in the (normalized) name.")


# ── classifier discovery ────────────────────────────────────────────────────

def discover_classifiers(base_dir):
    """Every subfolder of base_dir is treated as one classifier to ablate."""
    found = {}
    for entry in sorted(os.listdir(base_dir)):
        subdir = os.path.join(base_dir, entry)
        joblib_path = os.path.join(subdir, "classifier.joblib")
        pred_path = os.path.join(subdir, "predictions.parquet")
        if os.path.isdir(subdir) and os.path.exists(joblib_path) and os.path.exists(pred_path):
            found[entry] = (joblib_path, pred_path)
    if not found:
        raise RuntimeError(f"No classifier.joblib + predictions.parquet found under any subfolder of {base_dir}.")
    print(f"[info] Discovered {len(found)} classifier(s) under {base_dir}: {list(found)}")
    return found


# ── data loading + alignment ────────────────────────────────────────────────

def load_parquet_robust(path, columns=None):
    """Reads a parquet file with polars. Pass columns=[...] to read only
    those columns and skip the rest because the file is extensive at this point."""
    size_mb = os.path.getsize(path) / (1024 * 1024)
    col_note = f", reading only {len(columns)} of its columns" if columns else ""
    print(f"  [info] Reading {path} ({size_mb:.0f} MB{col_note})...")
    t0 = time.time()
    df = pl.read_parquet(path, columns=columns).to_pandas()
    print(f"  [info] Read complete: {len(df)} row(s) in {time.time() - t0:.1f}s.")
    return df


def load_local_data():
    """Loads just the columns needed to attach images to a classifier's
    predictions.parquet."""
    needed_cols = sorted({c for c in (MERGE_KEY, "image") if c})
    return load_parquet_robust(LOCAL_DATA_PATH, columns=needed_cols)


def safe_merge(base_df, other_df, other_cols, label):
    """Merges other_df's other_cols onto base_df."""
    if MERGE_KEY not in base_df.columns or MERGE_KEY not in other_df.columns:
        raise ValueError(f"{label}: '{MERGE_KEY}' column missing from one or both dataframes -- "
                         f"required, no fallback alignment is attempted.")
    merged = base_df.merge(other_df[[MERGE_KEY] + other_cols], on=MERGE_KEY, how="left", validate="one_to_one")
    if merged[other_cols[0]].isna().any():
        raise ValueError(f"{label}: some rows have no matching '{MERGE_KEY}' -- "
                         f"check both cover the same images.")
    print(f"  [info] {label}: merged via '{MERGE_KEY}' ({len(merged)} rows, verified 1:1).")
    return merged


def combine_results(combined_df, new_df, new_cols, classifier_label):
    """First classifier seeds the combined dataframe; every subsequent one
    is merged in via safe_merge()."""
    if combined_df is None:
        return new_df[[MERGE_KEY, "artist", "main_class", "image"] + new_cols].copy()
    return safe_merge(combined_df, new_df, new_cols, f"combining '{classifier_label}'")


# ── model loading ────────────────────────────────────────────────────────────

def load_backbone(cfg, device):
    if cfg["backend"] == "dinov2":
        model = AutoModel.from_pretrained(cfg["model_name"])
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif cfg["backend"] == "open_clip":
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg["model_name"], pretrained=cfg["pretrained"], device=device)
        normalizer = next(t for t in preprocess.transforms if isinstance(t, T.Normalize))
    else:
        raise ValueError(f"Unknown backend: {cfg['backend']}")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, normalizer, cfg["image_size"]


def load_head_from_pt(pt_path, device):
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    sd, input_dim, proj_dim = ckpt["state_dict"], ckpt["input_dim"], ckpt["proj_dim"]
    hidden_dim = sd["net.0.weight"].shape[0]
    head = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
                          nn.Linear(hidden_dim, proj_dim))
    head.load_state_dict({k.replace("net.", ""): v for k, v in sd.items()})
    head.eval().to(device)
    return head


# ── image preprocessing / token extraction / pooling ─────────────────────────

def bytes_to_pil(raw):
    img = Image.open(io.BytesIO(raw)).convert("RGB")


def normalise(tensor, normalizer, device):
    mean = torch.tensor(normalizer.mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(normalizer.std, device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def image_to_grid_crops(img, normalizer, image_size, device):
    cell_size = min(img.size) // GRID_SIZE
    img_t = T.ToTensor()(img).to(device)
    _, H, W = img_t.shape
    crops = torch.stack([
        img_t[:, r * cell_size:(r + 1) * cell_size, c * cell_size:(c + 1) * cell_size]
        for r in range(H // cell_size) for c in range(W // cell_size)
    ])
    if cell_size != image_size:
        crops = F.interpolate(crops, size=(image_size, image_size), mode="bicubic",
                              align_corners=False, antialias=True)
    return normalise(crops, normalizer, device)


def _patch_tokens_open_clip(model, imgs):
    visual = model.visual
    if hasattr(visual, "conv1"):
        x = visual.conv1(imgs)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = visual.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + visual.positional_embedding
        if hasattr(visual, "ln_pre"):
            x = visual.ln_pre(x)
        x = visual.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        return x[:, 1:, :]
    if hasattr(visual, "trunk"):
        x = visual.trunk.forward_features(imgs)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        has_cls = getattr(visual.trunk, "cls_token", None) is not None
        if has_cls and x.shape[1] > 1:
            return x[:, 1:, :]
        return x
    raise RuntimeError(f"Unrecognised open_clip visual type: '{type(visual).__name__}'")


def get_patch_tokens(backend, model, imgs):
    if backend == "open_clip":
        return _patch_tokens_open_clip(model, imgs)
    return model(pixel_values=imgs).last_hidden_state[:, 1:, :]


def descriptor_from_tokens(patch_tokens, exclude_idx=None):
    """Pooling mean/std/mean-squared over a random subsample of
    PATCHES_SAMPLE*num_crops patch tokens."""
    if exclude_idx is not None:
        keep = [i for i in range(patch_tokens.shape[0]) if i != exclude_idx]
        patch_tokens = patch_tokens[keep]
    flat = F.normalize(patch_tokens, dim=-1).reshape(-1, patch_tokens.shape[-1])
    n = min(PATCHES_SAMPLE * patch_tokens.shape[0], flat.shape[0])
    idx = torch.randperm(flat.shape[0], device=flat.device)[:n]
    flat = flat[idx]
    pooled = torch.cat([flat.mean(0), flat.std(0), (flat ** 2).mean(0)])
    return F.normalize(pooled, dim=-1)


@torch.no_grad()
def ablate_image_deltas_only(img, backend, model, normalizer, image_size,
                              head, clf, danish_idx, device):
    crops = image_to_grid_crops(img, normalizer, image_size, device)
    patch_tokens = get_patch_tokens(backend, model, crops)
    num_crops = patch_tokens.shape[0]

    def desc_to_proba(desc):
        desc = F.normalize(head(desc.unsqueeze(0)).squeeze(0), dim=-1)
        return clf.predict_proba(desc.cpu().numpy().reshape(1, -1))[0]

    if num_crops == 1:
        return None

    crop_p_danish = []
    for i in range(num_crops):
        proba = desc_to_proba(descriptor_from_tokens(patch_tokens, exclude_idx=i))
        crop_p_danish.append(float(proba[danish_idx]))
    return crop_p_danish


# ── per-classifier ablation ───────────────────────────────────────────────────

def ablate_one_classifier(joblib_path, pred_path, local_df, device):
    """Runs the full leave-one-out ablation for one classifier. 
    Returns (result_df, new_cols) plus this classifier's 
    ablation_baseline_*/ablation_scores_* columns."""
    t_start = time.time()
    ckpt = load(joblib_path)
    clf_full = ckpt["model"]
    fold_models = ckpt.get("fold_models")
    if ABLATION_MODE == "fold" and not fold_models:
        raise ValueError(f"ABLATION_MODE='fold' but '{joblib_path}' has no 'fold_models' -- "
                         f"was it trained with the fold-aware training script?")
    le = ckpt["label_encoder"]
    embedding_col = ckpt["embedding_col"]
    backbone_key = infer_backbone_key(embedding_col)
    backbone_cfg = BACKBONE_REGISTRY[backbone_key]
    print(f"  embedding_col: {embedding_col}  (backbone={backbone_key})  [{time.time() - t_start:.1f}s]")

    if "backbone" in embedding_col.lower():
        raise ValueError(
            f"'{joblib_path}' was trained on a raw BACKBONE embedding ('{embedding_col}'), "
            f"not a projected/head embedding -- this script always applies a projection head, "
            f"which will produce a dimension mismatch against a backbone-trained classifier "
            f"(head output dim != what this classifier was trained on). "
            f"Move/remove this folder from CLASSIFIER_BASE_DIR if it's a leftover from earlier "
            f"experimentation -- only head-trained classifiers belong here.")

    if backbone_key not in HEAD_PATHS:
        raise ValueError(f"No HEAD_PATHS entry for backbone '{backbone_key}'.")
    classifier_label, head_path = HEAD_PATHS[backbone_key]

    # Only read the columns needed. In "fold" mode this skips the embedding 
    # column entirely, since it's only needed to compute the "full" baseline. 
    schema_cols = set(pl.scan_parquet(pred_path).collect_schema().keys())
    wanted = {"artist", "main_class", "p_true", "p_wrong", MERGE_KEY}
    wanted.add("fold_id" if ABLATION_MODE == "fold" else embedding_col)
    cols_to_read = [c for c in wanted if c in schema_cols]
    pred_df = load_parquet_robust(pred_path, columns=cols_to_read)
    if ABLATION_MODE == "fold" and "fold_id" not in pred_df.columns:
        raise ValueError(f"ABLATION_MODE='fold' but '{pred_path}' has no 'fold_id' column.")
    if ABLATION_MODE == "full" and embedding_col not in pred_df.columns:
        raise ValueError(f"ABLATION_MODE='full' needs the raw '{embedding_col}' column in "
                         f"'{pred_path}' to score final_clf on -- it isn't there.")

    aligned = safe_merge(pred_df, local_df, ["image"], "image alignment")

    danish_idx = list(le.classes_).index("danish")
    is_danish = (aligned["main_class"] == "danish").values

    if ABLATION_MODE == "fold":
        baseline_p_all = np.where(is_danish, aligned["p_true"].values, aligned["p_wrong"].values)
        fold_id_all = aligned["fold_id"].values
    else:
        t0 = time.time()
        X_full = np.vstack(aligned[embedding_col].values)
        baseline_p_all = clf_full.predict_proba(X_full)[:, danish_idx]
        print(f"  [info] Computed full-model baseline directly from final_clf "
              f"({len(aligned)} rows, {time.time() - t0:.1f}s).")

    t0 = time.time()
    backbone, normalizer, image_size = load_backbone(backbone_cfg, device)
    print(f"  [info] Backbone loaded ({time.time() - t0:.1f}s).")
    t0 = time.time()
    head = load_head_from_pt(head_path, device)
    print(f"  [info] Head loaded ({time.time() - t0:.1f}s). Starting per-image ablation...")

    safe_name = re.sub(r"[^a-z0-9]+", "_", classifier_label.lower()).strip("_")
    col_scores = f"ablation_scores_{safe_name}_{ABLATION_MODE}"
    col_baseline = f"ablation_baseline_{safe_name}_{ABLATION_MODE}"

    scores_list, baseline_list, n_bad_fold = [], [], 0

    for img_i, raw in enumerate(tqdm(aligned["image"], desc=f"  Ablating {classifier_label}",
                                      unit="img", total=len(aligned))):
        img = bytes_to_pil(raw)
        baseline_p = float(baseline_p_all[img_i])

        if ABLATION_MODE == "fold":
            fold_id = int(fold_id_all[img_i])
            if fold_id < 0 or fold_id >= len(fold_models):
                n_bad_fold += 1
                baseline_list.append(baseline_p)
                scores_list.append([0.0])
                continue
            row_clf = fold_models[fold_id]
        else:
            row_clf = clf_full

        crop_p_danish = ablate_image_deltas_only(
            img, backbone_cfg["backend"], backbone, normalizer, image_size,
            head, row_clf, danish_idx, device,
        )
        signed_scores = [0.0] if crop_p_danish is None else [baseline_p - p for p in crop_p_danish]
        baseline_list.append(baseline_p)
        scores_list.append(signed_scores)

        if SLEEP_BETWEEN_IMAGES > 0 and (img_i + 1) % SLEEP_EVERY_N == 0:
            time.sleep(SLEEP_BETWEEN_IMAGES)

    if n_bad_fold:
        print(f"  [warn] {n_bad_fold} row(s) had no valid fold model (fold_id out of range) -- "
              f"given a degenerate [0.0] score instead of being scored.")

    aligned[col_baseline] = baseline_list
    aligned[col_scores] = scores_list

    del backbone, head
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.empty_cache()
    gc.collect()

    return aligned, [col_baseline, col_scores]


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if ABLATION_MODE not in ("fold", "full"):
        raise ValueError(f"ABLATION_MODE must be 'fold' or 'full', got {ABLATION_MODE!r}")

    if DEVICE == "auto":
        device = (torch.device("cuda") if torch.cuda.is_available() else
                  torch.device("mps") if torch.backends.mps.is_available() else
                  torch.device("cpu"))
    else:
        device = torch.device(DEVICE)
    print(f"[info] Device: {device}  |  ABLATION_MODE: {ABLATION_MODE}")

    if MAX_CPU_THREADS is not None:
        torch.set_num_threads(MAX_CPU_THREADS)

    classifiers = discover_classifiers(CLASSIFIER_BASE_DIR)
    print("[info] Loading local data for image alignment...")
    local_df = load_local_data()

    combined_df = None
    for name, (joblib_path, pred_path) in classifiers.items():
        print(f"\n[classifier] {name}")
        result_df, new_cols = ablate_one_classifier(joblib_path, pred_path, local_df, device)
        combined_df = combine_results(combined_df, result_df, new_cols, name)

    print(f"\n[info] Saving combined attribution file -> {OUTPUT_PARQUET}")
    os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)
    combined_df.to_parquet(OUTPUT_PARQUET, index=False)
    ablation_cols = [c for c in combined_df.columns if c.startswith("ablation_scores_")]
    print(f"[done] {len(ablation_cols)} classifier(s) combined: {ablation_cols}")