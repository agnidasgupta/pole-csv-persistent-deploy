#!/usr/bin/env python3
import argparse
import csv
import glob
import io
import json
import os
import sys
import time
import traceback
import zipfile
import threading
import warnings
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Dict, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import cgi


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def json_bytes(obj) -> bytes:
    return json.dumps(obj, indent=2).encode("utf-8")


def normalize_name(name: str) -> str:
    return "".join(str(name).strip().split()).lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def resolve_model_path(model_path_arg: str) -> str:
    matches = sorted(glob.glob(model_path_arg)) if any(ch in model_path_arg for ch in "*?[]") else [model_path_arg]
    matches = [m for m in matches if os.path.isfile(m)]
    if not matches:
        raise FileNotFoundError(f"No model file found for --model_path={model_path_arg!r}")
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


REQUIRED_COLUMNS = [
    "x",
    "y",
    "count",
    "minHeight",
    "maxHeight",
    "groundHeight",
    "contiguous",
    "contiguousHeight",
]

CANONICAL_NAMES = {
    "x": "x",
    "y": "y",
    "count": "count",
    "minheight": "minHeight",
    "maxheight": "maxHeight",
    "groundheight": "groundHeight",
    "contiguous": "contiguous",
    "contiguousheight": "contiguousHeight",
    "pole": "pole",
}


def parse_csv_and_strip_pole(raw: bytes, filename: str) -> pd.DataFrame:
    text = None
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except Exception as e:
            last_err = e
    if text is None:
        raise ValueError(f"Could not decode CSV {filename}: {last_err}")

    df = pd.read_csv(io.StringIO(text))
    df = normalize_columns(df)

    new_cols = []
    seen = set()
    for c in df.columns:
        canon = CANONICAL_NAMES.get(normalize_name(c), c)
        if canon in seen:
            raise ValueError(f"Duplicate/ambiguous column after normalization in {filename}: {c}")
        seen.add(canon)
        new_cols.append(canon)
    df.columns = new_cols

    if "pole" in df.columns:
        df = df.drop(columns=["pole"])

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {filename}: {missing}")

    if len(df) == 0:
        raise ValueError(f"CSV has zero rows: {filename}")

    return df


def validate_and_prepare_input_df(df: pd.DataFrame, grid_size: int = 400) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = normalize_columns(df)
    if df.columns.duplicated().any():
        dupes = list(pd.Index(df.columns[df.columns.duplicated()]).unique())
        raise ValueError(f"Duplicate column names after normalization: {dupes}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(df) == 0:
        raise ValueError("CSV has zero rows")

    work = df.copy()
    for c in REQUIRED_COLUMNS:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    diag = {}
    diag["rows_total"] = int(len(work))

    valid_xy = (
        work["x"].notna() & work["y"].notna() &
        np.isfinite(work["x"].to_numpy(dtype=np.float64, na_value=np.nan)) &
        np.isfinite(work["y"].to_numpy(dtype=np.float64, na_value=np.nan))
    )
    work_valid = work.loc[valid_xy].copy()
    diag["rows_invalid_xy"] = int(len(work) - len(work_valid))
    if len(work_valid) == 0:
        raise ValueError("No valid rows remain after x/y finite checks")

    x_rounded = np.rint(work_valid["x"].to_numpy(dtype=np.float64)).astype(np.int64)
    y_rounded = np.rint(work_valid["y"].to_numpy(dtype=np.float64)).astype(np.int64)

    work_valid["x_i"] = np.clip(x_rounded, 0, grid_size - 1).astype(np.int32)
    work_valid["y_i"] = np.clip(y_rounded, 0, grid_size - 1).astype(np.int32)
    return work_valid, diag


def build_channels_from_valid_df(
    work_valid: pd.DataFrame,
    grid_size: int = 400,
    min_pole_height: float = 20.0,
    max_pole_height: float = 52.0,
    count_threshold: float = 35.0,
):
    w = h = int(grid_size)
    contiguous_grid = np.zeros((h, w), dtype=np.float32)
    height_ok_grid = np.zeros((h, w), dtype=np.float32)
    count_ok_grid = np.zeros((h, w), dtype=np.float32)

    cont = work_valid["contiguous"].fillna(0.0).to_numpy(dtype=np.float32)
    max_cont = float(np.nanmax(cont)) if cont.size > 0 else 0.0
    gray = np.clip(cont / max_cont, 0.0, 1.0) if max_cont > 0 else np.zeros_like(cont, dtype=np.float32)

    pole_height = (
        work_valid["contiguousHeight"].fillna(0.0).to_numpy(dtype=np.float32) -
        work_valid["groundHeight"].fillna(0.0).to_numpy(dtype=np.float32)
    )
    height_ok = ((pole_height >= float(min_pole_height)) & (pole_height <= float(max_pole_height))).astype(np.float32)
    count_ok = (work_valid["count"].fillna(0.0).to_numpy(dtype=np.float32) > float(count_threshold)).astype(np.float32)

    xs = work_valid["x_i"].to_numpy(dtype=np.int32)
    ys = work_valid["y_i"].to_numpy(dtype=np.int32)

    contiguous_grid[ys, xs] = np.maximum(contiguous_grid[ys, xs], gray)
    height_ok_grid[ys, xs] = np.maximum(height_ok_grid[ys, xs], height_ok)
    count_ok_grid[ys, xs] = np.maximum(count_ok_grid[ys, xs], count_ok)

    rgb = np.stack([contiguous_grid, height_ok_grid, count_ok_grid], axis=0)
    rgb_u8 = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    return rgb_u8


def make_norm(num_channels: int, num_groups: int = 8):
    groups = num_groups
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.12):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            make_norm(out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout2d(p_drop),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            make_norm(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.12):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch, p_drop)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, p_drop=0.12):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, p_drop)

    def forward(self, x, skip):
        x = self.up(x)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class DualHeadUNetAux(nn.Module):
    def __init__(self, in_ch=3, base=24, p_drop=0.12):
        super().__init__()
        self.inc = ConvBlock(in_ch, base, p_drop)
        self.down1 = Down(base, base * 2, p_drop)
        self.down2 = Down(base * 2, base * 4, p_drop)
        self.down3 = Down(base * 4, base * 8, p_drop)
        self.up1 = Up(base * 8, base * 4, base * 4, p_drop)
        self.up2 = Up(base * 4, base * 2, base * 2, p_drop)
        self.up3 = Up(base * 2, base, base, p_drop)
        self.mask_head = nn.Conv2d(base, 1, kernel_size=1)
        self.center_head = nn.Conv2d(base, 1, kernel_size=1)
        self.aux_center_head = nn.Conv2d(base * 2, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        d1 = self.up1(x4, x3)
        d2 = self.up2(d1, x2)
        d3 = self.up3(d2, x1)
        mask_logits = self.mask_head(d3)
        center_logits = self.center_head(d3)
        aux_center_logits = self.aux_center_head(d2)
        aux_center_logits = F.interpolate(aux_center_logits, size=center_logits.shape[-2:], mode="bilinear", align_corners=False)
        return mask_logits, center_logits, aux_center_logits


class ProposalFilterNet(nn.Module):
    def __init__(self, in_ch=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 1, 96),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(96, 1),
        )

    def forward(self, x, peak_score):
        feat = self.features(x).flatten(1)
        z = torch.cat([feat, peak_score], dim=1)
        return self.head(z)


@dataclass
class ProposalExample:
    crop: torch.Tensor
    peak_score: float
    label: int
    image_idx: int
    x: int
    y: int


@dataclass
class ProposalCandidate:
    x: int
    y: int
    proposal_score: float
    filter_prob: float


def crop_with_pad_chw(x: torch.Tensor, cx: int, cy: int, crop_size: int) -> torch.Tensor:
    _, h, w = x.shape
    half = crop_size // 2
    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + crop_size, y0 + crop_size
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w)
    pad_b = max(0, y1 - h)
    x_pad = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
    x0 += pad_l
    x1 += pad_l
    y0 += pad_t
    y1 += pad_t
    return x_pad[:, y0:y1, x0:x1].contiguous()


def _extract_crop_from_image_u8(image_u8: torch.Tensor, x: int, y: int, crop_size: int):
    image_f = image_u8.float()
    crop = crop_with_pad_chw(image_f, x, y, crop_size=crop_size).clamp(0, 255).to(torch.uint8)
    return crop


def _extract_filter_crop_u8(image_u8: torch.Tensor, prob_map, x: int, y: int, crop_size: int):
    rgb_crop = _extract_crop_from_image_u8(image_u8, x, y, crop_size)
    hm = torch.from_numpy(prob_map).float().unsqueeze(0)
    hm_crop = crop_with_pad_chw(hm, x, y, crop_size=crop_size).clamp(0.0, 1.0)
    hm_u8 = (hm_crop * 255.0).round().to(torch.uint8)
    return torch.cat([rgb_crop, hm_u8], dim=0)


def local_maxima_2d(arr: np.ndarray, thresh: float, radius: int = 3) -> List[Tuple[int, int, float]]:
    h, w = arr.shape
    peaks = []
    for y in range(h):
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        for x in range(w):
            v = arr[y, x]
            if v < thresh:
                continue
            x0 = max(0, x - radius)
            x1 = min(w, x + radius + 1)
            patch = arr[y0:y1, x0:x1]
            if v >= patch.max():
                peaks.append((x, y, float(v)))
    peaks.sort(key=lambda t: t[2], reverse=True)
    kept = []
    for x, y, s in peaks:
        ok = True
        for kx, ky, _ in kept:
            if (kx - x) ** 2 + (ky - y) ** 2 <= radius ** 2:
                ok = False
                break
        if ok:
            kept.append((x, y, s))
    return kept


def _final_candidate_score(cand: ProposalCandidate) -> float:
    proposal = max(float(cand.proposal_score), 1e-8)
    filt = max(float(cand.filter_prob), 1e-8)
    geom = np.sqrt(proposal * filt)
    harm = (2.0 * proposal * filt) / (proposal + filt + 1e-8)
    blend = 0.15 * proposal + 0.85 * filt
    return float(max(harm, min(geom, blend)))


def _is_local_top(cand: ProposalCandidate, all_cands: List[ProposalCandidate], radius: int = 5) -> bool:
    s = _final_candidate_score(cand)
    r2 = radius * radius
    for other in all_cands:
        if other is cand:
            continue
        if (cand.x - other.x) ** 2 + (cand.y - other.y) ** 2 <= r2 and _final_candidate_score(other) > s:
            return False
    return True


def _has_nearby_accepted(cand: ProposalCandidate, accepted: List[Tuple[int, int, float]], radius: int = 7) -> bool:
    r2 = radius * radius
    for x, y, _ in accepted:
        if (cand.x - x) ** 2 + (cand.y - y) ** 2 <= r2:
            return True
    return False


def _apply_final_nms(pred_pts: List[Tuple[int, int, float]], radius: int = 5) -> List[Tuple[int, int, float]]:
    kept = []
    r2 = radius * radius
    for x, y, s in sorted(pred_pts, key=lambda t: t[2], reverse=True):
        ok = True
        for kx, ky, _ in kept:
            if (x - kx) ** 2 + (y - ky) ** 2 <= r2:
                ok = False
                break
        if ok:
            kept.append((x, y, s))
    return kept


def _blob_top_rescues(rescue_pool: List[ProposalCandidate], radius: int = 5) -> List[ProposalCandidate]:
    tops = []
    r2 = radius * radius
    for cand in sorted(rescue_pool, key=_final_candidate_score, reverse=True):
        keep = True
        for top in tops:
            if (cand.x - top.x) ** 2 + (cand.y - top.y) ** 2 <= r2:
                keep = False
                break
        if keep:
            tops.append(cand)
    return tops


@torch.inference_mode()
def run_filter_on_proposals(filter_model, proposal_examples: List[ProposalExample], device, batch_size=2048):
    filter_model.eval()
    probs = []
    for start in range(0, len(proposal_examples), batch_size):
        chunk = proposal_examples[start:start + batch_size]
        if not chunk:
            continue
        crops = torch.stack([ex.crop.float().div(255.0) for ex in chunk], dim=0).to(device, non_blocking=True)
        crops = crops.contiguous(memory_format=torch.channels_last)
        scores = torch.tensor([[ex.peak_score] for ex in chunk], dtype=torch.float32, device=device)
        with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = filter_model(crops, scores)
        p = torch.sigmoid(logits).flatten().cpu().numpy()
        probs.append(p)
    if len(probs) == 0:
        return np.array([], dtype=np.float32)
    return np.concatenate(probs)


@torch.inference_mode()
def build_candidates_for_image(
    proposal_model,
    filter_model,
    image_u8: np.ndarray,
    device,
    proposal_min_thresh=0.001,
    proposal_nms_radius=3,
    crop_size=48,
    filter_batch_size=2048,
    max_candidates_before_filter=12000,
):
    image_t = torch.from_numpy(image_u8)
    image = image_t.float().div(255.0).unsqueeze(0).to(device, non_blocking=True)
    image = image.contiguous(memory_format=torch.channels_last)

    with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
        _, center_logits, _ = proposal_model(image)
    if device.type == "cuda":
        torch.cuda.synchronize()

    prob = torch.sigmoid(center_logits)[0, 0].detach().cpu().numpy()
    cand_pts = local_maxima_2d(prob, thresh=proposal_min_thresh, radius=proposal_nms_radius)
    if len(cand_pts) > max_candidates_before_filter:
        cand_pts = cand_pts[:max_candidates_before_filter]

    examples = []
    for x, y, score in cand_pts:
        crop_u8 = _extract_filter_crop_u8(image_t, prob, int(x), int(y), crop_size)
        examples.append(ProposalExample(crop=crop_u8, peak_score=float(score), label=0, image_idx=0, x=int(x), y=int(y)))

    cls_probs = run_filter_on_proposals(filter_model, examples, device, batch_size=filter_batch_size) if examples else np.array([], dtype=np.float32)
    candidates = [
        ProposalCandidate(x=ex.x, y=ex.y, proposal_score=float(ex.peak_score), filter_prob=float(cp))
        for ex, cp in zip(examples, cls_probs)
    ]
    return candidates


def infer_candidates(candidates: List[ProposalCandidate], ckpt: Dict):
    proposal_thresh = float(ckpt.get("proposal_threshold", 0.05))
    filter_thresh = float(ckpt.get("filter_threshold", 0.5))
    rescue_proposal_thresh = float(ckpt.get("rescue_proposal_threshold", 1.1))
    rescue_filter_thresh = float(ckpt.get("rescue_filter_threshold", 1.1))
    final_nms_radius = int(ckpt.get("final_nms_radius", 4))
    rescue_limit_per_image = int(ckpt.get("rescue_limit_per_image", 1))
    max_detections_per_image = int(ckpt.get("max_detections_per_image", 999999))

    hard_accepted = []
    rescue_pool = []
    for cand in candidates:
        if cand.proposal_score >= proposal_thresh and cand.filter_prob >= filter_thresh:
            hard_accepted.append((cand.x, cand.y, _final_candidate_score(cand)))
        elif cand.proposal_score >= rescue_proposal_thresh and cand.filter_prob >= rescue_filter_thresh:
            rescue_pool.append(cand)

    hard_accepted = _apply_final_nms(hard_accepted, radius=final_nms_radius)

    rescues = []
    if rescue_limit_per_image > 0:
        for cand in _blob_top_rescues(rescue_pool, radius=4):
            if len(rescues) >= rescue_limit_per_image:
                break
            if not _is_local_top(cand, candidates, radius=4):
                continue
            if _has_nearby_accepted(cand, hard_accepted + rescues, radius=8):
                continue
            rescues.append((cand.x, cand.y, _final_candidate_score(cand)))

    pred_pts = _apply_final_nms(hard_accepted + rescues, radius=final_nms_radius)
    if len(pred_pts) > max_detections_per_image:
        pred_pts = sorted(pred_pts, key=lambda t: t[2], reverse=True)[:max_detections_per_image]
    return pred_pts


class PoleInferenceEngine:
    def __init__(
        self,
        model_path: str,
        device_str: str = "cuda",
        grid_size: int = 400,
        min_pole_height: float = 20.0,
        max_pole_height: float = 52.0,
        count_threshold: float = 35.0,
        proposal_nms_radius: int = 3,
        filter_batch_size: int = 2048,
        max_candidates_before_filter: int = 12000,
        positive_only_output: bool = True,
    ):
        self.model_path = model_path
        self.ckpt = torch.load(model_path, map_location="cpu")

        use_cuda = device_str.startswith("cuda") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.grid_size = int(grid_size)
        self.min_pole_height = float(min_pole_height)
        self.max_pole_height = float(max_pole_height)
        self.count_threshold = float(count_threshold)
        self.proposal_nms_radius = int(proposal_nms_radius)
        self.filter_batch_size = int(filter_batch_size)
        self.max_candidates_before_filter = int(max_candidates_before_filter)
        self.crop_size = int(self.ckpt.get("crop_size", 48))
        self.proposal_min_thresh = min(
            0.001,
            float(self.ckpt.get("proposal_threshold", 0.001)),
            float(self.ckpt.get("rescue_proposal_threshold", 1.1)),
        )
        self.positive_only_output = bool(positive_only_output)

        torch.set_num_threads(1)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        self.proposal_model = DualHeadUNetAux(in_ch=3, base=24, p_drop=0.12).to(self.device)
        self.proposal_model = self.proposal_model.to(memory_format=torch.channels_last)
        self.proposal_model.load_state_dict(self.ckpt["proposal_model_state"])
        self.proposal_model.eval()

        self.filter_model = ProposalFilterNet(in_ch=4).to(self.device)
        self.filter_model = self.filter_model.to(memory_format=torch.channels_last)
        self.filter_model.load_state_dict(self.ckpt["filter_model_state"])
        self.filter_model.eval()

        self.lock = threading.Lock()
        self._warmup()

    @torch.inference_mode()
    def _warmup(self):
        dummy = torch.zeros((1, 3, self.grid_size, self.grid_size), dtype=torch.float32, device=self.device)
        dummy = dummy.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", enabled=(self.device.type == "cuda")):
            _ = self.proposal_model(dummy)

        dummy_filter_x = torch.zeros((64, 4, self.crop_size, self.crop_size), dtype=torch.float32, device=self.device)
        dummy_filter_x = dummy_filter_x.contiguous(memory_format=torch.channels_last)
        dummy_filter_s = torch.zeros((64, 1), dtype=torch.float32, device=self.device)
        with torch.autocast(device_type="cuda", enabled=(self.device.type == "cuda")):
            _ = self.filter_model(dummy_filter_x, dummy_filter_s)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df0 = normalize_columns(df)
        if "pole" in df0.columns:
            df0 = df0.drop(columns=["pole"])

        work_valid, _ = validate_and_prepare_input_df(df0, grid_size=self.grid_size)
        rgb_u8 = build_channels_from_valid_df(
            work_valid,
            grid_size=self.grid_size,
            min_pole_height=self.min_pole_height,
            max_pole_height=self.max_pole_height,
            count_threshold=self.count_threshold,
        )

        with self.lock:
            candidates = build_candidates_for_image(
                self.proposal_model,
                self.filter_model,
                rgb_u8,
                self.device,
                proposal_min_thresh=self.proposal_min_thresh,
                proposal_nms_radius=self.proposal_nms_radius,
                crop_size=self.crop_size,
                filter_batch_size=self.filter_batch_size,
                max_candidates_before_filter=self.max_candidates_before_filter,
            )
            pred_pts = infer_candidates(candidates, self.ckpt)

        pred_set = set((int(x), int(y)) for x, y, _ in pred_pts)

        out_df = df0.copy()
        if "predicted_pole" in out_df.columns:
            out_df = out_df.drop(columns=["predicted_pole"])
        if "Predicted_Pole" in out_df.columns:
            out_df = out_df.drop(columns=["Predicted_Pole"])

        x_num = pd.to_numeric(out_df.get("x", pd.Series(index=out_df.index, dtype=float)), errors="coerce")
        y_num = pd.to_numeric(out_df.get("y", pd.Series(index=out_df.index, dtype=float)), errors="coerce")

        pred_col = []
        for xv, yv in zip(x_num, y_num):
            if pd.isna(xv) or pd.isna(yv) or not np.isfinite(xv) or not np.isfinite(yv):
                pred_col.append(0)
            else:
                key = (
                    int(np.clip(round(float(xv)), 0, self.grid_size - 1)),
                    int(np.clip(round(float(yv)), 0, self.grid_size - 1)),
                )
                pred_col.append(1 if key in pred_set else 0)

        out_df["predicted_pole"] = pred_col
        if self.positive_only_output:
            out_df = out_df.loc[out_df["predicted_pole"] == 1].copy()
        return out_df


class PoleInferenceServer(BaseHTTPRequestHandler):
    server_version = "PolePersistentInferenceHTTP/1.0"

    def _send_bytes(self, body: bytes, status: int = 200, content_type: str = "application/octet-stream", filename: str = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, status: int = 200):
        self._send_bytes(json_bytes(obj), status=status, content_type="application/json")

    def _error(self, status: int, detail: str):
        self._send_json({"status": "error", "detail": detail}, status=status)

    def log_message(self, fmt, *args):
        sys.stdout.write("[%s] %s - %s\n" % (now_ts(), self.address_string(), fmt % args))
        sys.stdout.flush()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json({
                "status": "ok",
                "time": now_ts(),
                "model_path": self.server.model_path,
                "device": str(self.server.engine.device),
                "grid_size": self.server.engine.grid_size,
                "positive_only_output": self.server.engine.positive_only_output,
                "filter_batch_size": self.server.engine.filter_batch_size,
                "max_candidates_before_filter": self.server.engine.max_candidates_before_filter,
            })
            return
        if path == "/":
            self._send_json({
                "status": "ok",
                "message": "POST CSV files to /predict-csv or /predict-csvs",
                "time": now_ts(),
            })
            return
        self._error(404, f"Unknown path: {path}")

    def _parse_multipart_files(self) -> List[Tuple[str, bytes]]:
        ctype, _ = cgi.parse_header(self.headers.get("Content-Type", ""))
        if ctype != "multipart/form-data":
            raise ValueError("Content-Type must be multipart/form-data")

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            },
        )

        files = []
        candidate_keys = []
        if "file" in form:
            candidate_keys.append("file")
        if "files" in form:
            candidate_keys.append("files")
        if not candidate_keys:
            raise ValueError('No uploaded file fields found. Use -F "file=@..." or -F "files=@..."')

        for key in candidate_keys:
            field = form[key]
            items = field if isinstance(field, list) else [field]
            for item in items:
                if not getattr(item, "filename", None):
                    continue
                data = item.file.read()
                files.append((Path(item.filename).name, data))

        if not files:
            raise ValueError("No uploaded files were found in the multipart form.")
        return files

    def _predict_many(self, uploaded: List[Tuple[str, bytes]], zip_output: bool):
        results = []
        for filename, raw in uploaded:
            df = parse_csv_and_strip_pole(raw, filename)
            out_df = self.server.engine.predict_dataframe(df)
            bio = io.StringIO(newline="")
            out_df.to_csv(bio, index=False)
            body = bio.getvalue().encode("utf-8")
            stem = Path(filename).stem
            results.append((f"{stem}_pred.csv", body))

        if not zip_output and len(results) == 1:
            filename, body = results[0]
            self._send_bytes(body, status=200, content_type="text/csv", filename=filename)
            return

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for filename, body in results:
                zf.writestr(filename, body)
        self._send_bytes(zip_buf.getvalue(), status=200, content_type="application/zip", filename="predicted_csvs.zip")

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            uploaded = self._parse_multipart_files()
            if path == "/predict-csv":
                return self._predict_many(uploaded, zip_output=False)
            if path == "/predict-csvs":
                return self._predict_many(uploaded, zip_output=True)
            self._error(404, f"Unknown path: {path}")
        except Exception as e:
            self._error(500, f"{e}\n\n{traceback.format_exc()}")


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--grid-size", type=int, default=400)
    ap.add_argument("--min-pole-height", type=float, default=20.0)
    ap.add_argument("--max-pole-height", type=float, default=52.0)
    ap.add_argument("--count-threshold", type=float, default=35.0)
    ap.add_argument("--proposal-nms-radius", type=int, default=3)
    ap.add_argument("--filter-batch-size", type=int, default=2048)
    ap.add_argument("--max-candidates-before-filter", type=int, default=12000)
    ap.add_argument("--positive-only-output", type=int, default=1, choices=[0, 1])
    return ap


def main():
    args = build_arg_parser().parse_args()
    model_path = resolve_model_path(args.model_path)

    engine = PoleInferenceEngine(
        model_path=model_path,
        device_str=args.device,
        grid_size=args.grid_size,
        min_pole_height=args.min_pole_height,
        max_pole_height=args.max_pole_height,
        count_threshold=args.count_threshold,
        proposal_nms_radius=args.proposal_nms_radius,
        filter_batch_size=args.filter_batch_size,
        max_candidates_before_filter=args.max_candidates_before_filter,
        positive_only_output=bool(args.positive_only_output),
    )

    server = ThreadingHTTPServer((args.host, args.port), PoleInferenceServer)
    server.engine = engine
    server.model_path = model_path

    print("[start] Persistent pole inference HTTP server")
    print(f"[start] host={args.host} port={args.port}")
    print(f"[start] model_path={model_path}")
    print(f"[start] device={engine.device}")
    print(f"[start] positive_only_output={engine.positive_only_output}")
    print("[start] endpoints: GET /health, POST /predict-csv, POST /predict-csvs")
    print(f"[start] warmup_done_at={now_ts()}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[stop] interrupted by user")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
