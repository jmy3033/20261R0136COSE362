
"""Minimal reproducible implementation for COSE362 final project.

Few-shot Hangul glyph generation with a Direct U-Net baseline and a lightweight
conditional DDPM. This file is intentionally self-contained so that the Colab
notebook can be reviewed and rerun from the GitHub repository.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEFAULT_REFERENCE_CHARS = ["가", "넋", "둘", "레", "묽", "빛", "산", "잊", "쾌", "흙"]
DEFAULT_TARGET_CHARS = ["값", "곳", "꽃", "눈", "몸", "들", "힘", "별"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def all_hangul_syllables() -> List[str]:
    return [chr(c) for c in range(ord("가"), ord("힣") + 1)]


def _default_char_subset(max_chars: int | None, seed: int = 42) -> List[str]:
    chars = all_hangul_syllables()
    if max_chars is None or max_chars >= len(chars):
        selected = chars
    else:
        rng = random.Random(seed)
        required = []
        for ch in DEFAULT_REFERENCE_CHARS + DEFAULT_TARGET_CHARS:
            if ch not in required:
                required.append(ch)
        remaining = [c for c in chars if c not in required]
        rng.shuffle(remaining)
        selected = required + remaining[: max(0, max_chars - len(required))]
        selected = selected[:max_chars]
    return selected


def _render_glyph(font_path: Path, char: str, glyph_size: int = 64) -> Image.Image:
    img = Image.new("L", (glyph_size, glyph_size), 255)
    draw = ImageDraw.Draw(img)
    # Start with a large font size and fit into the canvas.
    font_size = int(glyph_size * 0.86)
    font = ImageFont.truetype(str(font_path), font_size)
    try:
        bbox = draw.textbbox((0, 0), char, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (glyph_size - tw) / 2 - bbox[0]
        y = (glyph_size - th) / 2 - bbox[1]
    except Exception:
        w, h = draw.textsize(char, font=font)
        x = (glyph_size - w) / 2
        y = (glyph_size - h) / 2
    draw.text((x, y), char, fill=0, font=font)
    return img


def render_font_dataset(font_dir: str | Path, render_dir: str | Path, glyph_size: int = 64,
                        max_chars: int | None = 256, seed: int = 42,
                        chars: Sequence[str] | None = None) -> List[dict]:
    font_dir = Path(font_dir)
    render_dir = Path(render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)
    img_dir = render_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    font_paths = sorted([p for p in font_dir.iterdir() if p.suffix.lower() in {".ttf", ".otf", ".ttc"}])
    if not font_paths:
        raise FileNotFoundError(f"No font files found in {font_dir}")

    # Prefer NanumGothic as base/content font.
    font_paths = sorted(font_paths, key=lambda p: (0 if p.name.lower() == "nanumgothic.ttf" else 1, p.name.lower()))
    char_list = list(chars) if chars is not None else _default_char_subset(max_chars, seed)

    records: List[dict] = []
    for fi, font_path in enumerate(tqdm(font_paths, desc="render fonts")):
        font_id = f"font_{fi}"
        fdir = img_dir / font_id
        fdir.mkdir(parents=True, exist_ok=True)
        for ch in char_list:
            fname = f"U{ord(ch):04X}.png"
            out_path = fdir / fname
            try:
                img = _render_glyph(font_path, ch, glyph_size=glyph_size)
                img.save(out_path)
                records.append({
                    "font_id": font_id,
                    "font_path": str(font_path),
                    "font_file": font_path.name,
                    "char": ch,
                    "unicode": f"U{ord(ch):04X}",
                    "image_path": str(out_path),
                    "glyph_size": glyph_size,
                })
            except Exception as e:
                # Skip unsupported glyph/font errors.
                continue

    metadata = render_dir / "metadata.jsonl"
    with metadata.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records


def load_records(metadata_path: str | Path) -> List[dict]:
    metadata_path = Path(metadata_path)
    records = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def index_records(records: Sequence[dict]) -> Dict[str, Dict[str, dict]]:
    out: Dict[str, Dict[str, dict]] = {}
    for r in records:
        out.setdefault(str(r["font_id"]), {})[r["char"]] = r
    return out


def make_image_grid(images: Sequence[Image.Image], labels: Sequence[str] | None = None, cols: int = 8,
                    cell_size: int | None = None, pad: int = 8) -> Image.Image:
    imgs = [im.convert("L") if isinstance(im, Image.Image) else Image.open(im).convert("L") for im in images]
    if not imgs:
        return Image.new("L", (64, 64), 255)
    if cell_size is None:
        cell_size = max(imgs[0].size)
    label_h = 18 if labels is not None else 0
    rows = math.ceil(len(imgs) / cols)
    grid = Image.new("L", (cols * (cell_size + pad) + pad, rows * (cell_size + label_h + pad) + pad), 255)
    draw = ImageDraw.Draw(grid)
    for i, img in enumerate(imgs):
        r, c = divmod(i, cols)
        x = pad + c * (cell_size + pad)
        y = pad + r * (cell_size + label_h + pad)
        img = img.resize((cell_size, cell_size), Image.BILINEAR)
        grid.paste(img, (x, y))
        if labels is not None and i < len(labels):
            draw.text((x, y + cell_size + 2), str(labels[i]), fill=0)
    return grid


def save_image_grid(images: Sequence[Image.Image], path: str | Path, labels: Sequence[str] | None = None,
                    cols: int = 8) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_image_grid(images, labels=labels, cols=cols)
    grid.save(path)
    return path


@dataclass
class TrainConfig:
    metadata_path: str
    output_dir: str
    epochs: int = 999
    batch_size: int = 32
    max_steps: int = 1000
    base_channels: int = 32
    reference_count: int = 10
    reference_chars: Tuple[str, ...] = tuple(DEFAULT_REFERENCE_CHARS)
    diffusion_steps: int = 200
    beta_end: float = 0.02
    lr: float = 2e-4
    num_workers: int = 2
    recon_weight: float = 0.0
    image_size: int = 64
    test_font_ratio: float = 0.2
    split: str = "train"
    holdout_fonts: Tuple[str, ...] = ()
    seed: int = 42
    device: str | None = None


def split_fonts_by_id(records: Sequence[dict], test_font_ratio: float = 0.2,
                      holdout_fonts: Sequence[str] = (), base_font_id: str | None = None,
                      seed: int = 42):
    indexed = index_records(records)
    fonts = sorted(indexed)
    if base_font_id is None:
        base_font_id = fonts[0]
    style_fonts = [f for f in fonts if f != base_font_id]
    holdout = [f for f in holdout_fonts if f in style_fonts]
    remaining = [f for f in style_fonts if f not in holdout]
    rng = random.Random(seed)
    rng.shuffle(remaining)
    n_test = max(1, int(round(len(remaining) * test_font_ratio))) if remaining else 0
    test_fonts = sorted(set(holdout + remaining[:n_test]))
    train_fonts = [base_font_id] + sorted([f for f in style_fonts if f not in test_fonts])
    return train_fonts, test_fonts


def _load_tensor(path: str | Path, image_size: int = 64) -> torch.Tensor:
    img = Image.open(path).convert("L").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


class FewShotGlyphDataset(Dataset):
    def __init__(self, cfg: TrainConfig, split: str = "train"):
        self.cfg = cfg
        self.records = load_records(cfg.metadata_path)
        self.indexed = index_records(self.records)
        self.fonts = sorted(self.indexed)
        self.base_font_id = self.fonts[0]
        train_fonts, test_fonts = split_fonts_by_id(
            self.records, cfg.test_font_ratio, cfg.holdout_fonts, self.base_font_id, cfg.seed
        )
        style_fonts = train_fonts if split == "train" else test_fonts
        style_fonts = [f for f in style_fonts if f != self.base_font_id]
        if not style_fonts:
            style_fonts = [f for f in self.fonts if f != self.base_font_id]
        self.samples = []
        base_chars = set(self.indexed[self.base_font_id])
        for fid in style_fonts:
            chars = sorted(base_chars & set(self.indexed[fid]))
            for ch in chars:
                if ch in cfg.reference_chars:
                    continue
                self.samples.append((fid, ch))
        if not self.samples:
            raise RuntimeError("No training samples found. Check rendered metadata and reference chars.")

    def __len__(self):
        return len(self.samples)

    def _refs(self, fid: str) -> torch.Tensor:
        refs = []
        available = self.indexed[fid]
        fallback_char = sorted(available)[0]
        for ch in list(self.cfg.reference_chars)[: self.cfg.reference_count]:
            rec = available.get(ch, available[fallback_char])
            refs.append(_load_tensor(rec["image_path"], self.cfg.image_size))
        while len(refs) < self.cfg.reference_count:
            refs.append(refs[-1].clone())
        return torch.stack(refs, dim=0)  # [R,1,H,W]

    def __getitem__(self, idx):
        fid, ch = self.samples[idx]
        target = _load_tensor(self.indexed[fid][ch]["image_path"], self.cfg.image_size)
        content = _load_tensor(self.indexed[self.base_font_id][ch]["image_path"], self.cfg.image_size)
        refs = self._refs(fid)
        return {"target": target, "content": content, "refs": refs, "font_id": fid, "char": ch}


class SimpleCondUNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32, out_channels: int = 1):
        super().__init__()
        b = base_channels
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, b, 3, padding=1), nn.GroupNorm(4, b), nn.SiLU(), nn.Conv2d(b, b, 3, padding=1), nn.SiLU())
        self.down1 = nn.Conv2d(b, b * 2, 4, stride=2, padding=1)
        self.enc2 = nn.Sequential(nn.GroupNorm(8, b * 2), nn.SiLU(), nn.Conv2d(b * 2, b * 2, 3, padding=1), nn.SiLU())
        self.down2 = nn.Conv2d(b * 2, b * 4, 4, stride=2, padding=1)
        self.mid = nn.Sequential(nn.GroupNorm(8, b * 4), nn.SiLU(), nn.Conv2d(b * 4, b * 4, 3, padding=1), nn.SiLU(), nn.Conv2d(b * 4, b * 4, 3, padding=1), nn.SiLU())
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(nn.GroupNorm(8, b * 4), nn.SiLU(), nn.Conv2d(b * 4, b * 2, 3, padding=1), nn.SiLU())
        self.up1 = nn.ConvTranspose2d(b * 2, b, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(nn.GroupNorm(4, b * 2), nn.SiLU(), nn.Conv2d(b * 2, b, 3, padding=1), nn.SiLU(), nn.Conv2d(b, out_channels, 1))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        m = self.mid(self.down2(e2))
        u2 = self.up2(m)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return d1


def _ref_mean(refs: torch.Tensor) -> torch.Tensor:
    # refs: [B,R,1,H,W]
    return refs.mean(dim=1)


def _device(cfg: TrainConfig):
    return torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))


def train_unet_baseline(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = _device(cfg)
    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ds = FewShotGlyphDataset(cfg, split="train")
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    model = SimpleCondUNet(2, cfg.base_channels, 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = []
    step = 0
    pbar = tqdm(total=cfg.max_steps, desc="train baseline")
    while step < cfg.max_steps:
        for batch in loader:
            target = batch["target"].to(device)
            content = batch["content"].to(device)
            refs = batch["refs"].to(device)
            cond = torch.cat([content, _ref_mean(refs)], dim=1)
            pred = torch.sigmoid(model(cond))
            l1 = F.l1_loss(pred, target)
            mse = F.mse_loss(pred, target)
            loss = l1 + mse
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % 50 == 0 or step == 1:
                history.append({"step": float(step), "loss": float(loss.item()), "l1": float(l1.item()), "mse": float(mse.item())})
            pbar.update(1)
            if step >= cfg.max_steps: break
    pbar.close()
    ckpt = out_dir / "baseline_unet.pt"
    torch.save({"model_type": "baseline", "state_dict": model.state_dict(), "cfg": cfg.__dict__}, ckpt)
    return model, history, str(ckpt)


class DiffusionSchedule:
    def __init__(self, T: int, beta_end: float, device):
        self.T = T
        self.betas = torch.linspace(1e-4, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)


def train_conditional_ddpm(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = _device(cfg)
    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ds = FewShotGlyphDataset(cfg, split="train")
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    model = SimpleCondUNet(3, cfg.base_channels, 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = DiffusionSchedule(cfg.diffusion_steps, cfg.beta_end, device)
    history = []
    step = 0
    pbar = tqdm(total=cfg.max_steps, desc="train ddpm")
    while step < cfg.max_steps:
        for batch in loader:
            x0 = batch["target"].to(device)
            content = batch["content"].to(device)
            refs = batch["refs"].to(device)
            B = x0.size(0)
            t = torch.randint(0, sched.T, (B,), device=device)
            noise = torch.randn_like(x0)
            ab = sched.alpha_bars[t].view(B, 1, 1, 1)
            xt = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
            inp = torch.cat([xt, content, _ref_mean(refs)], dim=1)
            pred_noise = model(inp)
            loss_noise = F.mse_loss(pred_noise, noise)
            loss = loss_noise
            if cfg.recon_weight and cfg.recon_weight > 0:
                x0_hat = (xt - torch.sqrt(1 - ab) * pred_noise) / torch.sqrt(ab).clamp_min(1e-6)
                loss = loss + cfg.recon_weight * F.l1_loss(torch.clamp(x0_hat, 0, 1), x0)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % 50 == 0 or step == 1:
                history.append({"step": float(step), "loss": float(loss.item())})
            pbar.update(1)
            if step >= cfg.max_steps: break
    pbar.close()
    ckpt = out_dir / "conditional_ddpm.pt"
    torch.save({"model_type": "ddpm", "state_dict": model.state_dict(), "cfg": cfg.__dict__}, ckpt)
    return model, history, str(ckpt)


def _cfg_from_ckpt(d: dict) -> TrainConfig:
    cfgd = d.get("cfg", {})
    if isinstance(cfgd.get("reference_chars"), list):
        cfgd["reference_chars"] = tuple(cfgd["reference_chars"])
    if isinstance(cfgd.get("holdout_fonts"), list):
        cfgd["holdout_fonts"] = tuple(cfgd["holdout_fonts"])
    return TrainConfig(**{k: v for k, v in cfgd.items() if k in TrainConfig.__dataclass_fields__})


def _load_model_from_ckpt(ckpt, device=None):
    d = torch.load(ckpt, map_location=device or "cpu")
    cfg = _cfg_from_ckpt(d)
    model_type = d.get("model_type", "ddpm")
    in_ch = 2 if model_type == "baseline" else 3
    model = SimpleCondUNet(in_ch, cfg.base_channels, 1)
    model.load_state_dict(d["state_dict"], strict=False)
    if device is not None:
        model.to(device)
    model.eval()
    return model, cfg, model_type


def build_fewshot_batch(target_font: str, target_char: str, reference_chars: Sequence[str],
                        metadata_path: str | Path, reference_count: int = 10):
    records = load_records(metadata_path)
    indexed = index_records(records)
    base_font = sorted(indexed)[0]
    any_rec = next(iter(indexed[base_font].values()))
    image_size = int(any_rec.get("glyph_size", 64))
    target = _load_tensor(indexed[target_font][target_char]["image_path"], image_size).unsqueeze(0)
    content = _load_tensor(indexed[base_font][target_char]["image_path"], image_size).unsqueeze(0)
    refs = []
    fallback = sorted(indexed[target_font])[0]
    for ch in list(reference_chars)[:reference_count]:
        rec = indexed[target_font].get(ch, indexed[target_font][fallback])
        refs.append(_load_tensor(rec["image_path"], image_size))
    while len(refs) < reference_count:
        refs.append(refs[-1].clone())
    refs = torch.stack(refs, dim=0).unsqueeze(0)
    return {"target": target, "content": content, "refs": refs, "char": target_char, "target_font": target_font}


def evaluate_checkpoint_on_batch(ckpt, batch) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, model_type = _load_model_from_ckpt(ckpt, device)
    target = batch["target"].to(device)
    content = batch["content"].to(device)
    refs = batch["refs"].to(device)
    with torch.no_grad():
        if model_type == "baseline":
            pred = torch.sigmoid(model(torch.cat([content, _ref_mean(refs)], dim=1)))
        else:
            # Fast proxy: denoise a moderately noised target once for batch metric.
            t = torch.full((target.size(0),), max(0, cfg.diffusion_steps // 2), device=device, dtype=torch.long)
            sched = DiffusionSchedule(cfg.diffusion_steps, cfg.beta_end, device)
            noise = torch.randn_like(target)
            ab = sched.alpha_bars[t].view(target.size(0), 1, 1, 1)
            xt = torch.sqrt(ab) * target + torch.sqrt(1 - ab) * noise
            pred_noise = model(torch.cat([xt, content, _ref_mean(refs)], dim=1))
            pred = torch.clamp((xt - torch.sqrt(1 - ab) * pred_noise) / torch.sqrt(ab).clamp_min(1e-6), 0, 1)
        l1 = F.l1_loss(pred, target).item()
        mse = F.mse_loss(pred, target).item()
    return {"l1": l1, "mse": mse, "rmse": math.sqrt(mse)}


def pick_reference_image_paths(metadata_path: str | Path, target_font: str, reference_chars: Sequence[str]) -> List[Path]:
    indexed = index_records(load_records(metadata_path))
    paths = []
    for ch in reference_chars:
        if ch in indexed[target_font]:
            paths.append(Path(indexed[target_font][ch]["image_path"]))
    if not paths:
        raise ValueError("No reference images found for target font.")
    return paths


def _tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu().clamp(0, 1).squeeze().numpy()
    return Image.fromarray((x * 255).astype(np.uint8), mode="L")


def generate_glyphs(ckpt, reference_paths: Sequence[str | Path], target_chars: Sequence[str],
                    metadata_path: str | Path, base_font_id: str | None = None,
                    out_dir: str | Path | None = None) -> List[Image.Image]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, model_type = _load_model_from_ckpt(ckpt, device)
    records = load_records(metadata_path)
    indexed = index_records(records)
    if base_font_id is None:
        base_font_id = sorted(indexed)[0]
    out_dir = Path(out_dir or Path(cfg.output_dir) / "generated")
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = [_load_tensor(p, cfg.image_size) for p in reference_paths]
    while len(refs) < cfg.reference_count:
        refs.append(refs[-1].clone())
    refs = torch.stack(refs[:cfg.reference_count], dim=0).unsqueeze(0).to(device)  # [1,R,1,H,W]
    refm = _ref_mean(refs)
    images = []
    sched = DiffusionSchedule(cfg.diffusion_steps, cfg.beta_end, device)
    for ch in tqdm(list(target_chars), desc=f"generate {model_type}"):
        if ch not in indexed[base_font_id]:
            continue
        content = _load_tensor(indexed[base_font_id][ch]["image_path"], cfg.image_size).unsqueeze(0).to(device)
        with torch.no_grad():
            if model_type == "baseline":
                x = torch.sigmoid(model(torch.cat([content, refm], dim=1)))
            else:
                x = torch.randn(1, 1, cfg.image_size, cfg.image_size, device=device)
                for ti in reversed(range(cfg.diffusion_steps)):
                    t = torch.tensor([ti], device=device)
                    beta = sched.betas[t].view(1, 1, 1, 1)
                    alpha = sched.alphas[t].view(1, 1, 1, 1)
                    abar = sched.alpha_bars[t].view(1, 1, 1, 1)
                    eps = model(torch.cat([x, content, refm], dim=1))
                    mean = (x - beta / torch.sqrt(1 - abar).clamp_min(1e-6) * eps) / torch.sqrt(alpha).clamp_min(1e-6)
                    if ti > 0:
                        x = mean + torch.sqrt(beta) * torch.randn_like(x)
                    else:
                        x = mean
                x = torch.clamp(x, 0, 1)
        img = _tensor_to_pil(x)
        img.save(out_dir / f"{ch}_U{ord(ch):04X}.png")
        images.append(img)
    return images
