# Colab Reproduction Guide

This document explains how a grader can run the submitted code from GitHub in Google Colab.

## 1. Open the notebook

Open:

```text
notebooks/hangul_font_generation_conditional_ddpm_colab.ipynb
```

from the submitted public GitHub repository.

Colab URL pattern:

```text
https://colab.research.google.com/github/<Github-ID>/20261R0136COSE362/blob/main/notebooks/hangul_font_generation_conditional_ddpm_colab.ipynb
```

## 2. Use GPU runtime

In Colab, select:

```text
Runtime → Change runtime type → GPU
```

## 3. Run repository setup

The first code cell in the notebook contains:

```python
REPO_URL = "https://github.com/<Github-ID>/20261R0136COSE362.git"
PROJECT_DIR = "/content/20261R0136COSE362"
```

Replace `<Github-ID>` with the submitted GitHub ID and run the cell. This clones the full repository to `/content/20261R0136COSE362`, installs packages, and prepares the Python import path.

This step is necessary because opening a notebook from GitHub in Colab does not automatically clone the repository files such as `src/hangul_font_diffusion_mvp.py`.

## 4. Prepare fonts

Font files are not included in the repository. Place Korean font files under:

```text
/content/drive/MyDrive/ML fonts/fonts/
```

Accepted file types:

```text
.ttf, .otf, .ttc
```

The experiment used Nanum Gothic as the base/content font and Nanum handwriting fonts as style fonts. See `docs/dataset.md` for the full list.

## 5. Execute the notebook

Run the cells sequentially. The notebook will:

1. mount Google Drive,
2. import the project source code,
3. render glyph images,
4. train the Direct U-Net baseline,
5. train the Conditional DDPM,
6. generate target-style glyphs,
7. compute L1, MSE, RMSE, SSIM, LPIPS, and FID metrics.

Outputs are saved under:

```text
/content/hangul_font_mvp/outputs/
```

## 6. Notes

- Running with different fonts will produce different results.
- Model checkpoints and generated datasets are excluded from GitHub to keep the repository small.
- The training configuration is compact enough for Colab, but DDPM training can still take a substantial amount of time depending on the assigned GPU.
