# Final Project Summary

## Project topic

Few-shot Hangul font generation using a Conditional DDPM.

## Motivation

Korean fonts require many complete Hangul syllables. Manually designing a full font library is expensive, especially for handwriting-like fonts. This project explores whether a small number of reference glyphs can guide a generative model to synthesize additional Hangul glyphs in the same style.

## Prior work

The project was motivated by the following model families:

- pix2pix: conditional image-to-image translation with U-Net and PatchGAN.
- CKFont: component-aware GAN for Korean font generation.
- DM-Font: dual-memory few-shot font generation using compositional script structure.
- Diff-Font: diffusion-based one-shot font generation with content/style/component conditioning.
- DK-Font: diffusion-based handwriting font generation pipeline.

## Method

The project implements:

1. A Direct U-Net baseline.
2. A Conditional DDPM for few-shot Hangul glyph generation.

The DDPM receives a noisy target glyph, a base/content glyph, and the mean representation of target-style reference glyphs. It is trained mainly with noise-prediction MSE, with optional reconstruction loss controlled by `recon_weight`.

## Dataset

The reported experiment used 16 Korean fonts rendered into 64×64 grayscale glyph images. Each font was rendered with 256 characters in the compact Colab setting. The reference characters were:

```text
가넋둘레묽빛산잊쾌흙
```

## Evaluation

The final Colab notebook includes evaluation for:

- L1
- MSE
- RMSE
- SSIM
- LPIPS
- FID

The notebook saves quantitative metric summaries and generated glyph outputs for inspection.

## Limitations

- The current setup is a Colab-scale prototype, not a full 11,172-glyph production font generator.
- The model generates glyph images, not a complete `.ttf` font file.
- The model does not explicitly guarantee correct initial/medial/final Hangul component composition.
- Font files and model checkpoints are not included in this repository.

## Future work

- Expand training/evaluation to a larger balanced Hangul set.
- Add explicit Hangul component conditioning.
- Generate all 11,172 complete Hangul syllables.
- Convert generated glyph images into a usable font file format.
