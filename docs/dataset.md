# Dataset and Font List

## Dataset overview

This project uses Korean font files rendered into grayscale glyph images.

- Rendered glyph images in the main reported run: 4096
- Number of rendered fonts: 16
- Approximate glyphs per font: 256
- Image size: 64 × 64
- Base/content font: `NanumGothic.ttf`
- Evaluation target font in the reported run: `나눔손글씨 마고체.ttf`
- Reference characters: `가넋둘레묽빛산잊쾌흙`
- Example target characters: `값곳꽃눈몸들힘별`

## Font usage

The base/content font provides the structural content glyph image. The target handwriting fonts are used as style fonts for training and evaluation.

## Font files used

| No. | Font file name | Role |
|---:|---|---|
| 1 | NanumGothic.ttf | Base/content font |
| 2 | 나눔손글씨 강인한 위로.ttf | Rendered style font |
| 3 | 나눔손글씨 바른정신.ttf | Rendered style font |
| 4 | 나눔손글씨 백의의 천사.ttf | Rendered style font |
| 5 | 나눔손글씨 소방관의 기도.ttf | Rendered style font |
| 6 | 나눔손글씨 손편지체.ttf | Rendered style font |
| 7 | 나눔손글씨 아인맘 손글씨.ttf | Rendered style font |
| 8 | 나눔손글씨 안쌍체.ttf | Rendered style font |
| 9 | 나눔손글씨 고딕 아니고 고딩.ttf | Rendered style font |
| 10 | 나눔손글씨 고려글꼴.ttf | Rendered style font |
| 11 | 나눔손글씨 곰신체.ttf | Rendered style font |
| 12 | 나눔손글씨 규리의 일기.ttf | Rendered style font |
| 13 | 나눔손글씨 기쁨밝음.ttf | Rendered style font |
| 14 | 나눔손글씨 나는 이겨낸다.ttf | Rendered style font |
| 15 | 나눔손글씨 대광유리.ttf | Rendered style font |
| 16 | 나눔손글씨 마고체.ttf | Rendered style font / evaluation target font |

## License and distribution notice

The original font files are not included in this GitHub repository due to file-size and license considerations. Only the names of the fonts used in the experiment are documented.

To reproduce the experiment, place Korean font files under:

```text
/content/drive/MyDrive/ML fonts/fonts/
```

The notebook will render glyph images from the font files during execution.
