# 20261R0136COSE362
20261R0136COSE362
한글 Few-shot 폰트 이미지 생성 (조건부 Diffusion MVP)
Machine Learning Project — 기계학습 K2
소수의 reference glyph만으로 목표 폰트 스타일의 새로운 한글 glyph 이미지를 생성하는 MVP입니다. content glyph와 10개의 reference glyph를 입력받아 target style glyph를 생성하며, `p(target | content, references)`를 근사하는 것이 학습 목표입니다. Direct U-Net baseline과 Conditional DDPM 두 모델을 학습·비교합니다.
⚠️ 폰트 파일 안내
학습에 쓰는 폰트(.ttf / .otf)는 라이선스·용량 문제로 이 저장소에 포함하지 않습니다. 아래에서 직접 받아주세요.
🔗 https://clova.ai/handwriting/list.html
받은 폰트는 GitHub에 다시 올리지 말고 로컬 또는 Google Drive의 폴더에 넣은 뒤, 노트북 Cell 6의 `MY\_FONT\_DIR` 경로를 본인 폴더로 수정하세요. base font인 NanumGothic은 Colab 시스템 폰트에서 자동으로 추가됩니다.
실행 방법
Google Colab + GPU 런타임 기준입니다.
노트북과 `src/hangul\_font\_diffusion\_mvp.py`를 준비합니다.
위 링크에서 폰트를 받아 Drive 폴더에 넣습니다.
Colab에서 노트북을 열고 GPU 런타임으로 설정합니다.
셀을 순서대로 실행합니다. Cell 3에서 `ROOT`/`SRC` 경로, Cell 6에서 `MY\_FONT\_DIR`을 본인 환경에 맞게 수정하세요.
결과 요약과 생성 이미지는 `outputs/` 및 Drive의 `team\_share\_results/` 폴더에 저장됩니다.
저장소 구조
```text
.
├── README.md
├── hangul\_font\_diffusion\_colab\_mvp\_teamshare\_v4.ipynb   # 메인 노트북 (실행 진입점)
└── src/
    └── hangul\_font\_diffusion\_mvp.py                     # 데이터 렌더링·학습·평가 모듈
```
결과 요약
Direct U-Net은 구조 정렬(SSIM, L1)에서, Conditional DDPM은 생성 품질(RMSE, LPIPS, FID)에서 우세했습니다. 자세한 정량 결과와 논문 대비 비교는 ppt 자료를 참고하세요.
팀원 (기계학습 K2)
홍재원 (2019160115) · 전진서 (2023140108) · 우진성 (2021150407) · 정무영 (2019320112) · 신효철 (2024320152)
