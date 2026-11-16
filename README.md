# JPEG-NAFNet

CP461 Introduction to Computer Vision — restoring JPEG-compressed and blurred images by fine-tuning
NAFNet (Chen et al., ECCV 2022) on REDS + RealBlur-J + synthetic JPEG (QF 10-50).

- `app.py` — Streamlit demo (upload an image, or simulate JPEG damage on a sharp image)
- `model.py` — NAFNet architecture (same as training notebook)
- `tiling.py` — tile-based inference so large images fit in limited memory
- `convert_fp16.py` — shrink the fine-tuned weights from 272 MB to ~136 MB

Model weights are not stored in this repo (too large for GitHub); the app downloads them from Hugging Face Hub.

## Run locally

```
pip install torch --index-url https://download.pytorch.org/whl/cu121   # NVIDIA GPU
pip install -r requirements-local.txt
python convert_fp16.py        # once, creates nafnet_ft_fp16.pth
streamlit run app.py
```

## Results (300 test images)

| Model | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Degraded input | 29.51 | 0.831 | 0.209 |
| NAFNet-REDS pretrained | 30.68 | 0.866 | 0.190 |
| **NAFNet fine-tuned (ours)** | **32.27** | **0.896** | **0.158** |
