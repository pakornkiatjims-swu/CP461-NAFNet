"""JPEG-NAFNet demo: restore JPEG-compressed / blurred images with a fine-tuned NAFNet."""
import io
import math
import os
import time

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from model import build_model
from tiling import psnr_u8, tile_restore

# ---- config -----------------------------------------------------------------------------
# After uploading the weights to Hugging Face Hub, put "<your-username>/<model-repo>" here.
HF_REPO_ID = "YOUR-HF-USERNAME/jpeg-nafnet"
HF_FILENAME = "nafnet_ft_fp16.pth"
LOCAL_CANDIDATES = ["nafnet_ft.pth", "nafnet_ft_fp16.pth"]   # used first if present (local run)
BASE = os.path.dirname(os.path.abspath(__file__))


def pick_device():
    """Use the GPU only if a real convolution runs on it; otherwise fall back to CPU."""
    if not torch.cuda.is_available():
        return "cpu", None
    try:
        torch.nn.functional.conv2d(torch.zeros(1, 3, 8, 8, device="cuda"),
                                   torch.zeros(4, 3, 3, 3, device="cuda"))
        torch.cuda.synchronize()
        return "cuda", None
    except Exception as e:                      # e.g. "no kernel image is available"
        return "cpu", str(e).splitlines()[0]


if st.session_state.get("gpu_failed"):      # the real model already failed on GPU in this session
    DEVICE, GPU_ERROR = "cpu", st.session_state["gpu_failed"]
else:
    DEVICE, GPU_ERROR = pick_device()
ON_GPU = DEVICE == "cuda"
if not ON_GPU:
    torch.set_num_threads(max(1, os.cpu_count() or 1))

st.set_page_config(page_title="JPEG-NAFNet", page_icon="🖼️", layout="wide")


@st.cache_resource(show_spinner="กำลังโหลดโมเดล (ครั้งแรกอาจใช้เวลาสักครู่)...")
def load_model(device):
    """Returns (model, None) or (None, error) if the model cannot run on `device`."""
    path = next((os.path.join(BASE, p) for p in LOCAL_CANDIDATES
                 if os.path.exists(os.path.join(BASE, p))), None)
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    ck = torch.load(path, map_location="cpu")
    sd = {k: v.float() for k, v in ck.get("params", ck).items()}
    model = build_model()
    model.load_state_dict(sd, strict=True)
    model = model.eval().to(device)
    if device == "cuda":                      # dry run with the real model: every kernel must exist
        try:
            with torch.inference_mode():
                model(torch.zeros(1, 3, 64, 64, device="cuda"))
            torch.cuda.synchronize()
        except Exception as e:
            return None, str(e).splitlines()[0]
    return model, None


@torch.inference_mode()
def run_tile(model, arr):
    """arr: HxWx3 float32 [0,1] -> HxWx3 float32 [0,1]."""
    x = torch.from_numpy(arr).permute(2, 0, 1)[None].to(next(model.parameters()).device)
    h, w = x.shape[2:]
    ph, pw = (-h) % 16, (-w) % 16
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    y = model(x)[:, :, :h, :w].clamp(0, 1)
    return y[0].permute(1, 2, 0).float().cpu().numpy()


def limit_megapixels(img, max_mp):
    w, h = img.size
    if w * h <= max_mp * 1e6:
        return img, False
    s = math.sqrt(max_mp * 1e6 / (w * h))
    return img.resize((max(32, int(w * s)), max(32, int(h * s))), Image.LANCZOS), True


def jpeg_degrade(img, qf):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(qf))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def to_png_bytes(arr_u8):
    buf = io.BytesIO()
    Image.fromarray(arr_u8).save(buf, "PNG")
    return buf.getvalue()


# ---- UI ---------------------------------------------------------------------------------
st.title("🖼️ JPEG-NAFNet")
st.caption("ฟื้นฟูภาพที่ถูกบีบอัด JPEG และภาพเบลอ ด้วย NAFNet ที่ fine-tune บน REDS + RealBlur-J + JPEG สังเคราะห์ · CP461")

tab_demo, tab_about = st.tabs(["ทดลองใช้งาน", "เกี่ยวกับโมเดล"])

with st.sidebar:
    st.header("ตั้งค่า")
    st.write(f"อุปกรณ์ประมวลผล: **{'GPU (CUDA)' if ON_GPU else 'CPU'}**")
    st.caption(f"PyTorch {torch.__version__}" + (f" · {torch.cuda.get_device_name(0)}" if ON_GPU else ""))
    if GPU_ERROR:
        st.error("พบ GPU แต่ PyTorch รุ่นที่ติดตั้งใช้กับการ์ดนี้ไม่ได้ จึงใช้ CPU แทน\n\n"
                 f"สาเหตุ: {GPU_ERROR}")
    mode = st.radio("โหมด", ["ภาพเสียอยู่แล้ว (เบลอ/JPEG)", "ภาพคมชัด → จำลอง JPEG แล้วฟื้นฟู"])
    qf = 15
    if mode.startswith("ภาพคมชัด"):
        qf = st.slider("JPEG quality (ยิ่งต่ำยิ่งเสียมาก)", 5, 60, 15)
    max_mp = st.slider("ขนาดภาพสูงสุด (เมกะพิกเซล)", 0.1, 2.5 if ON_GPU else 1.0,
                       1.0 if ON_GPU else 0.3, step=0.1)
    if not ON_GPU:
        st.info("รันบน CPU: ภาพใหญ่จะช้า จึงจำกัดขนาดภาพไว้ ลองเพิ่มได้ถ้ารอไหว")

with tab_demo:
    up = st.file_uploader("อัปโหลดภาพ (JPG / PNG / WEBP)", type=["jpg", "jpeg", "png", "webp"])
    if up is not None:
        src = ImageOps.exif_transpose(Image.open(up)).convert("RGB")
        if min(src.size) < 64:
            st.error("ภาพเล็กเกินไป (ด้านสั้นต้องอย่างน้อย 64 พิกเซล)")
            st.stop()
        src, resized = limit_megapixels(src, max_mp)
        if resized:
            st.warning(f"ภาพใหญ่เกินกำหนด จึงย่อเหลือ {src.size[0]}×{src.size[1]} px ก่อนประมวลผล")

        if mode.startswith("ภาพคมชัด"):
            gt = np.array(src)
            inp_img = jpeg_degrade(src, qf)
        else:
            gt = None
            inp_img = src
        inp = np.array(inp_img)

        if st.button("ฟื้นฟูภาพ", type="primary"):
            model, err = load_model(DEVICE)
            if model is None:                  # GPU unusable -> switch to CPU and reload the page
                st.session_state["gpu_failed"] = err
                st.rerun()
            bar = st.progress(0.0, text="กำลังประมวลผล...")
            t0 = time.time()
            try:
                out = tile_restore(inp.astype(np.float32) / 255.0, lambda a: run_tile(model, a),
                                   tile=768 if ON_GPU else 384, overlap=32,
                                   progress=lambda i, n: bar.progress(i / n, text=f"กำลังประมวลผล {i}/{n}"))
            except RuntimeError as e:
                if ON_GPU and "CUDA" in str(e):
                    st.session_state["gpu_failed"] = str(e).splitlines()[0]
                    st.rerun()
                raise
            out_u8 = (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)
            bar.empty()
            st.session_state["res"] = dict(inp=inp, out=out_u8, gt=gt, sec=time.time() - t0,
                                           name=os.path.splitext(up.name)[0])

        res = st.session_state.get("res")
        if res is not None and res["inp"].shape == inp.shape and np.array_equal(res["inp"], inp):
            c1, c2 = st.columns(2)
            c1.image(res["inp"], caption="ก่อน (ภาพเสีย)", use_container_width=True)
            c2.image(res["out"], caption="หลัง (JPEG-NAFNet)", use_container_width=True)
            msg = f"ใช้เวลา {res['sec']:.1f} วินาที บน {'GPU' if ON_GPU else 'CPU'}"
            if res["gt"] is not None:
                p0, p1 = psnr_u8(res["gt"], res["inp"]), psnr_u8(res["gt"], res["out"])
                msg += f" · PSNR เทียบภาพคมชัดต้นฉบับ: {p0:.2f} → {p1:.2f} dB ({p1 - p0:+.2f})"
            st.success(msg)
            st.download_button("ดาวน์โหลดภาพที่ฟื้นฟู (PNG)", to_png_bytes(res["out"]),
                               file_name=f"{res['name']}_restored.png", mime="image/png")
    else:
        st.info("อัปโหลดภาพเพื่อเริ่มต้น · หรือเลือกโหมด \"ภาพคมชัด → จำลอง JPEG\" ที่แถบด้านซ้าย "
                "เพื่อดูว่าโมเดลกู้ภาพที่ถูกบีบอัดหนักๆ ได้แค่ไหน และดู PSNR เทียบต้นฉบับ")

with tab_about:
    st.subheader("โมเดลและการฝึก")
    st.markdown(
        "- **สถาปัตยกรรม:** NAFNet width64 (67.9M พารามิเตอร์), Chen et al., ECCV 2022\n"
        "- **เริ่มจาก:** NAFNet-REDS-width64 (pretrained) แล้ว fine-tune ด้วย PSNR loss\n"
        "- **ข้อมูลฝึก:** REDS 40% · RealBlur-J 40% · JPEG สังเคราะห์ QF10–50 20%\n"
        "- **ผลการทดสอบ (300 ภาพ):**"
    )
    st.table({
        "วิธี": ["ภาพเสีย (ไม่แก้ไข)", "NAFNet-REDS pretrained", "NAFNet fine-tuned (ของเรา)"],
        "PSNR ↑": [29.51, 30.68, 32.27],
        "SSIM ↑": [0.831, 0.866, 0.896],
        "LPIPS ↓": [0.209, 0.190, 0.158],
    })
    st.markdown(
        "**ข้อจำกัด:** ภาพ JPEG คุณภาพสูงที่มีพื้นผิวละเอียด (หญ้า ใบไม้) อาจถูกทำให้เรียบเกินไป · "
        "เว็บนี้ย่อภาพและประมวลผลเป็นช่อง (tile) เพื่อประหยัดหน่วยความจำ ผลจึงอาจต่างจากการรันภาพเต็มบน GPU เล็กน้อย"
    )
