from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import re
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from transformers import SegformerConfig, SegformerForSemanticSegmentation


APP_ROOT = Path(__file__).resolve().parent
CHECKPOINTS = {
    "SegFormer-B0": APP_ROOT / "checkpoints" / "segformer-b0_best.pth",
    "DeepLabV3-MobileNetV3": (
        APP_ROOT / "checkpoints" / "deeplabv3-mobilenetv3_best.pth"
    ),
}
IMAGE_SIZE = (512, 512)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepLabBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = deeplabv3_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
        )
        self.model.classifier[-1] = nn.Conv2d(256, 1, 1)
        self.model.aux_classifier = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)["out"]


class SegFormerBinary(nn.Module):
    def __init__(self):
        super().__init__()
        config = SegformerConfig(num_labels=1)
        self.model = SegformerForSemanticSegmentation(config)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=images).logits
        return F.interpolate(
            logits,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


MODEL_BUILDERS = {
    "SegFormer-B0": SegFormerBinary,
    "DeepLabV3-MobileNetV3": DeepLabBinary,
}


def convert_segformer_state_dict(state_dict: dict) -> dict:
    """Map newer Transformers SegFormer names to older equivalent names."""

    def convert_key(key: str) -> str:
        key = key.replace(
            "model.decode_head.linear_projections.",
            "model.decode_head.linear_c.",
        )

        match = re.match(
            r"model\.segformer\.stages\.(\d+)\.patch_embeddings\.(.+)",
            key,
        )
        if match:
            return (
                "model.segformer.encoder.patch_embeddings."
                f"{match.group(1)}.{match.group(2)}"
            )

        match = re.match(
            r"model\.segformer\.stages\.(\d+)\.layer_norm\.(.+)",
            key,
        )
        if match:
            return (
                "model.segformer.encoder.layer_norm."
                f"{match.group(1)}.{match.group(2)}"
            )

        match = re.match(
            r"model\.segformer\.stages\.(\d+)\.blocks\.(\d+)\.(.+)",
            key,
        )
        if not match:
            return key

        stage, block, tail = match.groups()
        replacements = {
            "layernorm_before.": "layer_norm_1.",
            "layernorm_after.": "layer_norm_2.",
            "attention.q_proj.": "attention.self.query.",
            "attention.k_proj.": "attention.self.key.",
            "attention.v_proj.": "attention.self.value.",
            "attention.o_proj.": "attention.output.dense.",
            "attention.sequence_reduction.sequence_reduction.": (
                "attention.self.sr."
            ),
            "attention.sequence_reduction.layer_norm.": (
                "attention.self.layer_norm."
            ),
            "mlp.fc1.": "mlp.dense1.",
            "mlp.fc2.": "mlp.dense2.",
        }
        for source, destination in replacements.items():
            if tail.startswith(source):
                tail = destination + tail[len(source) :]
                break

        return f"model.segformer.encoder.block.{stage}.{block}.{tail}"

    return {convert_key(key): value for key, value in state_dict.items()}


def load_checkpoint_state(model: nn.Module, model_name: str, state_dict: dict):
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        if model_name != "SegFormer-B0":
            raise

    converted = convert_segformer_state_dict(state_dict)
    expected = model.state_dict()
    if set(converted) != set(expected):
        raise RuntimeError(
            "The SegFormer checkpoint is incompatible with the installed "
            "Transformers version."
        )
    for key in expected:
        if converted[key].shape != expected[key].shape:
            raise RuntimeError(
                f"SegFormer tensor shape mismatch for {key}: "
                f"{tuple(converted[key].shape)} != {tuple(expected[key].shape)}"
            )
    model.load_state_dict(converted)


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    checkpoint_path = CHECKPOINTS[model_name]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = MODEL_BUILDERS[model_name]()
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )
    load_checkpoint_state(model, model_name, checkpoint["model"])
    model = model.to(DEVICE)
    if DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.eval()

    metadata = {
        "epoch": checkpoint.get("epoch"),
        "validation": checkpoint.get("validation", {}),
    }
    return model, metadata


def preprocess(images: list[np.ndarray]) -> torch.Tensor:
    tensors = []
    for image in images:
        pil_image = Image.fromarray(image)
        pil_image = TF.resize(
            pil_image,
            IMAGE_SIZE,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        tensor = TF.normalize(
            TF.to_tensor(pil_image),
            IMAGENET_MEAN,
            IMAGENET_STD,
        )
        tensors.append(tensor)

    batch = torch.stack(tensors).to(DEVICE, non_blocking=True)
    if DEVICE.type == "cuda":
        batch = batch.contiguous(memory_format=torch.channels_last)
    return batch


@torch.inference_mode()
def predict_probabilities(model, images: list[np.ndarray]) -> list[np.ndarray]:
    batch = preprocess(images)
    with torch.autocast(
        device_type=DEVICE.type,
        dtype=torch.float16,
        enabled=DEVICE.type == "cuda",
    ):
        probabilities = torch.sigmoid(model(batch))

    results = []
    for probability, image in zip(probabilities[:, 0], images):
        height, width = image.shape[:2]
        resized = F.interpolate(
            probability[None, None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        results.append(resized.float().cpu().numpy())
    return results


def remove_small_regions(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    if minimum_area <= 0:
        return mask

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    cleaned = np.zeros_like(mask, dtype=bool)
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] >= minimum_area:
            cleaned[labels == component] = True
    return cleaned


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    opacity: float,
) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    color = np.zeros_like(overlay)
    color[..., 1] = 255
    overlay[mask] = (
        (1 - opacity) * overlay[mask] + opacity * color[mask]
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def create_probability_map(probability: np.ndarray) -> np.ndarray:
    heatmap = cv2.applyColorMap(
        np.clip(probability * 255, 0, 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def process_image(
    model,
    image: Image.Image,
    threshold: float,
    opacity: float,
    minimum_area: int,
):
    image_array = np.asarray(image.convert("RGB"))
    start = time.perf_counter()
    probability = predict_probabilities(model, [image_array])[0]
    elapsed = time.perf_counter() - start
    mask = remove_small_regions(probability >= threshold, minimum_area)

    return (
        image_array,
        create_probability_map(probability),
        (mask.astype(np.uint8) * 255),
        create_overlay(image_array, mask, opacity),
        elapsed,
    )


def open_video_writer(path: Path, fps: float, width: int, height: int):
    return cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )


def transcode_for_browser(source: Path, destination: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        return source

    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    try:
        subprocess.run(command, check=True)
        return destination
    except subprocess.CalledProcessError:
        return source


def process_video(
    model,
    uploaded_file,
    threshold: float,
    opacity: float,
    minimum_area: int,
    batch_size: int,
    progress_bar,
):
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.TemporaryDirectory() as temp_directory:
        temp_directory = Path(temp_directory)
        input_path = temp_directory / f"input{suffix}"
        raw_output_path = temp_directory / "segmented_raw.mp4"
        browser_output_path = temp_directory / "segmented.mp4"
        input_path.write_bytes(uploaded_file.getbuffer())

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("OpenCV could not open the uploaded video.")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = open_video_writer(raw_output_path, fps, width, height)
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("Could not create the output video.")

        frames_rgb = []
        processed_frames = 0
        inference_seconds = 0.0

        def flush_batch():
            nonlocal processed_frames, inference_seconds
            if not frames_rgb:
                return

            start = time.perf_counter()
            probabilities = predict_probabilities(model, frames_rgb)
            inference_seconds += time.perf_counter() - start

            for frame_rgb, probability in zip(frames_rgb, probabilities):
                mask = remove_small_regions(
                    probability >= threshold,
                    minimum_area,
                )
                overlay = create_overlay(frame_rgb, mask, opacity)
                writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                processed_frames += 1

            frames_rgb.clear()
            if frame_count > 0:
                progress_bar.progress(
                    min(processed_frames / frame_count, 1.0),
                    text=f"Processed {processed_frames}/{frame_count} frames",
                )

        try:
            while True:
                success, frame_bgr = capture.read()
                if not success:
                    break
                frames_rgb.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                if len(frames_rgb) >= batch_size:
                    flush_batch()
            flush_batch()
        finally:
            capture.release()
            writer.release()

        progress_bar.progress(1.0, text=f"Processed {processed_frames} frames")
        final_path = transcode_for_browser(
            raw_output_path,
            browser_output_path,
        )
        output_bytes = final_path.read_bytes()

    model_fps = (
        processed_frames / inference_seconds if inference_seconds > 0 else 0
    )
    return output_bytes, processed_frames, fps, model_fps


def show_model_metadata(metadata: dict):
    validation = metadata.get("validation", {})
    columns = st.columns(4)
    columns[0].metric("Checkpoint epoch", metadata.get("epoch", "N/A"))
    columns[1].metric("Validation IoU", f"{validation.get('iou', 0):.3f}")
    columns[2].metric("Validation Dice", f"{validation.get('dice', 0):.3f}")
    columns[3].metric("Validation recall", f"{validation.get('recall', 0):.3f}")


def run_app():
    st.set_page_config(
        page_title="Surgical Instrument Segmentation",
        layout="wide",
    )
    st.title("Surgical Instrument Segmentation")
    st.caption(
        "Upload unseen images or videos and compare the two fine-tuned models."
    )

    with st.sidebar:
        st.header("Inference settings")
        model_name = st.radio(
            "Model",
            options=list(MODEL_BUILDERS),
            index=0,
        )
        threshold = st.slider(
            "Mask threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.05,
        )
        opacity = st.slider(
            "Overlay opacity",
            min_value=0.10,
            max_value=0.90,
            value=0.45,
            step=0.05,
        )
        minimum_area = st.number_input(
            "Minimum mask region (pixels)",
            min_value=0,
            max_value=10000,
            value=100,
            step=50,
        )
        video_batch_size = st.select_slider(
            "Video inference batch size",
            options=[1, 2, 4, 8],
            value=4,
            help="Use 1-2 on CPU or if GPU memory is limited.",
        )
        st.write(f"Device: **{DEVICE}**")

    try:
        with st.spinner(f"Loading {model_name}..."):
            model, metadata = load_model(model_name)
    except Exception as error:
        st.error(str(error))
        st.stop()

    show_model_metadata(metadata)

    image_tab, video_tab = st.tabs(["Image", "Video"])

    with image_tab:
        image_file = st.file_uploader(
            "Upload an unseen image",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            key="image_upload",
        )
        if image_file is not None:
            image = Image.open(image_file)
            with st.spinner("Segmenting image..."):
                original, heatmap, mask, overlay, elapsed = process_image(
                    model,
                    image,
                    threshold,
                    opacity,
                    int(minimum_area),
                )

            st.metric("Inference time", f"{elapsed * 1000:.1f} ms")
            columns = st.columns(2)
            columns[0].image(original, caption="Original", use_container_width=True)
            columns[1].image(overlay, caption="Segmentation overlay", use_container_width=True)
            columns = st.columns(2)
            columns[0].image(heatmap, caption="Probability map", use_container_width=True)
            columns[1].image(mask, caption="Binary mask", use_container_width=True)

    with video_tab:
        video_file = st.file_uploader(
            "Upload an unseen video",
            type=["mp4", "avi", "mov", "mkv", "mpeg", "mpg"],
            key="video_upload",
        )
        if video_file is not None:
            st.video(video_file)
            if st.button("Process video", type="primary"):
                progress_bar = st.progress(0.0, text="Starting video inference...")
                try:
                    output, frame_count, input_fps, model_fps = process_video(
                        model,
                        video_file,
                        threshold,
                        opacity,
                        int(minimum_area),
                        int(video_batch_size),
                        progress_bar,
                    )
                except Exception as error:
                    st.error(f"Video processing failed: {error}")
                else:
                    st.success(f"Processed {frame_count} frames.")
                    columns = st.columns(2)
                    columns[0].metric("Input FPS", f"{input_fps:.1f}")
                    columns[1].metric("Model inference FPS", f"{model_fps:.1f}")
                    st.video(output)
                    st.download_button(
                        "Download segmented video",
                        data=output,
                        file_name=f"{Path(video_file.name).stem}_{model_name}_segmented.mp4",
                        mime="video/mp4",
                    )

    st.info(
        "These models were trained for binary instrument segmentation. "
        "The UI shows where an instrument is present, not its instrument type."
    )


if __name__ == "__main__":
    run_app()
