import ncnn
import torch
import numpy as np
import cv2

# Paths
MODEL_PARAM_PATH = "yolo12n_ncnn_model_320/model.ncnn.param"
MODEL_BIN_PATH = "yolo12n_ncnn_model_320/model.ncnn.bin"
IMAGE_PATH = "ros2_camera_feed.jpg"

# Constants
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 320

def preprocess_image(image_path):
    """Load image and preprocess for NCNN inference using Torch."""
    img = cv2.imread(image_path)  # Read the image
    if img is None:
        print(f"❌ Failed to load image: {image_path}")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img_resized = cv2.resize(img, (CAMERA_WIDTH, CAMERA_HEIGHT))  # Resize to model input size

    # Convert to Torch Tensor
    img_tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    img_tensor /= 255.0  # Normalize to [0,1]

    print(f"✅ Final Processed Shape for NCNN: {img_tensor.shape}, dtype: {img_tensor.dtype}")

    return img_tensor

def test_inference():
    """Run inference using NCNN with Torch preprocessed input."""
    torch.manual_seed(0)

    # Load Model
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False  # Set True if using GPU

    if net.load_param(MODEL_PARAM_PATH) != 0:
        print("❌ Failed to load param")
        return
    if net.load_model(MODEL_BIN_PATH) != 0:
        print("❌ Failed to load bin")
        return

    # Preprocess Image
    img_tensor = preprocess_image(IMAGE_PATH)
    if img_tensor is None:
        return

    with net.create_extractor() as ex:
        # Convert to NCNN Mat and Clone to Ensure Memory Alignment
        ex.input("in0", ncnn.Mat(img_tensor.numpy()).clone())  # 🔹 Clone ensures proper handling

        ret, out0 = ex.extract("out0")
        if ret != 0:
            print("❌ Extract out0 failed")
        else:
            print("✅ Inference done, shape =", np.array(out0).shape)

test_inference()
