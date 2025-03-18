import ncnn
import numpy as np
import cv2

IMAGE_PATH = "ros2_camera_feed.jpg"  # Use the correct file path

# Constants
# CAMERA_WIDTH = 320
# CAMERA_HEIGHT = 320
# def preprocess_image(image_path):
#     """Load image and preprocess for NCNN inference."""
#     img = cv2.imread(image_path)  # Read the image
#     if img is None:
#         print(f"❌ Failed to load image: {image_path}")
#         return None

#     print(f"✅ Original Image Shape: {img.shape}, dtype: {img.dtype}")  # (H, W, C)

#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
#     img_resized = cv2.resize(img, (CAMERA_WIDTH, CAMERA_HEIGHT))  # Resize

#     print(f"✅ Resized Image Shape: {img_resized.shape}, dtype: {img_resized.dtype}")  # (320, 320, 3)

#     # 🔹 **Explicitly Convert to (C, H, W)**
#     img_transposed = np.transpose(img_resized, (2, 0, 1))  # (H, W, C) -> (C, H, W)
#     img_transposed = np.random.rand(3, 320, 320).astype(np.float32)
#     #img_normalized = img_transposed.astype(np.float32) / 255.0  # Normalize
#     mat_in = ncnn.Mat(img_transposed)
#     #print(f"✅ Final Processed Shape for NCNN: {img_normalized.shape}, dtype: {img_normalized.dtype}")

#     # 🔹 **Fix: Create `ncnn.Mat` with `from_pixels`**
#     #mat_in = ncnn.Mat.from_pixels(
#     #    img_resized, 
#     #    ncnn.Mat.PixelType.PIXEL_RGB, 
#     #    CAMERA_WIDTH, CAMERA_HEIGHT
#     #)

#     # 🔹 **Normalize**
#     #mat_in.substract_mean_normalize([], [1/255.0, 1/255.0, 1/255.0])

#     return mat_in

def testMe():
    img_transposed = np.random.rand(3, 320, 320).astype(np.float32)
    #img_normalized = img_transposed.astype(np.float32) / 255.0  # Normalize
    mat_in = ncnn.Mat(img_transposed)
    return mat_in
def test_inference():
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False  # or True, to check GPU

    ret = net.load_param("yolov8n_ncnn_model_320/model.ncnn.param")
    if ret != 0:
        print("❌ Failed to load param")

    ret = net.load_model("yolov8n_ncnn_model_320/model.ncnn.bin")
    if ret != 0:
        print("❌ Failed to load bin")

    #mat_in = preprocess_image(IMAGE_PATH)    
    #data = np.random.rand(3, 320, 320).astype(np.float32)
    #mat_in = ncnn.Mat(data)
    mat_in = testMe()
    ex = net.create_extractor()
    ex.input("in0", mat_in)

    ret, out0 = ex.extract("out0")
    if ret != 0:
        print("❌ Extract out0 failed")
    else:
        print("✅ Inference done, shape =", np.array(out0).shape)

test_inference()