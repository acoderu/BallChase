import MNN
import MNN.cv as mnn_cv
import MNN.numpy as mnn_np

import cv2  # We'll use OpenCV for final drawing
import numpy as np

def run_inference(model_path, image_path):
    """
    Loads a YOLO11N model via MNN's nn module, preprocesses an image,
    runs inference, decodes boxes, applies NMS, and saves the result.
    """

    # 1) Configure MNN runtime
    config = {
        "precision": "normal",   # could be "low", "high", "lowBF"
        "backend": "CPU",        # could be "OPENCL", "VULKAN", etc.
        "numThread": 4
    }
    rt = MNN.nn.create_runtime_manager((config,))

    # 2) Load YOLO model with empty input and output names (they'll be inferred)
    net = MNN.nn.load_module_from_file(model_path, [], [], runtime_manager=rt)

    # 3) Read the original image (OpenCV: BGR format)
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    oh, ow, _ = original_image.shape

    # 4) Pad to a square and resize to the model's trained size
    #    If your YOLO is trained at 320x320, we do a letterbox approach here.
    #    The snippet from the example used max(oh, ow), then scaled to 640x640.
    #    We'll adapt to 320 for YOLO11N_320, but feel free to change as needed.
    input_size = 320
    length = max(oh, ow)
    # Pad to square
    # MNN "cv2" and "numpy" are different from standard Python
    # We'll do it in MNN style for consistency
    padded_img = mnn_np.pad(
        mnn_np.array(original_image),
        [[0, length - oh], [0, length - ow], [0, 0]],
        mode="constant"
    )

    # 5) Resize + normalize with MNN's cv2 API
    #    (MNN.cv2.resize(, dst_size, fx, fy, interpolation, borderType, mean, normal))
    #    Note: MNN expects BGR→RGB flips, etc. We'll do manual flipping if needed.
    resized_img = mnn_cv.resize(
        padded_img,
        (input_size, input_size),
        0.0, 0.0,
        mnn_cv.INTER_LINEAR,
        -1,
        [0.0, 0.0, 0.0],  # mean (for normalization)
        [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]  # scale
    )

    # Convert from BGR to RGB
    resized_img = resized_img[..., ::-1]

    # 6) Expand dims to (1, H, W, C), then convert to NC4HW4
    input_var = mnn_np.expand_dims(resized_img, 0)        # shape: (1, H, W, C)
    input_var = MNN.expr.convert(input_var, MNN.expr.NC4HW4)

    # 7) Forward inference
    output_var = net.forward(input_var)
    # Convert output to NCHW for easier indexing
    output_var = MNN.expr.convert(output_var, MNN.expr.NCHW)
    # Squeeze out batch dimension if it exists
    output_var = output_var.squeeze()

    # 8) Suppose the model outputs shape [84, X] => YOLO for 80 classes: (cx, cy, w, h) + 80 scores = 84
    #    The second dim is total candidate boxes. Let's read the shape:
    out_shape = list(output_var.shape)
    print(f"Output shape: {out_shape}")

    # Example: [84, 8400] => 84 channels, 8400 candidates
    # Split them
    cx = output_var[0]
    cy = output_var[1]
    ww = output_var[2]
    hh = output_var[3]
    probs = output_var[4:]  # shape: [80, 8400]

    # 9) Convert (cx, cy, w, h) => [x0, y0, x1, y1], all in [0,1] range
    x0 = cx - ww * 0.5
    y0 = cy - hh * 0.5
    x1 = cx + ww * 0.5
    y1 = cy + hh * 0.5
    boxes = mnn_np.stack([x0, y0, x1, y1], axis=1)

    # Clamp box coords to [0,1]
    boxes = MNN.expr.clip(boxes, 0, 1)

    # 10) Find per-box max class probability + class ID
    scores = mnn_np.max(probs, axis=0)         # shape: (8400,)
    class_ids = mnn_np.argmax(probs, axis=0)   # shape: (8400,)

    # 11) Non-Max Suppression with MNN
    # MNN's built-in NMS: MNN.expr.nms(boxes, scores, max_detections, iou_threshold, score_threshold)
    # Adjust thresholds to your liking
    # shape: [N]
    nms_indices = MNN.expr.nms(boxes, scores, maxDet=100, iou=0.45, score=0.25)

    # Convert MNN Tensors to standard Python
    nms_indices_np = np.array(nms_indices.read())  # shape: (num_dets,)
    boxes_np = np.array(boxes.read())
    scores_np = np.array(scores.read())
    class_ids_np = np.array(class_ids.read())

    # 12) Draw detections on the original (un-padded) image
    for idx in nms_indices_np:
        # coords in range [0,1], scale them
        # read_as_tuple => converting to python
        x0_i, y0_i, x1_i, y1_i = boxes_np[idx]
        conf_i = scores_np[idx]
        cls_i = int(class_ids_np[idx])

        # Scale to padded square size
        # Our padded square is dimension `length`
        # Then we scale back to the original image size
        x0_i = x0_i * length
        y0_i = y0_i * length
        x1_i = x1_i * length
        y1_i = y1_i * length

        # Because we only padded bottom or right, we clamp to original width/height
        x0_i = max(0, min(ow, x0_i))
        y0_i = max(0, min(oh, y0_i))
        x1_i = max(0, min(ow, x1_i))
        y1_i = max(0, min(oh, y1_i))

        print(f"Detected Class: {cls_i}, Confidence: {conf_i:.2f}")
        cv2.rectangle(original_image, (int(x0_i), int(y0_i)), (int(x1_i), int(y1_i)), (0, 0, 255), 2)
        cv2.putText(
            original_image,
            f"ID:{cls_i} ({conf_i:.2f})",
            (int(x0_i), int(y0_i) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

    # 13) Save results
    cv2.imwrite("res.jpg", original_image)
    print("Inference complete. Results saved to res.jpg")

if __name__ == "__main__":
    # Edit these paths if needed
    MODEL_PATH = "yolo11n_320.mnn"
    IMAGE_PATH = "ros2_camera_feed.jpg"
    run_inference(MODEL_PATH, IMAGE_PATH)
