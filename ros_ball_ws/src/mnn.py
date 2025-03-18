import argparse
import time
import MNN
import MNN.cv as cv2
import MNN.numpy as np

def inference(model, img, precision, backend, thread, num_runs=100):
    config = {
        "precision": precision,
        "backend": backend,
        "numThread": thread
    }
    rt = MNN.nn.create_runtime_manager((config,))
    
    # Load the model
    net = MNN.nn.load_module_from_file(model, [], [], runtime_manager=rt)

    # Load and preprocess the image
    original_image = cv2.imread(img)
    if original_image is None:
        raise FileNotFoundError(f"Could not load image: {img}")

    ih, iw, _ = original_image.shape

    # Resize to 320x320 (letterbox)
    image = cv2.resize(
        original_image, (320, 320), 0.0, 0.0, cv2.INTER_LINEAR, -1, 
        [0.0, 0.0, 0.0], [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
    )

    # Convert BGR to RGB
    image = image[..., ::-1]  

    # Expand dimensions to (1, H, W, C) and convert to MNN tensor format
    input_var = np.expand_dims(image, 0)

    # Warm-up run (optional, helps stabilize performance readings)
    warmup_var = MNN.expr.convert(input_var, MNN.expr.NC4HW4)
    net.forward(warmup_var)

    # Measure inference time for multiple runs
    inference_times = []

    for _ in range(num_runs):
        input_var = np.expand_dims(image, 0)  # Reconstruct input tensor each time
        input_var = MNN.expr.convert(input_var, MNN.expr.NC4HW4)

        start_time = time.time()
        output_var = net.forward(input_var)
        end_time = time.time()

        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        inference_times.append(inference_time)

    # Compute average inference time
    avg_time = sum(inference_times) / len(inference_times)
    print(f"Average inference time over {num_runs} runs: {avg_time:.2f} ms")

    # Convert output tensor
    output_var = MNN.expr.convert(output_var, MNN.expr.NCHW)
    output_var = output_var.squeeze()

    # Output shape should be [84, num_boxes] (e.g., [84, 8400] for YOLO)
    cx = output_var[0]
    cy = output_var[1]
    w = output_var[2]
    h = output_var[3]

    print(cx[0])
    probs = output_var[4:]  # Class probabilities

    x0 = (cx - w * 0.5)   
    y0 = (cy - h * 0.5)   
    x1 = (cx + w * 0.5) 
    y1 = (cy + h * 0.5) 

    boxes = np.stack([x0, y0, x1, y1], axis=1)

    # Ensure values stay within valid range [0, original_width/height]
    boxes[:, 0] = np.clip(boxes[:, 0], 0, iw)  # x0
    boxes[:, 1] = np.clip(boxes[:, 1], 0, ih)  # y0
    boxes[:, 2] = np.clip(boxes[:, 2], 0, iw)  # x1
    boxes[:, 3] = np.clip(boxes[:, 3], 0, ih)  # y1

    # Get max probability and corresponding class ID
    scores = np.max(probs, axis=0)
    class_ids = np.argmax(probs, axis=0)

    # Apply Non-Maximum Suppression (NMS)
    result_ids = MNN.expr.nms(boxes, scores, 100, 0.45, 0.25)
    
    # If there are detections, process them
    if result_ids.shape[0] > 0:
        result_boxes = boxes[result_ids]
        result_scores = scores[result_ids]
        result_class_ids = class_ids[result_ids]

        # Draw detections on the original image
        for i in range(len(result_boxes)):
            x0, y0, x1, y1 = result_boxes[i].read_as_tuple()
            print(f"x0: {x0}, y0: {y0}, x1: {x1}, y1: {y1}")
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)  # Convert to int

            # Ensure valid box coordinates
            x0, x1 = max(0, min(iw, x0)), max(0, min(iw, x1))
            y0, y1 = max(0, min(ih, y0)), max(0, min(ih, y1))

            print(f"Detected Class ID: {result_class_ids[i]}, Confidence: {result_scores[i]:.2f}")
            print(f"x0: {x0}, y0: {y0}, x1: {x1}, y1: {y1}")

            cv2.rectangle(original_image, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # Save the detection results
        cv2.imwrite("res_320_fixed.jpg", original_image)
        print("Detection results saved to res_320_fixed.jpg")
    else:
        print("No detections found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLO11N MNN model")
    parser.add_argument("--img", type=str, required=True, help="Path to the input image")
    parser.add_argument("--precision", type=str, default="normal", help="Inference precision: normal, low, high, lowBF")
    parser.add_argument(
        "--backend",
        type=str,
        default="CPU",
        help="Inference backend: CPU, OPENCL, OPENGL, NN, VULKAN, METAL, TRT, CUDA, HIAI"
    )
    parser.add_argument("--thread", type=int, default=1, help="Number of threads to use for inference")

    args = parser.parse_args()
    inference(args.model, args.img, args.precision, args.backend, args.thread)
