import MNN  
import numpy as np  
import cv2  
from MNN import expr  

def process_yolo_output(output_data, input_size, original_size, num_classes=80, conf_threshold=0.5, nms_threshold=0.5):  
    """  
    Process YOLOv11 output tensor to detect objects  
    """  
    # Reshape and transpose output (assuming [1, 84, n] format)  
    output_data = output_data.reshape(84, -1)  
    cx = output_data[0, :]  
    cy = output_data[1, :]  
    w = output_data[2, :]  
    h = output_data[3, :]  
    obj_conf = output_data[4, :]  
    cls_conf = output_data[5:5+num_classes, :]  
    
    # Combine object and class confidence  
    conf_scores = obj_conf * cls_conf.T  
    max_scores = np.max(conf_scores, axis=1)  
    class_ids = np.argmax(conf_scores, axis=1)  
    
    # Convert to box coordinates (relative to input_size)  
    x0 = cx - w/2  
    y0 = cy - h/2  
    x1 = cx + w/2  
    y1 = cy + h/2  
    
    # Filter by confidence threshold  
    valid = max_scores > conf_threshold  
    boxes = np.stack([x0[valid], y0[valid], x1[valid], y1[valid]], axis=1)  
    scores = max_scores[valid]  
    class_ids = class_ids[valid]  

    # Convert to MNN tensors for NMS  
    boxes_tensor = expr.const(boxes.reshape(-1, 4), expr.NCHW, expr.float)  
    scores_tensor = expr.const(scores.reshape(-1), expr.NCHW, expr.float)  
    
    # Perform NMS  
    keep_indices = expr.NMS(boxes_tensor, scores_tensor, 100, nms_threshold, conf_threshold)  
    final_boxes = boxes[keep_indices.read()]  
    final_scores = scores[keep_indices.read()]  
    final_class_ids = class_ids[keep_indices.read()]  

    # Scale boxes to original image size  
    scale_h = original_size[0] / input_size  
    scale_w = original_size[1] / input_size  
    
    scaled_boxes = []  
    for box in final_boxes:  
        x0, y0, x1, y1 = box  
        scaled_boxes.append([  
            int(x0 * scale_w),  
            int(y0 * scale_h),  
            int(x1 * scale_w),  
            int(y1 * scale_h)  
        ])  
    
    return scaled_boxes, final_scores, final_class_ids  

def main():  
    # Configuration  
    MODEL_PATH = "yolo11n_320.mnn"  
    IMAGE_PATH = "ros2_camera_feed.jpg"  
    INPUT_SIZE = 320  
    NUM_CLASSES = 20  # Adjust based on your model  
    CONF_THRESHOLD = 0.5  
    NMS_THRESHOLD = 0.45  

    # Load model with runtime configuration  
    runtime_cfg = {  
        "backend": "CPU",  
        "precision": "normal",  
        "numThread": 4  
    }  
    rt = expr.nn.create_runtime_manager(runtime_cfg)  
    net = expr.nn.load_module_from_file(MODEL_PATH, [], [], runtime_manager=rt)  

    # Load and preprocess image  
    original_image = cv2.imread(IMAGE_PATH)  
    original_h, original_w = original_image.shape[:2]  
    
    # Resize with padding (maintain aspect ratio)  
    resized_image = cv2.resize(original_image, (INPUT_SIZE, INPUT_SIZE))  
    input_data = resized_image.astype(np.float32) / 255.0  
    input_data = input_data[..., ::-1]  # BGR to RGB  
    input_data = np.expand_dims(input_data.transpose(2, 0, 1), axis=0)  # HWC to NCHW  
    
    # Convert to MNN tensor  
    input_var = expr.const(input_data, expr.NCHW, expr.float)  
    input_var = expr.convert(input_var, expr.NC4HW4)  

    # Run inference  
    output_var = net.forward(input_var)  
    output_var = expr.convert(output_var, expr.NCHW)  
    output_data = np.array(output_var.read())  

    # Process output  
    boxes, scores, class_ids = process_yolo_output(  
        output_data,   
        INPUT_SIZE,   
        (original_h, original_w),  
        NUM_CLASSES,  
        CONF_THRESHOLD,  
        NMS_THRESHOLD  
    )  

    # Draw results  
    for box, score, cls_id in zip(boxes, scores, class_ids):  
        x0, y0, x1, y1 = box  
        label = f"Class {int(cls_id)}: {score:.2f}"  
        cv2.rectangle(original_image, (x0, y0), (x1, y1), (0, 255, 0), 2)  
        cv2.putText(original_image, label, (x0, y0-5),   
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  

    # Save and display  
    cv2.imwrite("detection_result.jpg", original_image)  
    cv2.imshow("YOLOv11n Detection", original_image)  
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  

if __name__ == "__main__":  
    main()