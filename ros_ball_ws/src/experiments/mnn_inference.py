import rclpy  
from rclpy.node import Node  
import MNN  
import MNN.cv as mnn_cv2  
import MNN.numpy as mnn_np  
from sensor_msgs.msg import Image  
from cv_bridge import CvBridge  
import cv2 as std_cv2  
import numpy as np  
import time  
import os  
import gc  
import threading  
import psutil  

# ========== CONFIGURATION SETTINGS ==========  
# We're keeping resolution at 320x320 as requested  
MODEL_PATH = "yolo12n_320.mnn"  
IMG_WIDTH = 320  
IMG_HEIGHT = 320  

# MNN engine settings - these control how the AI model runs  
PRECISION = "lowBF"          # Uses lower precision to run faster  
BACKEND = "CPU"              # We use CPU since Pi doesn't have good GPU support  
THREAD_COUNT = 4             # Number of CPU threads to use (Pi 5 has 4 cores)  
NMS_THRESHOLD = 0.45         # Controls how object detections are combined  
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence to count a detection   

# We're not skipping frames as requested (FRAME_SKIP = 0)  
ENABLE_LOGGING = True        # Shows what's happening  
PREALLOC_BUFFERS = True      # Creates memory once instead of over and over  
CACHE_COMPILED_MODEL = True  # Saves the prepared model for faster startup next time  
COMPILED_MODEL_PATH = "/tmp/yolo_compiled.mnn"  

# New optimizations  
DISABLE_AUTO_GC = True       # Turns off automatic garbage collection  
CONTROL_MANUAL_GC = True     # We'll run garbage collection on our schedule  
GC_INTERVAL = 100            # Run garbage collection every 100 frames  
LOW_MEMORY_MODE = True       # Extra careful with memory usage  
MEMORY_MONITOR = True        # Keep track of memory usage  

class MNNInferenceNode(Node):  
    def __init__(self):  
        super().__init__('mnn_inference_node')  
        
        # === MEMORY OPTIMIZATION ===  
        # Disable Python's automatic garbage collection to prevent random slowdowns  
        if DISABLE_AUTO_GC:  
            gc.disable()  # This prevents Python from deciding when to clean up memory  
            self.get_logger().info("Automatic garbage collection disabled for better performance")  
        
        # === MESSAGE HANDLING ===  
        # Create subscription with small queue (1) to prevent memory buildup  
        # This is like saying "only keep one image waiting to be processed"  
        self.subscription = self.create_subscription(  
            Image, '/ascamera/camera_publisher/rgb0/image', self.image_callback, 1  
        )  

        # CvBridge helps convert between ROS images and OpenCV images  
        self.bridge = CvBridge()  
        
        # === MODEL CONFIGURATION ===  
        # These settings make the AI model run efficiently on the Pi 5  
        config = {  
            "precision": PRECISION,      # Use lower precision for speed  
            "backend": BACKEND,          # Use CPU since Pi doesn't have good GPU  
            "numThread": THREAD_COUNT,   # Use all 4 CPU cores  
            "memoryMode": "High",        # Optimize for performance  
            "forwardType": "Fast",       # Prioritize speed over other factors  
        }  
        
        self.get_logger().info(f"Setting up for {IMG_WIDTH}x{IMG_HEIGHT} images using {THREAD_COUNT} CPU threads")  
        
        # === MODEL LOADING ===  
        # Load the model in a separate thread so the program starts quickly  
        self.model_ready = False  
        self.model_thread = threading.Thread(target=self.load_model, args=(config,), daemon=True)  
        self.model_thread.start()  

        # === STATISTICS TRACKING ===  
        # Keep track of how fast we're processing  
        self.start_time = time.time()  
        self.image_count = 0  
        self.total_process_time = 0  
        self.last_gc_frame = 0  
        
        # === MEMORY MANAGEMENT ===  
        # Create memory buffers once instead of over and over  
        if PREALLOC_BUFFERS:  
            # Pre-allocate memory for image processing to prevent slowdowns  
            self.preallocated_input = None  
            # These are like reusable containers for the data at each step  
            self.resize_buffer = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)  
            self.float_buffer = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)  
            self.chw_buffer = np.zeros((3, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)  
            self.get_logger().info("Pre-allocated memory buffers for faster processing")  
        
        # Initialize memory tracking  
        self.last_memory_check = time.time()  
        if MEMORY_MONITOR:  
            self.memory_thread = threading.Thread(target=self.monitor_memory, daemon=True)  
            self.memory_thread.start()  
    
    def monitor_memory(self):  
        """This function runs in the background to monitor memory usage"""  
        try:  
            while True:  
                # Only check memory every 10 seconds to avoid overhead  
                if time.time() - self.last_memory_check > 10:  
                    process = psutil.Process(os.getpid())  
                    memory_info = process.memory_info()  
                    memory_mb = memory_info.rss / 1024 / 1024  # Convert bytes to MB  
                    self.get_logger().info(f"Memory usage: {memory_mb:.1f} MB")  
                    self.last_memory_check = time.time()  
                    
                    # If we're using too much memory, force garbage collection  
                    if memory_mb > 200 and time.time() - self.last_gc_time > 5:  
                        self.get_logger().warn("High memory usage - cleaning up")  
                        gc.collect()  
                        self.last_gc_time = time.time()  
                time.sleep(5)  
        except:  
            pass  
        
    def load_model(self, config):  
        """Loads the AI model and prepares it for processing images"""  
        try:  
            # === PROCESSING PRIORITY ===  
            # Try to make this thread higher priority so it gets more CPU time  
            try:  
                process = psutil.Process(os.getpid())  
                process.nice(10)  # Higher priority (lower value)  
            except:  
                pass  
                
            # === COMPILED MODEL LOADING ===  
            # Try to load a pre-compiled model (faster startup)  
            if CACHE_COMPILED_MODEL and os.path.exists(COMPILED_MODEL_PATH):  
                try:  
                    self.get_logger().info(f"Loading faster pre-compiled model")  
                    self.runtime_manager = MNN.nn.create_runtime_manager((config,))  
                    self.net = MNN.nn.load_module_from_file(COMPILED_MODEL_PATH, [], [],   
                                                         runtime_manager=self.runtime_manager)  
                    self.get_logger().info("✓ Using pre-compiled model!")  
                except Exception as e:  
                    self.get_logger().warn(f"Couldn't load compiled model: {e}")  
                    os.remove(COMPILED_MODEL_PATH)  # Remove corrupted model  
                    
            # === REGULAR MODEL LOADING ===  
            # Load the original model if needed  
            if not hasattr(self, 'net'):  
                self.get_logger().info("Loading regular model (this takes longer)")  
                self.runtime_manager = MNN.nn.create_runtime_manager((config,))  
                self.net = MNN.nn.load_module_from_file(MODEL_PATH, [], [], runtime_manager=self.runtime_manager)  
                
                # Save a compiled version for next time  
                if CACHE_COMPILED_MODEL:  
                    try:  
                        self.get_logger().info("Creating faster model version for next time")  
                        self.net.save_to_file(COMPILED_MODEL_PATH)  
                        self.get_logger().info(f"✓ Saved compiled model")  
                    except Exception as e:  
                        self.get_logger().warn(f"Couldn't save compiled model: {e}")  
            
            # === PRE-ALLOCATION ===  
            # Create memory for reuse during processing  
            if PREALLOC_BUFFERS:  
                dummy_image = np.zeros((3, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)  
                dummy_tensor = MNN.expr.const(dummy_image, [3, IMG_HEIGHT, IMG_WIDTH], MNN.expr.NCHW)  
                dummy_tensor = MNN.expr.convert(dummy_tensor, MNN.expr.NC4HW4)  
                self.preallocated_input = MNN.expr.reshape(dummy_tensor, [1, 3, IMG_HEIGHT, IMG_WIDTH])  
            
            # === WARMUP ===  
            # Run the model a few times with dummy data to get it ready  
            dummy_image = np.zeros((3, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)  
            dummy_tensor = MNN.expr.const(dummy_image, [3, IMG_HEIGHT, IMG_WIDTH], MNN.expr.NCHW)  
            dummy_tensor = MNN.expr.convert(dummy_tensor, MNN.expr.NC4HW4)  
            dummy_input = MNN.expr.reshape(dummy_tensor, [1, 3, IMG_HEIGHT, IMG_WIDTH])  
            
            self.get_logger().info("Warming up the model...")  
            for i in range(5):  
                self.net.forward(dummy_input)  
            
            # Final cleanup  
            gc.collect()  
            
            self.model_ready = True  
            self.last_gc_time = time.time()  
            self.get_logger().info("✓ Model loaded and ready! Now processing images...")  
            
        except Exception as e:  
            self.get_logger().error(f"⚠️ Model loading failed: {str(e)}")  
            import traceback  
            self.get_logger().error(traceback.format_exc())  

    def image_callback(self, msg):  
        """This function processes each image from the camera"""  
        # Skip if model not ready yet  
        if not self.model_ready:  
            return  
            
        # Start timing how long processing takes  
        inference_start = time.time()  
        self.image_count += 1  
        
        try:  
            # === IMAGE PREPARATION ===  
            # Convert ROS image to OpenCV format (like translating between languages)  
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')  
            
            # === MEMORY-EFFICIENT PROCESSING ===  
            if PREALLOC_BUFFERS:  
                # Resize directly into our pre-allocated buffer (no new memory needed)  
                if cv_image.shape[0] != IMG_HEIGHT or cv_image.shape[1] != IMG_WIDTH:  
                    std_cv2.resize(cv_image, (IMG_WIDTH, IMG_HEIGHT), dst=self.resize_buffer)  
                    image_data = self.resize_buffer  
                else:  
                    # Copy to ensure consistent memory layout  
                    np.copyto(self.resize_buffer, cv_image)  
                    image_data = self.resize_buffer  
                
                # Convert BGR to RGB (flip the color channels)  
                # This is like changing from blue-green-red to red-green-blue order  
                image_data = image_data[..., ::-1]  
                
                # Normalize to [0,1] range (convert 0-255 values to 0-1 for the AI model)  
                np.multiply(image_data, (1.0/255.0), out=self.float_buffer)  
                
                # Transpose HWC to CHW format (channels first format for AI processing)  
                # This is like reorganizing data from (height,width,colors) to (colors,height,width)  
                for c in range(3):  
                    self.chw_buffer[c] = self.float_buffer[:,:,c]  
                
                # Create MNN tensor from numpy array (reusing existing memory if possible)  
                # This converts our data to the format MNN expects  
                if hasattr(self, 'input_tensor'):  
                    # Try to update existing tensor data (reuse memory)  
                    try:  
                        self.input_tensor._reset_data(self.chw_buffer)  
                        input_tensor = self.input_tensor  
                    except:  
                        # Fallback if reset fails  
                        input_tensor = MNN.expr.const(self.chw_buffer, [3, IMG_HEIGHT, IMG_WIDTH], MNN.expr.NCHW)  
                else:  
                    # First time, create new tensor  
                    input_tensor = MNN.expr.const(self.chw_buffer, [3, IMG_HEIGHT, IMG_WIDTH], MNN.expr.NCHW)  
                    self.input_tensor = input_tensor  
            else:  
                # Original implementation without memory reuse  
                # This creates new memory for each step (slower)  
                if cv_image.shape[0] != IMG_HEIGHT or cv_image.shape[1] != IMG_WIDTH:  
                    cv_image = std_cv2.resize(cv_image, (IMG_WIDTH, IMG_HEIGHT))  
                    
                # Convert BGR to RGB  
                rgb_image = cv_image[..., ::-1]  
                
                # Convert to float and normalize [0,1]  
                rgb_image = rgb_image.astype(np.float32) * (1.0/255.0)  
                
                # Transpose from HWC to CHW format  
                chw_image = np.transpose(rgb_image, (2, 0, 1))  
                
                # Create MNN tensor from numpy array  
                input_tensor = MNN.expr.const(chw_image, [3, IMG_HEIGHT, IMG_WIDTH], MNN.expr.NCHW)  
            
            # === MODEL INPUT PREPARATION ===  
            # Convert to NC4HW4 format (special format MNN uses internally)  
            input_tensor = MNN.expr.convert(input_tensor, MNN.expr.NC4HW4)  
            input_tensor = MNN.expr.reshape(input_tensor, [1, 3, IMG_HEIGHT, IMG_WIDTH])  
                
            # === INFERENCE (AI PROCESSING) ===  
            # This is where the magic happens - the AI processes the image  
            output_var = self.net.forward(input_tensor)  
            
            # === RESULTS PROCESSING ===  
            # Convert output to standard format and remove extra dimensions  
            output_var = MNN.expr.convert(output_var, MNN.expr.NCHW).squeeze()  
            
            # Extract detection data (like reading the AI's answers)  
            cx, cy = output_var[0], output_var[1]  # Center X and Y coordinates  
            w, h = output_var[2], output_var[3]    # Width and height  
            probs = output_var[4:]                 # Probabilities for each class  

            # Convert center-based boxes to corner-based boxes  
            # (from center,width,height to top-left and bottom-right corners)  
            x0 = cx - w * 0.5  # Left edge  
            y0 = cy - h * 0.5  # Top edge  
            x1 = cx + w * 0.5  # Right edge  
            y1 = cy + h * 0.5  # Bottom edge  

            # Stack coordinates to create boxes  
            boxes = mnn_np.stack([x0, y0, x1, y1], axis=1)  

            # Get scores (confidence) and class IDs (what objects were detected)  
            scores = mnn_np.max(probs, axis=0)          # Highest probability for each detection  
            class_ids = mnn_np.argmax(probs, axis=0)    # Class with highest probability  

            # Filter by confidence threshold (only keep confident detections)  
            valid_indices = mnn_np.where(scores > CONFIDENCE_THRESHOLD)[0]  
            
            # Process detections if any found  
            if len(valid_indices) > 0:  
                # Get only the boxes, scores, and classes that passed our confidence filter  
                filtered_boxes = boxes[valid_indices]  
                filtered_scores = scores[valid_indices]  
                filtered_class_ids = class_ids[valid_indices]  
                
                # Apply Non-Maximum Suppression (NMS) to remove duplicate detections  
                # This is like finding the best box when multiple boxes detect the same object  
                result_ids = MNN.expr.nms(filtered_boxes, filtered_scores, 100, NMS_THRESHOLD)  
                
                # Log the detection results  
                if ENABLE_LOGGING:  
                    for i in range(len(result_ids)):  
                        box = filtered_boxes[result_ids[i]]  
                        x0_val, y0_val, x1_val, y1_val = box.read_as_tuple()  
                        cls_id = filtered_class_ids[result_ids[i]]  
                        score = filtered_scores[result_ids[i]]  
        
                        self.get_logger().info(f"Detection: Class={cls_id}, Score={score:.2f}, "  
                              f"Box=({x0_val:.1f}, {y0_val:.1f}, {x1_val:.1f}, {y1_val:.1f})")  
            
            # === PERFORMANCE TRACKING ===  
            # Calculate how long this frame took to process  
            total_time = (time.time() - inference_start) * 1000  # Convert to milliseconds  
            self.total_process_time += total_time  
            
            # Calculate frames per second (FPS)  
            elapsed_time = time.time() - self.start_time  
            fps = self.image_count / elapsed_time if elapsed_time > 0 else 0  

            # Display performance info  
            if ENABLE_LOGGING:  
                self.get_logger().info(f"Inference={total_time:.1f}ms, FPS={fps:.1f}")  

            # === MEMORY MANAGEMENT ===  
            # Manually run garbage collection periodically  
            if CONTROL_MANUAL_GC and self.image_count - self.last_gc_frame >= GC_INTERVAL:  
                if ENABLE_LOGGING:  
                    self.get_logger().info("Running scheduled memory cleanup")  
                gc.collect()  
                self.last_gc_frame = self.image_count  
                self.last_gc_time = time.time()  

        except Exception as e:  
            self.get_logger().error(f"Error: {str(e)}")  
            import traceback  
            self.get_logger().error(traceback.format_exc())  
                
    def __del__(self):  
        """This runs when the program is shutting down - cleanup!"""  
        # Cleanup  
        try:  
            # Release model resources  
            if hasattr(self, 'net'):  
                del self.net  
            if hasattr(self, 'runtime_manager'):  
                del self.runtime_manager  
            # Force garbage collection  
            gc.collect()  
        except:  
            pass  

def main(args=None):  
    # === SYSTEM OPTIMIZATION ===  
    # Try to set CPU to performance mode (may not work in container)  
    try:  
        os.system("echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null")  
        print("Set CPU to performance mode (max speed)")  
    except:  
        print("Could not set CPU to performance mode (expected in container)")  
    
    # Try to lock memory to prevent swapping (may not work in container)  
    try:  
        import resource  
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (-1, -1))  
        print("Locked memory to prevent swapping")  
    except:  
        print("Could not lock memory (expected in container)")  
    
    # Try to set process priority (may not work in container)  
    try:  
        os.nice(-10)  # Higher priority  
        print("Set high process priority")  
    except:  
        print("Could not set process priority (expected in container)")  
        
    # === ROS INITIALIZATION ===  
    # Start ROS system  
    rclpy.init(args=args)  
    node = MNNInferenceNode()  
    
    print("🚀 Optimized MNNInferenceNode is running in container. Press Ctrl+C to exit.")  
    
    try:  
        # Main processing loop - this runs until you press Ctrl+C  
        rclpy.spin(node)  
    except KeyboardInterrupt:  
        print("Node stopped by keyboard interrupt")  
    except Exception as e:  
        print(f"Error during node execution: {str(e)}")  
        import traceback  
        print(traceback.format_exc())  
    finally:  
        # Cleanup when done  
        node.destroy_node()  
        rclpy.shutdown()  
        
        # Final memory cleanup  
        gc.collect()  
        print("✓ Node shut down successfully")  

if __name__ == '__main__':  
    main()