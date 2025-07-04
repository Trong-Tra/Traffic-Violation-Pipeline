# yolo_predict_udf_robust.py
import base64
import cv2
import numpy as np

# Roboflow violation zone
roboflow_zone = [
    (0, 522.642),
    (107.5, 578.182),
    (367.458, 446.087),
    (0, 332.872)
]

def scale_zone(zone_points, orig_w, orig_h, scaled_w=640, scaled_h=640):
    scale_x = orig_w / scaled_w
    scale_y = orig_h / scaled_h
    return [(int(x * scale_x), int(y * scale_y)) for (x, y) in zone_points]

def is_bottom_half_inside_zone(xyxy, zone, num_points=10):
    x1, y1, x2, y2 = xyxy
    count_inside = 0
    for i in range(num_points + 1):
        x = int(x1 + i * (x2 - x1) / num_points)
        y = int(y2)
        if cv2.pointPolygonTest(np.array(zone, dtype=np.int32), (x, y), False) >= 0:
            count_inside += 1
    return count_inside > (num_points / 2)
import logging
from pyspark.sql.types import StringType
from pyspark.sql.functions import pandas_udf, col
import pandas as pd
from ultralytics import YOLO
import os
import gc
import traceback
import time
import psutil
from threading import Lock
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable and circuit breaker
_model = None
_model_load_attempts = 0
_max_load_attempts = 3
_model_lock = Lock()
_circuit_breaker_failures = 0
_circuit_breaker_threshold = 5
_circuit_breaker_reset_time = 300  # 5 minutes
_last_failure_time = 0

class CircuitBreakerException(Exception):
    """Custom exception for circuit breaker"""
    pass

def check_circuit_breaker():
    """Check if circuit breaker is open"""
    global _circuit_breaker_failures, _circuit_breaker_threshold, _last_failure_time, _circuit_breaker_reset_time
    
    current_time = time.time()
    
    # Reset circuit breaker after timeout
    if _circuit_breaker_failures >= _circuit_breaker_threshold:
        if current_time - _last_failure_time > _circuit_breaker_reset_time:
            logger.info("🔄 Circuit breaker reset - attempting to recover")
            _circuit_breaker_failures = 0
        else:
            remaining_time = _circuit_breaker_reset_time - (current_time - _last_failure_time)
            raise CircuitBreakerException(f"Circuit breaker is OPEN. Reset in {remaining_time:.0f} seconds")
    
    return True

def record_failure():
    """Record a failure for circuit breaker"""
    global _circuit_breaker_failures, _last_failure_time
    _circuit_breaker_failures += 1
    _last_failure_time = time.time()
    logger.warning(f"⚠️ Circuit breaker failure count: {_circuit_breaker_failures}")

def get_model():
    """Thread-safe lazy load model with circuit breaker"""
    global _model, _model_load_attempts, _max_load_attempts, _model_lock
    
    with _model_lock:
        # Check circuit breaker first
        try:
            check_circuit_breaker()
        except CircuitBreakerException as e:
            logger.error(f"❌ {str(e)}")
            return None
        
        if _model is None and _model_load_attempts < _max_load_attempts:
            try:
                _model_load_attempts += 1
                # Use relative path from the script location  
                model_path = os.path.join(os.path.dirname(__file__), "..", "model", "best.pt")
                model_path = os.path.abspath(model_path)
                
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                
                # Check available memory before loading
                memory = psutil.virtual_memory()
                if memory.percent > 85:
                    raise MemoryError(f"Insufficient memory to load model: {memory.percent}% used")
                
                logger.info(f"🔄 Loading YOLO model (attempt {_model_load_attempts})...")
                _model = YOLO(model_path)
                _model.overrides['verbose'] = False
                _model.overrides['device'] = 'cpu'  # Force CPU to reduce memory usage
                logger.info("✅ YOLO model loaded successfully")
                
                # Reset circuit breaker on successful load
                global _circuit_breaker_failures
                _circuit_breaker_failures = 0
                
            except Exception as e:
                logger.error(f"❌ Failed to load YOLO model (attempt {_model_load_attempts}): {str(e)}")
                record_failure()
                if _model_load_attempts >= _max_load_attempts:
                    logger.error("❌ Max model load attempts reached")
                    raise e
                _model = None
        
        return _model

def validate_base64_image(b64_str):
    """Enhanced validation of base64 string and basic image properties"""
    try:
        if not b64_str or pd.isna(b64_str):
            return False, "Empty or null input"
        
        if len(b64_str) < 100:
            return False, "String too short to be valid image"
        
        if len(b64_str) > 10000000:  # ~7.5MB limit
            return False, "String too large (>10MB)"
        
        # Check if it looks like base64
        if not b64_str.replace('+', '').replace('/', '').replace('=', '').replace('\n', '').replace('\r', '').isalnum():
            return False, "Invalid base64 characters"
        
        # Try to decode
        try:
            image_data = base64.b64decode(b64_str)
        except Exception as decode_error:
            return False, f"Base64 decode error: {str(decode_error)}"
        
        if len(image_data) < 1000:
            return False, "Decoded data too small"
        
        if len(image_data) > 50000000:  # 50MB limit
            return False, "Decoded data too large (>50MB)"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def safe_image_processing(b64_str, model, max_processing_time=30):
    """Safely process image with timeout and resource limits"""
    start_time = time.time()

    # Define Roboflow stop zone
    roboflow_zone = [
        (0, 522.642),
        (107.5, 578.182),
        (367.458, 446.087),
        (0, 332.872)
    ]

    def scale_zone(zone_points, orig_w, orig_h, scaled_w=640, scaled_h=640):
        scale_x = orig_w / scaled_w
        scale_y = orig_h / scaled_h
        return [(int(x * scale_x), int(y * scale_y)) for (x, y) in zone_points]

    def is_bottom_half_inside_zone(xyxy, zone, num_points=10):
        x1, y1, x2, y2 = xyxy
        count_inside = 0
        for i in range(num_points + 1):
            x = int(x1 + i * (x2 - x1) / num_points)
            y = int(y2)
            if cv2.pointPolygonTest(np.array(zone, dtype=np.int32), (x, y), False) >= 0:
                count_inside += 1
        return count_inside > (num_points / 2)

    try:
        is_valid, validation_msg = validate_base64_image(b64_str)
        if not is_valid:
            return "", f"Validation failed: {validation_msg}"

        image_data = base64.b64decode(b64_str)
        np_img = np.frombuffer(image_data, dtype=np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None or img.size == 0:
            return "", "Invalid image data after decode"

        h, w = img.shape[:2]
        if h < 32 or w < 32 or h > 2048 or w > 2048:
            return "", f"Unsupported image size: {w}x{h}"

        preds = model(img, verbose=False, conf=0.25, iou=0.45, max_det=50, device='cpu', half=False, augment=False)
        if not preds or len(preds) == 0 or len(preds[0].boxes) == 0:
            return "", "No objects detected"

        pred = preds[0]

        # === 🚦 Violation Detection Logic ===
        violation_zone_scaled = scale_zone(roboflow_zone, w, h)
        is_red_light = any(model.names[int(box.cls[0])] == "red_light" for box in pred.boxes)

        violation_detected = False
        for box in pred.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label not in ["car", "motorbike"]:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if is_bottom_half_inside_zone((x1, y1, x2, y2), violation_zone_scaled) and is_red_light:
                violation_detected = True
                break

        # === 📸 Annotate Image ===
        annotated_img = pred.plot(conf=True, line_width=1, font_size=0.5, pil=False)
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 60, cv2.IMWRITE_JPEG_OPTIMIZE, 1, cv2.IMWRITE_JPEG_PROGRESSIVE, 1]
        success, buffer = cv2.imencode(".jpg", annotated_img, encode_params)

        if not success or len(buffer) == 0:
            return "", "Failed to encode annotated image"

        b64_result = base64.b64encode(buffer).decode("utf-8")

        # === ✏️ Append violation tag if needed ===
        if violation_detected:
            b64_result = "[VIOLATION]|" + b64_result

        return b64_result, "Success"

    except Exception as e:
        return "", f"Processing error: {str(e)}"

    finally:
        try:
            del img, np_img, image_data
            if 'pred' in locals(): del pred
            if 'preds' in locals(): del preds
            if 'annotated_img' in locals(): del annotated_img
            gc.collect()
        except:
            pass

@pandas_udf(returnType=StringType())
def yolo_udf(base64_series: pd.Series) -> pd.Series:
    """
    Ultra-robust Pandas UDF for YOLO prediction with circuit breaker and comprehensive error handling
    """
    try:
        # Check circuit breaker
        try:
            check_circuit_breaker()
        except CircuitBreakerException as e:
            logger.error(f"❌ {str(e)}")
            return pd.Series(["CIRCUIT_BREAKER_OPEN"] * len(base64_series))
        
        # Check memory before processing
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            logger.error(f"❌ Memory usage too high: {memory.percent}%")
            record_failure()
            return pd.Series(["INSUFFICIENT_MEMORY"] * len(base64_series))
        
        model = get_model()
        if model is None:
            logger.error("❌ Model not available")
            record_failure()
            return pd.Series(["MODEL_UNAVAILABLE"] * len(base64_series))
        
        results = []
        batch_size = len(base64_series)
        logger.info(f"🔄 Processing batch of {batch_size} images")
        
        successful_predictions = 0
        start_time = time.time()
        max_batch_time = 300  # 5 minutes max per batch
        
        for idx, b64_str in enumerate(base64_series):
            try:
                # Check batch timeout
                if time.time() - start_time > max_batch_time:
                    logger.error(f"❌ Batch timeout after {idx} images")
                    # Fill remaining with timeout message
                    remaining = batch_size - idx
                    results.extend(["BATCH_TIMEOUT"] * remaining)
                    break
                
                # Check memory periodically
                if idx % 5 == 0:  # Check every 5 images
                    memory = psutil.virtual_memory()
                    if memory.percent > 95:
                        logger.error(f"❌ Critical memory usage during batch: {memory.percent}%")
                        remaining = batch_size - idx
                        results.extend(["MEMORY_CRITICAL"] * remaining)
                        record_failure()
                        break
                
                # Process single image
                result, error_msg = safe_image_processing(b64_str, model)
                
                if result:
                    results.append(result)
                    successful_predictions += 1
                else:
                    results.append("")
                    logger.debug(f"Failed to process image {idx}: {error_msg}")
                
            except Exception as e:
                logger.error(f"Error processing image at index {idx}: {str(e)}")
                results.append("")
                
                # If we get too many errors in a batch, stop processing
                if len([r for r in results if r.startswith("ERROR")]) > batch_size * 0.8:
                    logger.error("❌ Too many errors in batch, stopping")
                    remaining = batch_size - len(results)
                    results.extend(["BATCH_ERROR_LIMIT"] * remaining)
                    record_failure()
                    break
        
        # Force garbage collection after batch
        gc.collect()
        
        processing_time = time.time() - start_time
        success_rate = successful_predictions / batch_size if batch_size > 0 else 0
        
        logger.info(f"✅ Batch completed: {successful_predictions}/{batch_size} successful ({success_rate:.1%}) in {processing_time:.1f}s")
        
        # Record failure if success rate is too low
        if success_rate < 0.1:  # Less than 10% success
            logger.warning(f"⚠️ Low success rate: {success_rate:.1%}")
            record_failure()
        
        return pd.Series(results)
        
    except Exception as e:
        logger.error(f"Critical error in yolo_udf: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        record_failure()
        
        # Return error indicator for all items
        error_message = "UDF_CRITICAL_ERROR"
        return pd.Series([error_message] * len(base64_series))
        
    finally:
        # Always attempt cleanup
        try:
            gc.collect()
        except:
            pass