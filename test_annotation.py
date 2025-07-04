#!/usr/bin/env python3
"""
Test script to generate annotated images from video frames
"""
import cv2
import base64
import os
import sys
from ultralytics import YOLO

def test_yolo_annotation(video_path, output_dir="test_output", target_frame=100):
    """Test YOLO annotation on a specific video frame"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load YOLO model
    model_path = os.path.join("model", "best.pt")
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print("🔄 Loading YOLO model...")
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 Video has {total_frames} frames")
    print(f"🎯 Processing frame {target_frame}...")
    
    if target_frame >= total_frames:
        print(f"❌ Frame {target_frame} exceeds video length ({total_frames} frames)")
        cap.release()
        return
    
    frame_count = 0
    
    # Define violation zone (same as in your UDF)
    roboflow_zone = [
        (0, 522.642),
        (107.5, 578.182),
        (367.458, 446.087),
        (0, 332.872)
    ]
    
    # Skip to target frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    
    ret, frame = cap.read()
    if not ret:
        print(f"❌ Cannot read frame {target_frame}")
        cap.release()
        return
        
    print(f"🔄 Processing frame {target_frame}...")
    
    # Resize frame if needed
    height, width = frame.shape[:2]
    if width > 640:
        scale = 640 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height))
        height, width = frame.shape[:2]
    
    # Run YOLO prediction
    results = model(frame, verbose=False, conf=0.25, iou=0.45, max_det=50)
    
    if results and len(results) > 0:
        # Get annotated image
        annotated_frame = results[0].plot(conf=True, line_width=2, font_size=1.0)
        
        # Draw violation zone
        zone_scaled = [(int(x * width / 640), int(y * height / 640)) for x, y in roboflow_zone]
        cv2.polylines(annotated_frame, [np.array(zone_scaled, np.int32)], True, (0, 255, 255), 3)
        cv2.putText(annotated_frame, "VIOLATION ZONE", (zone_scaled[0][0], zone_scaled[0][1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Check for violations
        is_red_light = any(model.names[int(box.cls[0])] == "red_light" for box in results[0].boxes)
        violation_detected = False
        
        # Print detection summary
        print(f"🔍 Detections found:")
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            confidence = float(box.conf[0])
            print(f"  - {label}: {confidence:.2f}")
        
        print(f"🚦 Red light detected: {'YES' if is_red_light else 'NO'}")
        
        # Add status text to image
        status_text = f"Red Light: {'ON' if is_red_light else 'OFF'}"
        cv2.putText(annotated_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) if is_red_light else (0, 255, 0), 2)
        
        # === 🚦 Complete Violation Detection Logic ===
        def is_bottom_half_inside_zone(xyxy, zone, num_points=10):
            """Check if bottom half of bounding box is inside violation zone"""
            x1, y1, x2, y2 = xyxy
            count_inside = 0
            for i in range(num_points + 1):
                x = int(x1 + i * (x2 - x1) / num_points)
                y = int(y2)  # Bottom edge
                if cv2.pointPolygonTest(np.array(zone, dtype=np.int32), (x, y), False) >= 0:
                    count_inside += 1
            return count_inside > (num_points / 2)
        
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            confidence = float(box.conf[0])
            
            if label in ["car", "motorbike"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Check if vehicle's bottom half is in violation zone
                if is_bottom_half_inside_zone((x1, y1, x2, y2), zone_scaled) and is_red_light:
                    violation_detected = True
                    # Draw violation indicator
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)  # Red box
                    cv2.putText(annotated_frame, "VIOLATION!", (x1, y1-40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.putText(annotated_frame, f"{label} {confidence:.2f}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    print(f"🚨 VIOLATION: {label} detected in zone with red light!")
                else:
                    # Draw normal vehicle box in green if no violation
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"{label} {confidence:.2f}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    print(f"✅ Vehicle: {label} detected (no violation)")
        
        # Add violation summary to image
        if violation_detected:
            cv2.putText(annotated_frame, "TRAFFIC VIOLATION DETECTED", (10, height-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            cv2.putText(annotated_frame, "NO VIOLATIONS", (10, height-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Save annotated image with descriptive filename
        violation_status = "VIOLATION" if violation_detected else "NORMAL"
        red_light_status = "RED" if is_red_light else "GREEN"
        filename = f"frame_{target_frame}_{violation_status}_{red_light_status}.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, annotated_frame)
        
        print(f"\n🎯 SUMMARY for Frame {target_frame}:")
        print(f"   🚦 Red Light: {'ON' if is_red_light else 'OFF'}")
        print(f"   🚨 Violation: {'DETECTED' if violation_detected else 'NONE'}")
        print(f"   💾 Saved: {output_path}")
        
        return violation_detected
        
        # Also save original frame for comparison
        original_filename = f"frame_{target_frame}_original.jpg"
        original_path = os.path.join(output_dir, original_filename)
        cv2.imwrite(original_path, frame)
        print(f"✅ Saved original: {original_path}")
        
    else:
        print(f"⚠️ No detections in frame {target_frame}")
    
    cap.release()
    print(f"\n✅ Annotation complete! Check {output_dir}/ for images")
    print(f"🔍 To view images: nautilus {output_dir} (or your file manager)")

if __name__ == "__main__":
    import numpy as np
    
    video_path = "data/video.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        print("💡 Make sure you're running from the project root directory")
        sys.exit(1)
    
    test_yolo_annotation(video_path, target_frame=100)
