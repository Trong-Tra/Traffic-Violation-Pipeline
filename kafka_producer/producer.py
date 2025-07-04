# kafka_producer/producer.py
import cv2
from kafka import KafkaProducer
import base64
import time
import logging
import json
from kafka.errors import KafkaError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_producer():
    """Create Kafka producer with error handling"""
    try:
        producer = KafkaProducer(
            bootstrap_servers='localhost:9092',
            value_serializer=lambda x: x.encode('utf-8') if isinstance(x, str) else x,
            retries=3,
            acks='all',
            batch_size=16384,
            linger_ms=10,
            buffer_memory=33554432
        )
        logger.info("✅ Kafka producer created successfully")
        return producer
    except Exception as e:
        logger.error(f"❌ Failed to create Kafka producer: {str(e)}")
        raise e

def send_video_frames(video_path, topic='traffic_frames', frame_rate=10):
    """
    Send video frames to Kafka topic
    
    Args:
        video_path: Path to video file
        topic: Kafka topic name
        frame_rate: Frames per second to send (default 2 fps)
    """
    producer = None
    cap = None
    
    try:
        # Create producer
        producer = create_producer()
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"📹 Video info: {total_frames} frames, {original_fps} FPS")
        logger.info(f"🚀 Starting to send frames at {frame_rate} FPS")
        
        frame_count = 0
        sent_count = 0
        sleep_time = 1.0 / frame_rate
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                logger.info("📹 End of video reached")
                break
            
            frame_count += 1
            
            try:
                # Resize frame to reduce payload size (optional)
                height, width = frame.shape[:2]
                if width > 640:  # Resize if too large
                    scale = 640 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Encode frame to JPEG with quality control
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                success, buffer = cv2.imencode('.jpg', frame, encode_param)
                
                if not success:
                    logger.warning(f"Failed to encode frame {frame_count}")
                    continue
                
                # Convert to base64
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send to Kafka
                future = producer.send(topic, frame_b64)
                
                # Optional: wait for send confirmation (blocking)
                # record_metadata = future.get(timeout=10)
                
                sent_count += 1
                if sent_count % 10 == 0:
                    logger.info(f"📤 Sent {sent_count} frames")
                
                # Control frame rate
                time.sleep(sleep_time)
                
            except KafkaError as e:
                logger.error(f"❌ Kafka error for frame {frame_count}: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"❌ Error processing frame {frame_count}: {str(e)}")
                continue
        
        logger.info(f"✅ Successfully sent {sent_count}/{frame_count} frames")
        
    except KeyboardInterrupt:
        logger.info("🛑 Received keyboard interrupt, stopping producer...")
    except Exception as e:
        logger.error(f"❌ Critical error in producer: {str(e)}")
        raise e
    finally:
        # Cleanup resources
        if producer is not None:
            try:
                producer.flush()  # Ensure all messages are sent
                producer.close()
                logger.info("✅ Kafka producer closed")
            except:
                pass
        
        if cap is not None:
            try:
                cap.release()
                logger.info("✅ Video capture released")
            except:
                pass

def main():
    import os
    video_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'video.mp4')
    video_path = os.path.abspath(video_path)
    
    # Check if video file exists
    if not os.path.exists(video_path):
        logger.error(f"❌ Video file not found: {video_path}")
        return
    
    # Start sending frames
    send_video_frames(video_path, frame_rate=10)

if __name__ == "__main__":
    main()