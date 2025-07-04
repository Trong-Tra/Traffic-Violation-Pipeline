# Traffic Violation Detection Pipeline

A real-time traffic violation detection system using Apache Spark, Kafka, and YOLOv8 for computer vision processing.

## 🎯 Project Overview

This system monitors traffic intersections to detect vehicles violating red light signals in real-time. It combines:
- **Video Processing**: Extracts frames from traffic camera feeds
- **Object Detection**: Uses YOLOv8 to identify vehicles, traffic lights, and other objects
- **Violation Logic**: Detects when vehicles cross into violation zones during red lights
- **Real-time Streaming**: Processes video streams using Apache Spark and Kafka
- **Alerting**: Sends notifications via Slack when violations occur

## 🏗️ Architecture

```
Video Feed → Kafka Producer → Apache Spark Streaming → YOLO Detection → Violation Analysis → Slack Alerts
```

### Components:
1. **Kafka Producer** (`kafka_producer/`): Extracts and streams video frames
2. **Spark Streaming** (`spark_streaming/`): Real-time frame processing
3. **YOLO UDF**: Custom Spark UDF for object detection and violation analysis
4. **Notification System**: Slack integration for violation alerts

## ⚙️ Features

- **Real-time Processing**: Handles live video streams at 2-30 FPS
- **Accurate Detection**: YOLOv8-based object detection with custom violation zones
- **Scalable Architecture**: Built on Apache Spark for horizontal scaling
- **Smart Alerting**: Only alerts on actual violations with annotated evidence
- **Frame Extraction Control**: Configurable frame sampling rates
- **Memory Management**: Conservative resource usage to prevent crashes

## 🛠️ Tech Stack

<div align="center">  
    <img src="https://skillicons.dev/icons?i=python,opencv,kafka,spark" alt="Tech stack icons"/> <br>
    <img src="https://skillicons.dev/icons?i=docker,git,github,vscode" alt="Tech stack icons"/> <br>
</div>

### Technologies Used

- **Python 3.10+** - Core programming language
- **Apache Spark 3.5.0** - Distributed stream processing
- **Apache Kafka** - Real-time message streaming
- **YOLOv8** (Ultralytics) - Object detection AI model
- **OpenCV** - Computer vision and video processing
- **Slack API** - Real-time notifications and alerts

## 📊 Model Performance

### Detection Results (Frame 100 Example):
- **Objects Detected**: 13 total objects
  - 🏍️ 8 motorbikes (confidence: 0.28-0.89)
  - 🚗 1 car (confidence: 0.85)
  - 🚶 3 persons (confidence: 0.26-0.72)
  - 🚦 2 red lights (confidence: 0.39-0.60)
  - ⬜ 1 stop line (confidence: 0.90)

### Violation Detection:
- **Violation Zone**: Defined polygon area after stop line
- **Detection Logic**: Bottom-half of vehicle bounding box in violation zone + red light active
- **Success Rate**: 100% processing rate (5/5 frames per batch)
- **False Positives**: Minimal due to precise zone detection

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Start Kafka server
./start-kafka.sh

# Ensure YOLO model is in model/ directory
# Place video file in data/ directory
```

### Running the Pipeline

1. **Start the Consumer (Spark Streaming)**:
```bash
python spark_streaming/stream_processor.py
```

2. **Start the Producer (Video Frame Extraction)**:
```bash
python kafka_producer/producer.py
```

### Configuration Options

**Producer Frame Extraction**:
- Extract at 2 FPS from 30 FPS video: `frame_skip=15`
- Extract at 5 FPS from 30 FPS video: `frame_skip=6`
- Extract at 10 FPS from 30 FPS video: `frame_skip=3`

**Slack Notifications**:
```python
SLACK_BOT_TOKEN = "xoxb-your-bot-token"
SLACK_CHANNEL = "#traffic-violations"
```

## 📈 Performance Metrics

- **Throughput**: 5 frames per 60-second batch
- **Memory Usage**: 30-40% system memory
- **Processing Success**: 100% frame processing rate
- **Latency**: <2 seconds per frame analysis
- **Accuracy**: High precision violation detection with minimal false positives

## 🔧 System Requirements

- **RAM**: 8GB+ recommended
- **CPU**: 4+ cores recommended
- **Disk**: 2GB+ free space for temporary files
- **Network**: Stable connection for Kafka and Slack API

## 📁 Project Structure

```
traffic_violation_pipeline/
├── kafka_producer/
│   └── producer.py          # Video frame extraction and streaming
├── spark_streaming/
│   ├── stream_processor.py  # Main Spark streaming application
│   ├── yolo_predict_udf.py  # YOLO detection UDF
│   └── diagnostic.py        # System diagnostics
├── requirements.txt         # Python dependencies
├── test_annotation.py       # Single frame testing tool
└── README.md               # This file
```

## 🎯 Testing Single Frames

Test the violation detection on specific frames:

```bash
python test_annotation.py
```

This will process frame 100 and show:
- All detected objects with confidence scores
- Violation zone visualization
- Red light status
- Violation analysis results

## 🔍 Example Output

```
🔍 Detections found:
  - stop_line: 0.90
  - motorbike: 0.89
  - car: 0.85
  - red_light: 0.60

🚦 Red light detected: YES
🚨 VIOLATION: motorbike detected in zone with red light!

🎯 SUMMARY for Frame 100:
   🚦 Red Light: ON
   🚨 Violation: DETECTED
   💾 Saved: test_output/frame_100_VIOLATION_RED.jpg
```

## 🤝 Contributing

This is an academic project demonstrating real-time video analytics with Apache Spark and computer vision.

## 📝 License

Educational use only.

---

**Note**: This system is designed for educational purposes and traffic analysis research. Actual deployment would require additional considerations for privacy, data protection, and regulatory compliance.
