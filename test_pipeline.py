#!/usr/bin/env python3
"""
Quick test to check Kafka topic messages and run producer if needed
"""
import subprocess
import time
import os

def check_kafka_messages():
    """Check if there are messages in the Kafka topic"""
    try:
        print("🔍 Checking Kafka topic for existing messages...")
        result = subprocess.run([
            '/home/tron/kafka_2.13-3.5.0/bin/kafka-run-class.sh',
            'kafka.tools.GetOffsetShell',
            '--broker-list', 'localhost:9092',
            '--topic', 'traffic_frames'
        ], capture_output=True, text=True, timeout=10)
        
        if result.stdout.strip():
            offsets = result.stdout.strip().split('\n')
            total_messages = sum(int(line.split(':')[-1]) for line in offsets if ':' in line)
            print(f"📊 Found {total_messages} messages in Kafka topic")
            return total_messages
        else:
            print("📭 No messages found in Kafka topic")
            return 0
            
    except Exception as e:
        print(f"❌ Error checking Kafka: {e}")
        return -1

def run_producer():
    """Run the producer to generate some frames"""
    print("🚀 Starting producer to generate frames...")
    try:
        subprocess.run(['python', 'kafka_producer/producer.py'], timeout=30)
        print("✅ Producer finished")
    except subprocess.TimeoutExpired:
        print("⏰ Producer stopped after 30 seconds")
    except Exception as e:
        print(f"❌ Producer error: {e}")

def main():
    print("🔧 Traffic Violation Pipeline - Quick Test")
    print("=" * 50)
    
    # Check current messages
    message_count = check_kafka_messages()
    
    # If no messages, run producer briefly
    if message_count <= 0:
        print("\n💡 No messages found. Running producer for 30 seconds...")
        run_producer()
        
        # Check again
        time.sleep(2)
        message_count = check_kafka_messages()
    
    if message_count > 0:
        print(f"\n✅ Ready to process {message_count} frames!")
        print("💡 You can now run the stream processor:")
        print("   python spark_streaming/stream_processor.py")
    else:
        print("\n⚠️  No messages available for processing")
        print("💡 Try running the producer manually:")
        print("   python kafka_producer/producer.py")

if __name__ == "__main__":
    main()
