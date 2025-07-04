# diagnostic.py
import socket
import logging
import os
import psutil
import time
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError, NoBrokersAvailable
import subprocess
import platform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_port_availability(host, port):
    """Check if a port is available/reachable"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        logger.error(f"Error checking port {host}:{port} - {e}")
        return False

def check_kafka_service():
    """Check if Kafka service is running"""
    try:
        # Check if Kafka process is running
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'kafka' in proc.info['name'].lower() or \
                   any('kafka' in cmd.lower() for cmd in proc.info['cmdline']):
                    logger.info(f"✅ Found Kafka process: PID {proc.info['pid']}, CMD: {' '.join(proc.info['cmdline'][:3])}")
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False
    except Exception as e:
        logger.error(f"Error checking Kafka service: {e}")
        return False

def check_zookeeper_service():
    """Check if Zookeeper service is running"""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'zookeeper' in proc.info['name'].lower() or \
                   any('zookeeper' in cmd.lower() for cmd in proc.info['cmdline']):
                    logger.info(f"✅ Found Zookeeper process: PID {proc.info['pid']}")
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False
    except Exception as e:
        logger.error(f"Error checking Zookeeper service: {e}")
        return False

def test_kafka_producer():
    """Test Kafka producer connection"""
    try:
        producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda x: x.encode('utf-8'),
            request_timeout_ms=30000,
            retries=3
        )
        
        # Send a test message
        future = producer.send('test_topic', 'test_message')
        record_metadata = future.get(timeout=10)
        producer.close()
        
        logger.info(f"✅ Kafka producer test successful - Topic: {record_metadata.topic}, Partition: {record_metadata.partition}")
        return True
    except Exception as e:
        logger.error(f"❌ Kafka producer test failed: {e}")
        return False

def test_kafka_consumer():
    """Test Kafka consumer connection"""
    try:
        consumer = KafkaConsumer(
            bootstrap_servers=['localhost:9092'],
            consumer_timeout_ms=10000,
            auto_offset_reset='latest'
        )
        
        # List topics
        topics = consumer.list_consumer_groups()
        consumer.close()
        
        logger.info("✅ Kafka consumer test successful")
        return True
    except Exception as e:
        logger.error(f"❌ Kafka consumer test failed: {e}")
        return False

def check_system_resources():
    """Check system resources"""
    try:
        # Memory usage
        memory = psutil.virtual_memory()
        logger.info(f"💾 Memory: {memory.percent}% used ({memory.available // (1024**3)}GB available)")
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        logger.info(f"🖥️  CPU: {cpu_percent}% used")
        
        # Disk usage
        disk = psutil.disk_usage('/')
        logger.info(f"💽 Disk: {disk.percent}% used ({disk.free // (1024**3)}GB free)")
        
        # Network connections
        connections = psutil.net_connections()
        kafka_connections = [c for c in connections if c.laddr.port == 9092 or c.raddr and c.raddr.port == 9092]
        logger.info(f"🔗 Active Kafka connections: {len(kafka_connections)}")
        
        return True
    except Exception as e:
        logger.error(f"Error checking system resources: {e}")
        return False

def check_java_version():
    """Check Java version"""
    try:
        result = subprocess.run(['java', '-version'], capture_output=True, text=True, timeout=10)
        java_version = result.stderr.split('\n')[0] if result.stderr else "Unknown"
        logger.info(f"☕ Java version: {java_version}")
        return True
    except Exception as e:
        logger.error(f"❌ Java not found or error: {e}")
        return False

def check_firewall_and_antivirus():
    """Check for potential firewall/antivirus issues"""
    try:
        if platform.system() == "Windows":
            # Check Windows Firewall
            try:
                result = subprocess.run(['netsh', 'advfirewall', 'show', 'allprofiles', 'state'], 
                                      capture_output=True, text=True, timeout=10)
                if 'ON' in result.stdout:
                    logger.warning("⚠️  Windows Firewall is ON - may block Kafka connections")
                else:
                    logger.info("✅ Windows Firewall appears to be OFF")
            except:
                logger.warning("⚠️  Could not check Windows Firewall status")
        
        # Check for Norton (based on your error log mentioning "Norton executor driver")
        for proc in psutil.process_iter(['name']):
            try:
                if 'norton' in proc.info['name'].lower():
                    logger.warning("⚠️  Norton antivirus detected - may interfere with network connections")
                    break
            except:
                continue
        
        return True
    except Exception as e:
        logger.error(f"Error checking firewall/antivirus: {e}")
        return False

def run_diagnostics():
    """Run all diagnostic checks"""
    logger.info("🔍 Starting Kafka and System Diagnostics")
    logger.info("=" * 50)
    
    # System checks
    logger.info("1. Checking system resources...")
    check_system_resources()
    
    logger.info("\n2. Checking Java installation...")
    check_java_version()
    
    logger.info("\n3. Checking firewall and antivirus...")
    check_firewall_and_antivirus()
    
    # Network checks
    logger.info("\n4. Checking network connectivity...")
    if check_port_availability('localhost', 9092):
        logger.info("✅ Port 9092 (Kafka) is reachable")
    else:
        logger.error("❌ Port 9092 (Kafka) is NOT reachable")
    
    if check_port_availability('localhost', 2181):
        logger.info("✅ Port 2181 (Zookeeper) is reachable")
    else:
        logger.error("❌ Port 2181 (Zookeeper) is NOT reachable")
    
    # Service checks
    logger.info("\n5. Checking Zookeeper service...")
    if check_zookeeper_service():
        logger.info("✅ Zookeeper service is running")
    else:
        logger.error("❌ Zookeeper service is NOT running")
    
    logger.info("\n6. Checking Kafka service...")
    if check_kafka_service():
        logger.info("✅ Kafka service is running")
    else:
        logger.error("❌ Kafka service is NOT running")
    
    # Kafka functionality checks
    logger.info("\n7. Testing Kafka producer...")
    test_kafka_producer()
    
    logger.info("\n8. Testing Kafka consumer...")
    test_kafka_consumer()
    
    logger.info("\n" + "=" * 50)
    logger.info("🏁 Diagnostics completed")
    
    # Recommendations
    logger.info("\n📋 RECOMMENDATIONS:")
    logger.info("1. Ensure Kafka and Zookeeper are running")
    logger.info("2. Check if Norton antivirus is blocking connections")
    logger.info("3. Verify firewall settings allow localhost:9092")
    logger.info("4. Try restarting Kafka services")
    logger.info("5. Check Kafka logs for detailed error messages")

if __name__ == "__main__":
    run_diagnostics()