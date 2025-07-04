# stream_processor_fixed.py
import requests
import os

# Clean up problematic environment variables
if "ARROW_PRE_0_15_IPC_FORMAT" in os.environ:
    del os.environ["ARROW_PRE_0_15_IPC_FORMAT"] 

# Slack configuration - OAuth token method (more reliable for file uploads)
# 
# Setup Instructions:
# 1. Go to https://api.slack.com/apps
# 2. Create a new app or use existing one
# 3. Go to "OAuth & Permissions" 
# 4. Add these scopes: chat:write, files:write, channels:read
# 5. Install app to workspace
# 6. Copy the "Bot User OAuth Token" (starts with xoxb-)
# 7. Set the token and channel below:
#
SLACK_BOT_TOKEN = ""  # Set your bot token here: "xoxb-your-bot-token"
SLACK_CHANNEL = "#traffic_violation_notification"  # or channel ID like "C1234567890"

def send_violation_to_slack(image_path, message="🚨 Violation detected!"):
    try:
        # Check if Slack is properly configured
        if not SLACK_BOT_TOKEN:
            print(f"[⚠️] Slack bot token not configured, skipping alert for: {image_path}")
            return
        
        # Use Slack Web API for file upload (much more reliable)
        import requests
        
        # First, send a text message
        text_payload = {
            "channel": SLACK_CHANNEL,
            "text": f"{message}\n📁 File: {os.path.basename(image_path)}",
            "username": "Traffic Violation Bot",
            "icon_emoji": ":warning:"
        }
        
        headers = {
            "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Send text message
        text_response = requests.post(
            "https://slack.com/api/chat.postMessage",
            json=text_payload,
            headers=headers
        )
        
        # Send image file
        with open(image_path, "rb") as file:
            files = {
                "file": (os.path.basename(image_path), file, "image/jpeg")
            }
            data = {
                "channels": SLACK_CHANNEL,
                "initial_comment": f"🚨 Traffic Violation Evidence",
                "filename": os.path.basename(image_path)
            }
            
            file_response = requests.post(
                "https://slack.com/api/files.upload",
                files=files,
                data=data,
                headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
            )
        
        if text_response.status_code == 200 and file_response.status_code == 200:
            print(f"[📤] Slack alert and image sent: {image_path}")
        else:
            print(f"[❌] Slack send failed - Text: {text_response.status_code}, File: {file_response.status_code}")
            
    except Exception as e:
        print(f"[❌] Slack error: {e}")


import base64
import cv2
import numpy as np
import time
import os

def save_processed_image(b64_str, filename):
    """Save all processed images (violations and normal frames)"""
    try:
        # Check if it's a violation
        is_violation = b64_str.startswith("[VIOLATION]|")
        if is_violation:
            b64_str = b64_str.replace("[VIOLATION]|", "")
            output_dir = "output_violation"
        else:
            output_dir = "output_processed"
        
        # Skip empty or error results
        if not b64_str or b64_str in ["", "PASSTHROUGH - UDF DISABLED", "MODEL_UNAVAILABLE", "CIRCUIT_BREAKER_OPEN"]:
            print(f"[⚠️] Skipping empty/error result: {b64_str}")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Decode and save image
        image_data = base64.b64decode(b64_str)
        np_img = np.frombuffer(image_data, dtype=np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        if img is not None:
            timestamp = int(time.time())
            violation_tag = "VIOLATION" if is_violation else "NORMAL"
            full_filename = f"{timestamp}_{violation_tag}_{filename}.jpg"
            path = os.path.join(output_dir, full_filename)
            cv2.imwrite(path, img)
            
            if is_violation:
                print(f"[�] Saved VIOLATION image: {path}")
                # Send Slack alert for violations
                send_violation_to_slack(path, message=f"🚨 Traffic Violation: {filename}")
            else:
                print(f"[📸] Saved processed image: {path}")
                
        else:
            print(f"[❌] Failed to decode image for {filename}")
            
    except Exception as e:
        print(f"[❌] Failed to save image {filename}: {e}")


import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit
from pyspark.sql.types import StringType
import time
import signal
import sys
import gc
import psutil
import os

# Add current directory to Python path for Spark workers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import UDF with error handling
try:
    from spark_streaming.yolo_predict_udf import yolo_udf
    UDF_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ YOLO UDF imported successfully")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Failed to import YOLO UDF: {e}")
    logger.info("🔄 Will run in passthrough mode")
    yolo_udf = None
    UDF_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_memory_status():
    """Check current memory status"""
    memory = psutil.virtual_memory()
    logger.info(f"💾 Memory: {memory.percent}% used, {memory.available // (1024**3)}GB available")
    
    if memory.percent > 90:
        logger.error("❌ CRITICAL: Memory usage > 90%. This will cause connection failures!")
        return False
    elif memory.percent > 80:
        logger.warning("⚠️  WARNING: Memory usage > 80%. Consider freeing memory.")
        return True
    else:
        logger.info("✅ Memory usage is acceptable")
        return True

def create_ultra_conservative_spark_session():
    """Create ultra-conservative Spark session to prevent Arrow socket errors"""
    try:
        # Force garbage collection before creating session
        gc.collect()
        
        # Set environment variables for Python/Arrow stability
        if "ARROW_PRE_0_15_IPC_FORMAT" in os.environ:
            del os.environ["ARROW_PRE_0_15_IPC_FORMAT"]  # Remove problematic variable
        os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
        
        spark = SparkSession.builder \
            .appName("TrafficViolationDetection") \
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.streaming.checkpointLocation", "./checkpoint") \
            .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true") \
            .config("spark.executor.memory", "512m") \
            .config("spark.driver.memory", "512m") \
            .config("spark.driver.maxResultSize", "256m") \
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "50") \
            .config("spark.executor.instances", "1") \
            .config("spark.executor.cores", "1") \
            .config("spark.default.parallelism", "1") \
            .config("spark.sql.shuffle.partitions", "2") \
            .config("spark.network.timeout", "600s") \
            .config("spark.executor.heartbeatInterval", "60s") \
            .config("spark.sql.streaming.kafka.useDeprecatedOffsetFetching", "false") \
            .config("spark.task.maxAttempts", "3") \
            .config("spark.stage.maxConsecutiveAttempts", "6") \
            .config("spark.sql.streaming.stopGracefullyOnShutdown", "true") \
            .config("spark.sql.streaming.kafka.consumer.pollTimeoutMs", "120000") \
            .config("spark.rpc.retry.wait", "5s") \
            .config("spark.rpc.numRetries", "5") \
            .config("spark.rpc.askTimeout", "120s") \
            .config("spark.storage.blockManagerSlaveTimeoutMs", "120s") \
            .config("spark.sql.execution.arrow.fallback.enabled", "true") \
            .config("spark.serializer.objectStreamReset", "50") \
            .config("spark.cleaner.periodicGC.interval", "5min") \
            .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true") \
            .config("spark.python.worker.memory", "256m") \
            .config("spark.python.worker.reuse", "false") \
            .config("spark.task.cpus", "1") \
            .config("spark.sql.execution.arrow.pyspark.selfDestruct.enabled", "true") \
            .config("spark.python.daemon.module", "pyspark.daemon") \
            .config("spark.python.worker.timeout", "300s") \
            .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
            .config("spark.sql.adaptive.skewJoin.enabled", "false") \
            .config("spark.sql.adaptive.localShuffleReader.enabled", "false") \
            .getOrCreate()
        
        # Set very conservative log level to reduce memory usage
        spark.sparkContext.setLogLevel("ERROR")
        logger.info("✅ Ultra-conservative Spark Session created successfully")
        return spark
        
    except Exception as e:
        logger.error(f"❌ Failed to create Spark session: {str(e)}")
        raise e

def free_system_memory():
    """Attempt to free system memory aggressively"""
    try:
        import ctypes
        if hasattr(ctypes, 'windll'):
            # Windows memory cleanup
            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
        
        # Python garbage collection - run multiple times
        for _ in range(3):
            gc.collect()
        
        logger.info("🧹 Attempted aggressive memory cleanup")
    except Exception as e:
        logger.warning(f"Could not free memory: {e}")

def signal_handler(signum, frame):
    """Handle graceful shutdown"""
    logger.info("🛑 Received shutdown signal, stopping gracefully...")
    sys.exit(0)

def create_fallback_processing_stream(spark):
    """Create a fallback stream that processes data without UDF if needed"""
    try:
        # Read stream from Kafka with settings to process all producer frames
        df = spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "localhost:9092") \
            .option("subscribe", "traffic_frames") \
            .option("startingOffsets", "earliest") \
            .option("failOnDataLoss", "false") \
            .option("maxOffsetsPerTrigger", "5") \
            .option("kafka.consumer.session.timeout.ms", "60000") \
            .option("kafka.consumer.heartbeat.interval.ms", "20000") \
            .option("kafka.consumer.request.timeout.ms", "70000") \
            .option("kafka.consumer.max.poll.records", "5") \
            .option("kafka.consumer.fetch.max.wait.ms", "10000") \
            .option("kafka.consumer.enable.auto.commit", "true") \
            .option("kafka.consumer.auto.commit.interval.ms", "30000") \
            .option("kafka.consumer.fetch.max.bytes", "2048000") \
            .load()
        
        logger.info("✅ Ultra-conservative Kafka stream source created")
        
        # Parse base64 string from Kafka with validation
        df_parsed = df.selectExpr("CAST(value AS STRING) as frame_b64") \
                     .filter(col("frame_b64").isNotNull()) \
                     .filter(col("frame_b64") != "") \
                     .filter(col("frame_b64").rlike("^[A-Za-z0-9+/]*={0,2}$"))
        
        return df_parsed
        
    except Exception as e:
        logger.error(f"❌ Failed to create fallback stream: {str(e)}")
        raise e

def main():
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check memory before starting
    if not check_memory_status():
        logger.error("❌ Insufficient memory to run safely. Please close other applications.")
        logger.info("💡 TIP: Close browser tabs, other applications, and restart this script.")
        return
    
    # Attempt to free memory aggressively
    free_system_memory()
    
    spark = None
    query = None
    udf_enabled = True
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            retry_count += 1
            logger.info(f"🔄 Starting attempt {retry_count}/{max_retries}")
            
            # Create ultra-conservative Spark session
            spark = create_ultra_conservative_spark_session()
            
            # Create base stream
            df_parsed = create_fallback_processing_stream(spark)
            
            # Try to apply UDF with circuit breaker pattern
            if udf_enabled and UDF_AVAILABLE:
                try:
                    logger.info("🔄 Attempting to apply YOLO UDF...")
                    df_predicted = df_parsed.withColumn(
                        "result", 
                        when(col("frame_b64").isNotNull(), yolo_udf(col("frame_b64")))
                        .otherwise(lit(""))
                    ).filter(col("result") != "")
                    
                    logger.info("✅ YOLO UDF applied successfully")
                    
                except Exception as udf_error:
                    logger.error(f"❌ UDF failed: {str(udf_error)}")
                    logger.info("🔄 Falling back to passthrough mode...")
                    udf_enabled = False
                    df_predicted = df_parsed.withColumn("result", lit("PASSTHROUGH - UDF DISABLED"))
            else:
                if not UDF_AVAILABLE:
                    logger.info("ℹ️ Running in passthrough mode (UDF not available)")
                else:
                    logger.info("ℹ️ Running in passthrough mode (UDF disabled)")
                df_predicted = df_parsed.withColumn("result", lit("PASSTHROUGH - UDF DISABLED"))
            
            # Write results with enhanced batch processing
            def process_batch(df, epoch_id):
                try:
                    logger.info(f"🔄 Processing batch {epoch_id}...")
                    rows = df.select("result").collect()
                    batch_size = len(rows)
                    
                    if batch_size == 0:
                        logger.info(f"📭 Batch {epoch_id}: No data to process")
                        return
                    
                    violations_count = 0
                    processed_count = 0
                    
                    for i, row in enumerate(rows):
                        result = row["result"]
                        if result and result.strip():
                            save_processed_image(result, f"epoch{epoch_id}_frame{i}")
                            processed_count += 1
                            
                            if result.startswith("[VIOLATION]|"):
                                violations_count += 1
                    
                    logger.info(f"✅ Batch {epoch_id} complete: {processed_count}/{batch_size} processed, {violations_count} violations")
                    
                except Exception as e:
                    logger.error(f"❌ Error in batch {epoch_id}: {str(e)}")
            
            query = df_predicted.writeStream \
                    .foreachBatch(process_batch) \
                    .outputMode("append") \
                    .trigger(processingTime='60 seconds') \
                    .option("checkpointLocation", f"./checkpoint/batch_{int(time.time())}_{retry_count}") \
                    .start()

            
            logger.info("✅ Streaming query started with ultra-conservative settings")
            
            # Monitor query health and memory with extended timeouts
            batch_count = 0
            consecutive_errors = 0
            max_consecutive_errors = 3
            
            while query.isActive:
                try:
                    time.sleep(30)  # Check every 30 seconds for faster monitoring
                    batch_count += 1
                    
                    # Check memory every batch
                    memory = psutil.virtual_memory()
                    
                    # Get query progress for better monitoring
                    progress = query.lastProgress
                    if progress:
                        input_rows = progress.get('inputRowsPerSecond', 0)
                        processed_rows = progress.get('numInputRows', 0)
                        logger.info(f"📊 Batch: {batch_count} | Memory: {memory.percent}% | Input: {processed_rows} rows | Rate: {input_rows:.2f} rows/sec")
                    else:
                        logger.info(f"📊 Memory: {memory.percent}% used, Batch: {batch_count}, Attempt: {retry_count}")
                    
                    if memory.percent > 85:
                        logger.warning("⚠️  High memory usage detected, forcing cleanup...")
                        free_system_memory()
                    
                    if memory.percent > 95:
                        logger.error("❌ Critical memory usage! Stopping to prevent system crash...")
                        break
                    
                    # Check for exceptions with timeout
                    exception = query.exception()
                    if exception:
                        consecutive_errors += 1
                        logger.error(f"❌ Query exception detected (#{consecutive_errors}): {exception}")
                        
                        if consecutive_errors >= max_consecutive_errors:
                            logger.error("❌ Too many consecutive errors, will retry...")
                            break
                    else:
                        consecutive_errors = 0  # Reset counter on success
                        
                except KeyboardInterrupt:
                    logger.info("🛑 Received keyboard interrupt, stopping...")
                    return
                except Exception as monitor_error:
                    logger.error(f"❌ Monitor error: {str(monitor_error)}")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        break
            
            # If we reach here, either query stopped or had too many errors
            if query.isActive:
                logger.info("🛑 Stopping query due to errors...")
                query.stop()
            
            # If UDF was enabled and we had errors, try disabling it
            if udf_enabled and consecutive_errors > 0:
                logger.info("🔄 UDF seems to be causing issues, will disable for next attempt...")
                udf_enabled = False
            
            # If we completed successfully (no errors), break the retry loop
            if consecutive_errors == 0:
                logger.info("✅ Stream completed successfully!")
                break
                
        except KeyboardInterrupt:
            logger.info("🛑 Received keyboard interrupt, stopping...")
            break
            
        except Exception as e:
            logger.error(f"❌ Error in attempt {retry_count}: {str(e)}")
            
            # Check if it's a memory-related error
            if any(keyword in str(e).lower() for keyword in 
                   ["outofmemoryerror", "connection abort", "socket", "arrow"]):
                logger.error("💾 This appears to be a memory/connection-related error!")
                logger.info("💡 SOLUTIONS:")
                logger.info("   1. Close other applications to free memory")
                logger.info("   2. Restart your computer")
                logger.info("   3. Process smaller batches")
                logger.info("   4. Use a machine with more RAM")
                
                # Disable UDF for next attempt if it's Arrow-related
                if "arrow" in str(e).lower() or "socket" in str(e).lower():
                    udf_enabled = False
                    logger.info("🔄 Disabling UDF for next attempt due to Arrow/Socket error")
        
        finally:
            # Cleanup with aggressive memory management
            if query is not None:
                try:
                    if query.isActive:
                        query.stop()
                    logger.info("✅ Query stopped")
                except Exception as stop_error:
                    logger.warning(f"Error stopping query: {stop_error}")
            
            if spark is not None:
                try:
                    spark.stop()
                    logger.info("✅ Spark session stopped")
                except Exception as stop_error:
                    logger.warning(f"Error stopping Spark: {stop_error}")
                finally:
                    spark = None
            
            # Aggressive memory cleanup between attempts
            free_system_memory()
            time.sleep(10)  # Give system time to cleanup
        
        if retry_count < max_retries:
            logger.info(f"⏳ Waiting before retry attempt {retry_count + 1}...")
            time.sleep(30)
    
    if retry_count >= max_retries:
        logger.error(f"❌ Failed after {max_retries} attempts. Please check system resources and Kafka connectivity.")
    
    # Final memory check
    check_memory_status()

if __name__ == "__main__":
    main()