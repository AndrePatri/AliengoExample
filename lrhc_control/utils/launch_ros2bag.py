#!/usr/bin/env python3

import argparse
import signal
import subprocess
import sys
from datetime import datetime
import rclpy
from rclpy.node import Node
from rosbag2_py import SequentialWriter, SequentialCompressionWriter, StorageOptions, ConverterOptions, TopicMetadata
from rclpy.qos import QoSProfile
from rclpy.serialization import deserialize_message, serialize_message

def get_topic_type(topic_name):
    """
    Get the data type of a ROS 2 topic using the `ros2 topic info` command.
    """
    try:
        result = subprocess.run(
            ["ros2", "topic", "info", topic_name, "-v"],
            text=True,
            capture_output=True,
            check=True
        )
        for line in result.stdout.splitlines():
            if "Type:" in line:
                return line.split(":")[1].strip()
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error retrieving type for topic {topic_name}: {e}")
        return None


class PeriodicRosbagRecorder(Node):
    def __init__(self, namespace, bag_id, output_path, rate_hz):
        super().__init__(f"rosbag_recorder_{namespace}")
        self.namespace = namespace
        self.bag_id = bag_id
        self.output_path = output_path
        self.rate_hz = rate_hz
        self.writer = None
        self.subscribers = {}
        self.topic_buffers = {}

        self.setup_writer()
        self.retrieve_topic_metadata()
        self.add_topics()
        self.create_subscriptions()

        # Timer to call the `record_data` method periodically at the prescribed rate
        self.timer = self.create_timer(1.0 / self.rate_hz, self.record_data)

        self.get_logger().info(f"Recording to {self.output_path} at {self.rate_hz} Hz...")

    def setup_writer(self):
        storage_options = StorageOptions(
            uri=self.output_path,
            storage_id="sqlite3",
        )
        converter_options = ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )

        # compression_options= CompressionOptions(compression_format='zstd',
        #     compression_mode=CompressionMode.MESSAGE,
        #     compression_queue_size=0,
        #     compression_threads=1)

        self.writer = SequentialCompressionWriter()
        self.writer.open(storage_options, converter_options)

    def retrieve_topic_metadata(self):
        """Fetch topic metadata using the `ros2 topic info` command."""
        topics = [
            f"/clock",
            f"/RHCViz_{self.namespace}_HandShake",
            f"/RHCViz_{self.namespace}_hl_refs",
            f"/RHCViz_{self.namespace}_rhc_actuated_jointnames",
            f"/RHCViz_{self.namespace}_rhc_q",
            f"/RHCViz_{self.namespace}_rhc_refs",
            f"/RHCViz_{self.namespace}_robot_actuated_jointnames",
            f"/RHCViz_{self.namespace}_robot_q",
        ]

        self.topics_with_types = {}
        for topic in topics:
            topic_type = get_topic_type(topic)
            if topic_type:
                self.topics_with_types[topic] = topic_type
                self.get_logger().info(f"Retrieved type '{topic_type}' for topic '{topic}'")
            else:
                self.get_logger().error(f"Unable to determine type for topic '{topic}'. Skipping...")
                sys.exit(1)

    def add_topics(self):
        """Register topics with the rosbag writer."""
        for topic, topic_type in self.topics_with_types.items():
            metadata = TopicMetadata(
                name=topic,
                type=topic_type,
                serialization_format="cdr",
            )
            self.writer.create_topic(metadata)
            self.get_logger().info(f"Registered topic: {topic} with type: {topic_type}")

    def create_subscriptions(self):
        """Create subscriptions to record data from the topics."""
        for topic, topic_type in self.topics_with_types.items():
            try:
                module_name, submodule_name, message_name = topic_type.split("/")
                module = __import__(f"{module_name}.{submodule_name}", fromlist=[message_name])
                message_type = getattr(module, message_name)
            except ImportError as e:
                self.get_logger().error(f"Failed to import message type '{topic_type}' for topic '{topic}': {e}")
                sys.exit(1)

            # Create a subscriber for the topic
            self.subscribers[topic] = self.create_subscription(
                message_type,
                topic,
                lambda msg, topic=topic: self.message_callback(msg, topic),
                QoSProfile(depth=10)
            )
            self.get_logger().info(f"Subscription created for topic: {topic}")

    def message_callback(self, message, topic):
        """Callback to buffer incoming messages for writing."""
        self.topic_buffers[topic] = message  # Store the latest message for each topic

    def record_data(self):
        """Write buffered messages to the rosbag at a periodic rate."""
        print("############")
        try:
            print(self.topic_buffers["/clock"])
            for topic, message in self.topic_buffers.items():
                if not topic == "/clock":
                    if message:
                        self.writer.write(topic, serialize_message(message), 0)
        
        except Exception as e:
            self.get_logger().error(f"Error while recording: {e}")

    def shutdown(self):
        if self.writer is not None:
            self.writer.close()
            self.writer=None


def main():
    recorder = None

    def signal_handler(signal_received, frame):
        print("[rosbag_dynamic.py]: SIGINT received, exiting...")
        recorder.shutdown()
        rclpy.shutdown()
        sys.exit(0)

    # Handle command-line arguments
    parser = argparse.ArgumentParser(description="ROS 2 Bag Recorder with Subscriptions")
    parser.add_argument("--ns", required=True, help="Namespace")
    parser.add_argument("--id", required=True, help="Bag ID")
    parser.add_argument("--output_path", help="Custom output path")
    parser.add_argument("--rate", type=float, default=10, help="Rate (Hz) for periodic writes")
    args = parser.parse_args()

    # Setup signal handler for clean exit
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize ROS
    rclpy.init()

    # Default output path if not provided
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = args.output_path or f"/tmp/rosbag_{args.ns}_{timestamp}_{args.id}"

    try:
        recorder = PeriodicRosbagRecorder(args.ns, args.id, output_path, args.rate)
        rclpy.spin(recorder)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if recorder:
            recorder.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
