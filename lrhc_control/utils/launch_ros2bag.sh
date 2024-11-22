#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 --ns <namespace> --id <bag_id> [--output_path <path>]"
    exit 1
}

# Function to handle SIGINT
sigint_handler() {
    echo "[launch_rosbag.sh]:SIGINT received, exiting..."
    exit 0
}
# Set the trap to catch SIGINT and call sigint_handler
trap sigint_handler SIGINT

# Check if the required argument is provided
if [ "$#" -lt 4 ] || [ "$1" != "--ns" ] || [ "$3" != "--id" ]; then
    usage
fi

# Extract the namespace from the arguments
NAMESPACE=$2
BAG_ID=$4

# Default output path to /tmp with namespace and date-time
OUTPUT_PATH="/tmp/rosbag_${NAMESPACE}_$(date +%Y-%m-%d_%H-%M-%S)_${BAG_ID}"

# Check for optional --output_path argument
if [ "$#" -eq 6 ] && [ "$5" == "--output_path" ]; then
    OUTPUT_PATH="$6/rosbag_${NAMESPACE}_$(date +%Y-%m-%d_%H-%M-%S)_${BAG_ID}"
elif [ "$#" -gt 4 ]; then
    usage
fi

# Source ROS setup
source /opt/ros/humble/setup.bash

# Change to the training data directory
cd $HOME/training_data

# Define the topics with the namespace replaced
TOPICS=(
    "/RHCViz_${NAMESPACE}_HandShake"
    "/RHCViz_${NAMESPACE}_hl_refs"
    "/RHCViz_${NAMESPACE}_rhc_actuated_jointnames"
    "/RHCViz_${NAMESPACE}_rhc_q"
    "/RHCViz_${NAMESPACE}_rhc_refs"
    "/RHCViz_${NAMESPACE}_robot_actuated_jointnames"
    "/RHCViz_${NAMESPACE}_robot_q"
)

# Record the topics
ros2 bag record --use-sim-time "${TOPICS[@]}" -o "$OUTPUT_PATH"
