#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 --ns <namespace> --id <bag_id> --xbot <0/1> [--output_path <path>]"
    exit 1
}

# Function to handle SIGINT
sigint_handler() {
    echo "[launch_rosbag.sh]:SIGINT received, exiting..."
    exit 0
}
# Set the trap to catch SIGINT and call sigint_handler
trap sigint_handler SIGINT

# Ensure the required arguments are provided
if [ "$#" -lt 8 ] || [ "$1" != "--ns" ] || [ "$3" != "--id" ] || [ "$5" != "--xbot" ]; then
    usage
fi

# Extract the namespace from the arguments
NAMESPACE=$2
BAG_ID=$4
ADD_XBOT_TOPICS=$6

# Validate ADD_XBOT_TOPICS input
if [[ "$ADD_XBOT_TOPICS" != "0" && "$ADD_XBOT_TOPICS" != "1" ]]; then
    echo "Error: --xbot must be 0 or 1, provided $ADD_XBOT_TOPICS"
    usage
fi

# Default output path to /tmp with namespace and date-time
OUTPUT_PATH="/tmp/rosbag_${NAMESPACE}_$(date +%Y-%m-%d_%H-%M-%S)_${BAG_ID}"
# Check for optional --output_path argument
if [ "$#" -eq 8 ] && [ "$7" == "--output_path" ]; then
    OUTPUT_PATH="$8/rosbag_${NAMESPACE}_$(date +%Y-%m-%d_%H-%M-%S)_${BAG_ID}"
fi

# Source ROS setup
source /opt/ros/humble/setup.bash

# Change to the training data directory
cd "$HOME/training_data"

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

# Add XBOT topics if the --xbot flag is provided
if [ "$XBOT" = true ]; then
    XBOT_TLIST=(
        "/xbotcore/command"
        "/xbotcore/imu/imu_link"
        "/xbotcore/joint_device_info"
        "/xbotcore/joint_states"
        "/xbotcore/lifecycle_events"
        "/xbotcore/statistics"
        "/xbotcore/status"
    )
    TOPICS+=("${XBOT_TLIST[@]}")
fi

# Record the topics
ros2 bag record --compression-mode file --compression-format zstd --use-sim-time "${TOPICS[@]}" -o "$OUTPUT_PATH"

