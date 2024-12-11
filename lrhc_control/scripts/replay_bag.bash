#!/bin/bash

# Check if rosbag file argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <rosbag_file_name> [-r playback_rate] [--ros1]"
    exit 1
fi

# Parse arguments
rosbag_file=$1
playback_rate=1
use_ros1=false

# Parse optional flags
shift
while [[ $# -gt 0 ]]; do
    case $1 in
        -r)
            playback_rate=$2
            shift 2
            ;;
        --ros1)
            use_ros1=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Play rosbag file using the appropriate ROS version
if [ "$use_ros1" = true ]; then
    echo "Replaying rosbag using ROS1 Noetic"
    source /opt/ros/noetic/setup.bash
    rosparam set /use_sim_time true
    rosbag play --pause -r $playback_rate $rosbag_file
else
    echo "Replaying rosbag using ROS2 Humble"
    source /opt/ros/humble/setup.bash
    ros2 bag play -p -r $playback_rate $rosbag_file
fi
