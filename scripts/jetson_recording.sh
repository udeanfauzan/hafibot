#!/bin/bash
# Control timestamped landmark recording inside the Hafibot ROS container.

set -e

CONTAINER_NAME="fall_detection_container"
ROS_SETUP="source /opt/ros/noetic/setup.bash && source /root/catkin_ws/devel/setup.bash"

usage() {
    echo "Usage:"
    echo "  $0 start <ADL|Fall|UNLABELED> <subject_id> <activity>"
    echo "  $0 stop"
    echo "  $0 status"
    echo "  $0 stats"
    echo ""
    echo "Example:"
    echo "  $0 start ADL S01 controlled_lying"
    echo "  $0 stop"
}

validate_component() {
    local value="$1"
    local name="$2"
    if [[ ! "$value" =~ ^[A-Za-z0-9_-]+$ ]]; then
        echo "Invalid $name: use only letters, numbers, underscore, or hyphen." >&2
        exit 2
    fi
}

run_ros() {
    sudo docker exec "$CONTAINER_NAME" /bin/bash -lc "$ROS_SETUP && $1"
}

case "${1:-}" in
    start)
        label="${2:-}"
        subject_id="${3:-}"
        activity="${4:-}"
        if [[ -z "$label" || -z "$subject_id" || -z "$activity" ]]; then
            usage
            exit 2
        fi
        if [[ "$label" != "ADL" && "$label" != "Fall" && "$label" != "UNLABELED" ]]; then
            echo "Label must be ADL, Fall, or UNLABELED." >&2
            exit 2
        fi
        validate_component "$subject_id" "subject_id"
        validate_component "$activity" "activity"
        run_ros "rosparam set /fall_detection/recording_label '$label'"
        run_ros "rosparam set /fall_detection/recording_subject '$subject_id'"
        run_ros "rosparam set /fall_detection/recording_activity '$activity'"
        run_ros "rosservice call /fall_detection/set_recording 'data: true'"
        ;;
    stop)
        run_ros "rosservice call /fall_detection/set_recording 'data: false'"
        ;;
    status)
        run_ros "rosservice call /fall_detection/recording_status"
        ;;
    stats)
        run_ros "rostopic echo /fall_detection/processing_stats"
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        usage
        exit 2
        ;;
esac
