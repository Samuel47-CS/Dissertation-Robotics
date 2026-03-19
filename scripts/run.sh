python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080

# Simultaneous scripts

python -m lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:8080 \
    --robot.type=bi_so_follower \
    --robot.left_arm_config.port=/dev/tty.usbmodem585A0076841 \
    --robot.right_arm_config.port=/dev/tty.usbmodem585A0076841 \
    --robot.id=curie \
    --robot.left_arm_config.cameras='{wrist: {"type": "opencv", "index_or_path": 8, "width": 480, "height": 640, "fps": 30, "rotation": ROTATE_270}}' \
    --robot.right_arm_config.cameras='{wrist: {"type": "opencv", "index_or_path": 6, "width": 480, "height": 640, "fps": 30, "rotation": ROTATE_270}, overhead {"type": "intelrealsense", "serial_number_or_name": 318122300856, "width": 640, "height": 480, "fps": 15, "rotation": NO_ROTATION, "color_mode": RGB}}' \
    --task="Horizontally fold the yellow cloth" \
    --policy_type=smolvla \
    --pretrained_name_or_path=the-sam-uel/folding-no-style-4-batch \
    --policy_device=mps \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True

