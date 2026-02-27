lerobot-calibrate \
    --robot.type=bi_so101_follower \
    --robot.left_arm_port=/dev/ttyACM0 \
    --robot.right_arm_port=/dev/ttyACM2 \
    --robot.id=curie

lerobot-calibrate \
    --teleop.type=bi_so101_leader \
    --teleop.left_arm_port=/dev/ttyACM1 \
    --teleop.right_arm_port=/dev/ttyACM3 \
    --teleop.id=maria