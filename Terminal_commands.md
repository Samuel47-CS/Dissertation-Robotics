# Terminal commands
### Ports:
/dev/tty.usbmodem5A680089551
/dev/tty.usbmodem5A680102541


Left is maria curie
Right is my assembly
### Robot names:
sams_left_follower
sams_right_follower

Left Robot port: /dev/tty.usbmodem5A460817631
Right Robot port: /dev/tty.usbmodem5A680089551

### Teleop names:
sams_left_leader
sams_right_leader

Left teleop port: /dev/tty.usbmodem5A460817331
Right teleop port: /dev/tty.usbmodem5A680102541

## Motor Setup
### Robot
lerobot-setup-motors \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5A680102541


lerobot-setup-motors \
    --robot.type=bi_so101_follower \
    --robot.left_arm_port=/dev/tty.usbmodem5A460817331 \
    --robot.right_arm_port=/dev/tty.usbmodem5A680102541

### Teleop
Left \
lerobot-setup-motors \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5A460817331

Right \
lerobot-setup-motors \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5A680102541

lerobot-setup-motors \
    --teleop.type=bi_so101_leader \
    --teleop.left_arm_port=/dev/tty.usbmodem5A460817631 \
    --teleop.right_arm_port=/dev/tty.usbmodem5A680089551


## Calibration
### Robot
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5A460817631 \
    --robot.id=sams_left_follower

lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5A680089551 \
    --robot.id=sams_right_follower


lerobot-calibrate \
    --robot.type=bi_so101_follower \
    --robot.left_arm_port=/dev/tty.usbmodem5A460817631 \
    --robot.right_arm_port=/dev/tty.usbmodem5A680089551 \
    --robot.id=sams_follower_arms

### Teleop
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5A460817331 \
    --teleop.id=sams_left_leader_arms

lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5A680102541 
    --teleop.id=sams_right_leader_arm

lerobot-calibrate \
    --teleop.type=bi_so101_leader \
    --teleop.left_arm_port=/dev/tty.usbmodem5A460817331 \
    --teleop.right_arm_port=/dev/tty.usbmodem5A680102541 \
    --teleop.id=sams_leader_arms


## Teleoperate (1 leader 1 follower)
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5A680089551 \
    --robot.id=sams_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5A680102541 \
    --teleop.id=sams_leader_arm


## Working out Teleoperate for 2 leaders 2 followers
lerobot-teleoperate \
    --robot.type=bi_so101_follower \
    --robot.left_arm_port=/dev/tty.usbmodem5A460817631 \
    --robot.right_arm_port=/dev/tty.usbmodem5A680089551 \
    --robot.id=sams_follower_arms \
    --teleop.type=bi_so101_leader \
    --teleop.left_arm_port=/dev/tty.usbmodem5A460817331 \
    --teleop.right_arm_port=/dev/tty.usbmodem5A680102541 \ 
    --teleop.id=sams_leader_arms
    
    --display_data=true

    <!-- --robot.cameras='{
    left_wrist: {"type": "opencv", "index_or_path": 0, "width": 1920, "height": 1080, "fps": 30},
    right_wrist: {"type": "opencv", "index_or_path": 2, "width": 1920, "height": 1080, "fps": 30},
    top: {"type": "opencv", "index_or_path": 1, "width": 1920, "height": 1080, "fps": 30}
    }' \ -->


lerobot-teleoperate \
    --robot.type=bi_so101_follower \
    --robot.left_arm_port=/dev/tty.usbmodem5A460817631 \
    --robot.right_arm_port=/dev/tty.usbmodem5A680089551 \
    --robot.id=sams_follower_arms \
    --teleop.type=bi_so101_leader \
    --teleop.left_arm_port=/dev/tty.usbmodem5A460817331 \
    --teleop.right_arm_port=/dev/tty.usbmodem5A680102541 \ 
    --teleop.id=sams_leader_arms
    
    --display_data=true

    