from controller import Robot, Motor
import cv2
import numpy as np
from utils import detect_yellow_ball, handle_received_message
from movement import FourWheelController
from message import Message

# Initialize Webots Robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())

mv_controller = FourWheelController(robot, timestep)

# Initialize the camera
camera = robot.getDevice("camera")
camera.enable(timestep)

# Initialize the receiver
receiver = robot.getDevice("receiver")
receiver.enable(timestep)

# Initialize 4-wheel motors
left_front_motor = robot.getDevice("front left wheel")
left_back_motor = robot.getDevice("back left wheel")
right_front_motor = robot.getDevice("front right wheel")
right_back_motor = robot.getDevice("back right wheel")

messenger = Message(robot, receiver)

# Main loop
while robot.step(timestep) != -1:
    # Handle received messages
    messenger.handle_received_message()

    # Capture image from Webots camera
    img = np.frombuffer(camera.getImage(), np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert to BGR format for OpenCV

    # Detect yellow ball in the image
    ball_detected = detect_yellow_ball(img)
    
    # If a yellow ball is detected, send a message
    if ball_detected:
        # for motor in [left_front_motor, left_back_motor, right_front_motor, right_back_motor]:
        #     motor.setVelocity(0)

        message = "Yellow ball detected! Stopping"
        print(f"{message}")

    # Get sensor values and update state
    wheel_weights = mv_controller.get_sensor_values()
    speeds = mv_controller.update_state(wheel_weights)
    
    # Update wheel velocities
    if(not messenger.go):
        mv_controller.update_speed([0, 0, 0, 0])
    else:
        mv_controller.update_speed(speeds)

    # Display the output (optional for debugging within Webots)
    cv2.imshow("Ball Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
