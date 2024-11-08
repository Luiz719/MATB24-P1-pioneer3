from controller import Robot
import cv2
import numpy as np

# Initialize Webots Robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Initialize the camera
camera = robot.getDevice("camera")
camera.enable(timestep)

# Initialize the receiver
receiver = robot.getDevice("receiver")
receiver.enable(timestep)

# Initialize HOG descriptor with pre-trained SVM for person detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Function to handle received messages
def handle_received_message():
    while receiver.getQueueLength() > 0:
        message = receiver.getData().decode("utf-8")  # Decode the message
        print(f"Received message: {message}")
        
        # Example: You can implement additional logic based on message content
        # For example, change robot behavior based on command
        if message == "stop":
            print("Stopping person detection as per command.")
        elif message == "start":
            print("Starting person detection as per command.")
        
        receiver.nextPacket()  # Move to the next message in the queue

# Main loop
while robot.step(timestep) != -1:
    # Handle received messages
    handle_received_message()

    # Capture image from Webots camera
    img = np.frombuffer(camera.getImage(), np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert to BGR format for OpenCV

    # Detect persons in the image
    (persons, _) = hog.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.05)
    
    # Draw bounding boxes around detected persons
    for (x, y, w, h) in persons:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the output (optional for debugging within Webots)
    cv2.imshow("Person Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
