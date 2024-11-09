from controller import Robot
import cv2
import numpy as np
from pedestrian_tracker import PedestrianTracker

# Initialize Webots Robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Initialize the camera
camera = robot.getDevice("camera")
camera.enable(timestep)

# Initialize the emitter
emitter = robot.getDevice("emitter")

# Initialize HOG descriptor with pre-trained SVM for person detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

tracker = PedestrianTracker()

while robot.step(timestep) != -1:
    # Capture image from Webots camera
    img = np.frombuffer(camera.getImage(), np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert to BGR format for OpenCV

    # Detect persons in the image
    (persons, _) = hog.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.05)
    
    # Update tracker and check for state change
    if tracker.update(len(persons) > 0):
        message = f"{robot.name}->Person detected"
        emitter.send(message)  
        print(f"==> {robot.name} sent message: {message}")
    
    # Draw bounding boxes around detected persons (for debugging)
    for (x, y, w, h) in persons:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the output (optional for debugging within Webots)
    cv2.imshow("Person Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()