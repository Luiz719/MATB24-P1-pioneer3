import cv2
import numpy as np


def detect_yellow_ball(img):
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    # Create a mask for yellow color
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Detect the largest contour, assumed to be the yellow ball
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500:  # Filter based on area to avoid noise
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Draw a circle around the detected yellow ball
            cv2.circle(img, center, radius, (0, 255, 255), 2)
            return True  # Ball detected
    return False  # No ball detected


# Function to handle received messages
def handle_received_message(receiver):
    while receiver.getQueueLength() > 0:
        message = receiver.getString() 
        print(f"<== Received message: {message}")
        
        # Example: You can implement additional logic based on message content
        # For example, change robot behavior based on command
        if message == "Person detected":
            print("<== Stopping...")
        else:
            print(f"<== {message} is not valid")
        
        receiver.nextPacket()  # Move to the next message in the queue

