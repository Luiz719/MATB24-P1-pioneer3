

class PedestrianTracker:
    def __init__(self):
        self.no_detection_frames = 0
        self.frames_threshold = 40  # Number of consecutive frames without detection to confirm absence
        self.person_detected = False  # Whether a person has been detected in the current sequence
        
    def update(self, detected):
        if detected:
            # Reset the counter when we see a person
            self.no_detection_frames = 0
            
            # If this is the first detection in this sequence, return True
            if not self.person_detected:
                self.person_detected = True
                return True
                
        else:
            # Increment counter when no person is detected
            self.no_detection_frames += 1
            
            # If we haven't seen a person for enough frames, reset the state
            if self.no_detection_frames >= self.frames_threshold:
                self.person_detected = False
                self.no_detection_frames = 0
                
        return False  # No new detection sequence started