

class Message:

    def __init__(self, robot, receiver):
        self.robot = robot
        self.receiver = receiver
        self.go = True
        self.last_emitter = ""

    def handle_received_message(self):

        while self.receiver.getQueueLength() > 0:

            message = self.receiver.getString() 

            if "Person detected" not in message :
                print(f"<== {message} is not valid")
            else:
                print(f"<== Received message: {message}")

            detectedBy = "R" if "R" in message else "L"
            if (self.last_emitter == "L" and detectedBy == "R") or (self.last_emitter == "R" and detectedBy == "L"):
                self.go = True
                print(f"{self.robot.name} is resuming action...")
            else:
                self.go = False
                print(f"{self.robot.name} is stopping...")

            self.last_emitter = detectedBy

            self.receiver.nextPacket()  