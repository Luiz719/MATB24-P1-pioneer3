from controller import Robot

# Constants
MAX_SPEED = 6.4
MAX_SENSOR_NUMBER = 16
MAX_SENSOR_VALUE = 1024
MIN_DISTANCE = 1.0
WHEEL_WEIGHT_THRESHOLD = 100

# Sensor weights for obstacle avoidance
# Format: [front_left, front_right, back_left, back_right]
SENSOR_WEIGHTS = [
    {"weights": [150, 0, 0, 0]},    # Sensor 0
    {"weights": [200, 0, 0, 0]},    # Sensor 1
    {"weights": [300, 0, 0, 0]},    # Sensor 2
    {"weights": [600, 0, 0, 0]},    # Sensor 3
    {"weights": [0, 600, 0, 0]},    # Sensor 4
    {"weights": [0, 300, 0, 0]},    # Sensor 5
    {"weights": [0, 200, 0, 0]},    # Sensor 6
    {"weights": [0, 150, 0, 0]},    # Sensor 7
    {"weights": [0, 0, 0, 0]},      # Sensor 8-15 (unused)
    {"weights": [0, 0, 0, 0]},
    {"weights": [0, 0, 0, 0]},
    {"weights": [0, 0, 0, 0]},
    {"weights": [0, 0, 0, 0]},
    {"weights": [0, 0, 0, 0]},
    {"weights": [0, 0, 0, 0]},
    {"weights": [0, 0, 0, 0]}
]

class State:
    FORWARD = 0
    LEFT = 1
    RIGHT = 2

class FourWheelController:
    def __init__(self, robot, timestep):
        # Initialize the Robot
        self.robot = robot
        self.time_step = timestep

        # Initialize wheels
        self.wheels = {
            'front_left': self.robot.getDevice('front left wheel'),
            'front_right': self.robot.getDevice('front right wheel'),
            'back_left': self.robot.getDevice('back left wheel'),
            'back_right': self.robot.getDevice('back right wheel')
        }
        
        # Set up wheels for velocity control
        for wheel in self.wheels.values():
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0.0)
        
        # Initialize sensors
        self.sensors = []
        for i in range(MAX_SENSOR_NUMBER):
            sensor = self.robot.getDevice(f'so{i}')
            if sensor:
                sensor.enable(self.time_step)
                self.sensors.append(sensor)
                
        
        # Initialize state variables
        self.state = State.FORWARD

    def get_sensor_values(self):
        """Read and process sensor values"""
        wheel_weight_total = [0.0] * 4  # For four wheels
        
        for i, sensor in enumerate(self.sensors):
            sensor_value = sensor.getValue()
            
            if sensor_value == 0.0:
                speed_modifier = 0.0
            else:
                # Convert sensor value to distance
                distance = 5.0 * (1.0 - (sensor_value / MAX_SENSOR_VALUE))
                
                # Calculate speed modifier based on distance
                if distance < MIN_DISTANCE:
                    speed_modifier = 1 - (distance / MIN_DISTANCE)
                else:
                    speed_modifier = 0.0
            
            # Apply weights to all four wheels
            for j in range(4):
                wheel_weight_total[j] += SENSOR_WEIGHTS[i]["weights"][j] * speed_modifier
                
        return wheel_weight_total

    def update_state(self, wheel_weights):
        """Update robot state and calculate wheel speeds"""
        speeds = [0.0] * 4
        
        if self.state == State.FORWARD:
            if wheel_weights[0] > WHEEL_WEIGHT_THRESHOLD:  # Left side detection
                speeds = [
                    0.7 * MAX_SPEED,   # front left
                    -0.7 * MAX_SPEED,  # front right
                    0.7 * MAX_SPEED,   # back left
                    -0.7 * MAX_SPEED   # back right
                ]
                self.state = State.LEFT
            elif wheel_weights[1] > WHEEL_WEIGHT_THRESHOLD:  # Right side detection
                speeds = [
                    -0.7 * MAX_SPEED,  # front left
                    0.7 * MAX_SPEED,   # front right
                    -0.7 * MAX_SPEED,  # back left
                    0.7 * MAX_SPEED    # back right
                ]
                self.state = State.RIGHT
            else:
                speeds = [MAX_SPEED] * 4  # All wheels forward
                
        elif self.state == State.LEFT:
            if any(w > WHEEL_WEIGHT_THRESHOLD for w in wheel_weights):
                speeds = [
                    0.7 * MAX_SPEED,   # front left
                    -0.7 * MAX_SPEED,  # front right
                    0.7 * MAX_SPEED,   # back left
                    -0.7 * MAX_SPEED   # back right
                ]
            else:
                speeds = [MAX_SPEED] * 4
                self.state = State.FORWARD
                
        elif self.state == State.RIGHT:
            if any(w > WHEEL_WEIGHT_THRESHOLD for w in wheel_weights):
                speeds = [
                    -0.7 * MAX_SPEED,  # front left
                    0.7 * MAX_SPEED,   # front right
                    -0.7 * MAX_SPEED,  # back left
                    0.7 * MAX_SPEED    # back right
                ]
            else:
                speeds = [MAX_SPEED] * 4
                self.state = State.FORWARD
                
        return speeds

    def update_speed(self, speeds):
        # Update wheel velocities
        self.wheels['front_left'].setVelocity(speeds[0])
        self.wheels['front_right'].setVelocity(speeds[1])
        self.wheels['back_left'].setVelocity(speeds[2])
        self.wheels['back_right'].setVelocity(speeds[3])
            

