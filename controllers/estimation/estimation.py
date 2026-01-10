from controller import Robot
import numpy as np

class DroneEKF:
    def __init__(self, timestep):
        self.timestep = timestep
        self.dt = timestep / 1000.0  # Convert to seconds
        
        # EKF State: [roll, pitch, yaw_rate, altitude, vz]
        self.x = np.zeros(5)
        
        # State covariance matrix
        self.P = np.eye(5) * 0.1
        
        # Process noise covariance (small since simulation is clean)
        self.Q = np.diag([0.0001, 0.0001, 0.001, 0.001, 0.01])
        
        # Measurement noise covariance (very small for clean simulation)
        self.R_imu = np.diag([0.0001, 0.0001, 0.001])  # roll, pitch, yaw_rate
        self.R_alt = 0.001  # altitude
        
        # For velocity estimation
        self.last_altitude = 0.0
        self.first_update = True
        
    def state_transition(self, x, dt):
        """Predict next state using motion model"""
        x_next = x.copy()
        # Integrate altitude with velocity
        x_next[3] += x[4] * dt  # altitude += vz * dt
        # Roll, pitch, yaw_rate assumed constant between updates
        return x_next
    
    def state_jacobian(self, dt):
        """Jacobian of state transition function"""
        F = np.eye(5)
        F[3, 4] = dt  # daltitude/dvz = dt
        return F
    
    def predict(self):
        """EKF Prediction Step"""
        # Predict state
        self.x = self.state_transition(self.x, self.dt)
        
        # Predict covariance
        F = self.state_jacobian(self.dt)
        self.P = F @ self.P @ F.T + self.Q
    
    def update_imu(self, roll_meas, pitch_meas, yaw_rate_meas):
        """Update step with IMU measurements"""
        # Measurement model: z = H*x + v
        H = np.array([
            [1, 0, 0, 0, 0],  # roll
            [0, 1, 0, 0, 0],  # pitch
            [0, 0, 1, 0, 0],  # yaw_rate
        ])
        
        z = np.array([roll_meas, pitch_meas, yaw_rate_meas])
        z_pred = H @ self.x
        
        # Innovation
        y = z - z_pred
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R_imu
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        self.P = (np.eye(5) - K @ H) @ self.P
    
    def update_altimeter(self, alt_meas):
        """Update step with altimeter measurement"""
        # Estimate velocity from altitude change
        if not self.first_update:
            vz_meas = (alt_meas - self.last_altitude) / self.dt
        else:
            vz_meas = 0.0
            self.first_update = False
        
        self.last_altitude = alt_meas
        
        # Measurement model for altitude and velocity
        H = np.array([
            [0, 0, 0, 1, 0],  # altitude
            [0, 0, 0, 0, 1],  # vz
        ])
        
        z = np.array([alt_meas, vz_meas])
        z_pred = H @ self.x
        
        # Innovation
        y = z - z_pred
        
        # Innovation covariance
        R = np.diag([self.R_alt, 0.01])
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        self.P = (np.eye(5) - K @ H) @ self.P
    
    def update(self, roll_meas, pitch_meas, yaw_rate_meas, altitude_meas):
        """Main EKF update - call this each timestep"""
        # Prediction step
        self.predict()
        
        # Update with IMU
        self.update_imu(roll_meas, pitch_meas, yaw_rate_meas)
        
        # Update with altimeter
        self.update_altimeter(altitude_meas)
        
        return self.get_state()
    
    def get_state(self):
        """Return current state estimate"""
        return {
            'roll': self.x[0],
            'pitch': self.x[1],
            'yaw_rate': self.x[2],
            'altitude': self.x[3],
            'vz': self.x[4]
        }


# Initialize robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())
dt = timestep / 1000.0

# Initialize motors
motor_names = ["front left propeller", "front right propeller", 
                "rear left propeller", "rear right propeller"]
motors = [robot.getDevice(name) for name in motor_names]
for m in motors:
    m.setPosition(float('inf'))
    m.setVelocity(1.0)

# Initialize sensors
gps = robot.getDevice("gps")
gps.enable(timestep)

imu = robot.getDevice("inertial unit")
imu.enable(timestep)

gyro = robot.getDevice("gyro")
gyro.enable(timestep)

# Initialize EKF
ekf = DroneEKF(timestep)

# Controller parameters
K_VERTICAL_THRUST = 68.5
K_VERTICAL_OFFSET = 0.6
K_VERTICAL_P, K_VERTICAL_D = 4.0, 3.0
K_ROLL_P, K_PITCH_P = 50.0, 30.0

# Mixing matrix
MIXING_MATRIX = np.array([
    [1, -1,  1, -1],
    [1,  1,  1,  1],
    [1, -1, -1,  1],
    [1,  1, -1, -1]
])
MOTOR_DIRECTIONS = np.array([1, -1, -1, 1])

# Control variables
target_altitude = 1.0
prev_position = np.zeros(3)
start_time = robot.getTime()

# Wait for initialization
while robot.step(timestep) != -1:
    if robot.getTime() - start_time > 1.0:
        break

print(f"Taking off to {target_altitude} meters...")
print("EKF State Estimation Active")
print("=" * 70)

# Main control loop
while robot.step(timestep) != -1:
    time = robot.getTime()
    
    # Get raw sensor measurements
    roll_raw, pitch_raw, yaw_raw = imu.getRollPitchYaw()
    position = np.array(gps.getValues())
    altitude_raw = position[2]
    roll_velocity_raw, pitch_velocity_raw, yaw_velocity_raw = gyro.getValues()
    
    # Update EKF with sensor measurements
    state = ekf.update(roll_raw, pitch_raw, yaw_velocity_raw, altitude_raw)
    
    # Use FILTERED state from EKF for control
    roll = state['roll']
    pitch = state['pitch']
    altitude = state['altitude']
    vertical_velocity = state['vz']
    yaw_velocity = state['yaw_rate']
    
    # Calculate velocity for first timestep fallback
    if time > start_time + 1.0:
        velocity = (position - prev_position) / dt
        # Use raw velocity as backup if needed
    
    prev_position = position.copy()
    
    # Attitude control using filtered IMU
    roll_input = K_ROLL_P * np.clip(roll, -1.0, 1.0) + roll_velocity_raw
    pitch_input = K_PITCH_P * np.clip(pitch, -1.0, 1.0) + pitch_velocity_raw
    
    # Altitude control using filtered altitude and velocity
    alt_error = target_altitude - altitude + K_VERTICAL_OFFSET
    clamped_error = np.clip(alt_error, -1.0, 1.0)
    
    vertical_input = K_VERTICAL_P * (clamped_error**3) - K_VERTICAL_D * vertical_velocity
    
    # Control vector
    control_vector = np.array([
        K_VERTICAL_THRUST + vertical_input,
        roll_input,
        pitch_input,
        0.0
    ])
    
    # Mix to motor velocities
    motor_velocities = (MIXING_MATRIX @ control_vector) * MOTOR_DIRECTIONS
    
    # Apply motor commands
    for i, motor in enumerate(motors):
        motor.setVelocity(motor_velocities[i])
    
    # Print state every second
    if int(time * 10) % 10 == 0:
        print(f"Time: {time:.1f}s | "
              f"Alt: {altitude:.2f}m (raw: {altitude_raw:.2f}m) | "
              f"Vz: {vertical_velocity:.2f}m/s | "
              f"Tilt: {np.degrees(max(abs(roll), abs(pitch))):.1f}° | "
              f"Yaw rate: {np.degrees(yaw_velocity):.1f}°/s")