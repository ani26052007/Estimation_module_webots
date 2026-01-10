from controller import Robot
import numpy as np

# Initialize Robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())
dt = timestep / 1000.0

# Motors
front_left_motor = robot.getDevice("front left propeller")
front_right_motor = robot.getDevice("front right propeller")
rear_left_motor = robot.getDevice("rear left propeller")
rear_right_motor = robot.getDevice("rear right propeller")
motors = [front_left_motor, front_right_motor, rear_left_motor, rear_right_motor]

for m in motors:
    m.setPosition(float('inf'))
    m.setVelocity(1.0)

# Sensors
gps = robot.getDevice("gps")
gps.enable(timestep)
imu = robot.getDevice("inertial unit")
imu.enable(timestep)
gyro = robot.getDevice("gyro")
gyro.enable(timestep)

# === CONSTANTS (from Webots demo) ===
K_VERTICAL_THRUST = 68.5      # Base thrust for hovering
K_VERTICAL_OFFSET = 0.6       # Target offset for stability
K_VERTICAL_P = 3         # Vertical P gain
K_VERTICAL_D = 3.0            # Vertical velocity damping
K_ROLL_P = 50.0               # Roll P gain
K_PITCH_P = 30.0              # Pitch P gain

# Mixing matrix for quadcopter (FL, FR, RL, RR)
# Each row: [thrust, roll, pitch, yaw] contributions
MIXING_MATRIX = np.array([
    [1, -1,  1, -1],  # Front Left
    [1,  1,  1,  1],  # Front Right
    [1, -1, -1,  1],  # Rear Left
    [1,  1, -1, -1]   # Rear Right
])

# Motor direction corrections (from PROTO file)
MOTOR_DIRECTIONS = np.array([1, -1, -1, 1])

# Variables
target_altitude = 1.0
prev_altitude = 0.0
prev_position = np.zeros(3)

print("Starting drone with NumPy-enhanced controller...")

# Wait 1 second before starting
start_time = robot.getTime()
while robot.step(timestep) != -1:
    if robot.getTime() - start_time > 1.0:
        break

print(f"Drone initialized. Taking off to {target_altitude} meters...")

# Main loop
while robot.step(timestep) != -1:
    time = robot.getTime()
    
    # === READ SENSORS (NumPy arrays) ===
    rpy = np.array(imu.getRollPitchYaw())  # [roll, pitch, yaw]
    roll, pitch, yaw = rpy
    
    position = np.array(gps.getValues())  # [x, y, z]
    altitude = position[2]
    
    gyro_values = np.array(gyro.getValues())  # [roll_rate, pitch_rate, yaw_rate]
    roll_velocity, pitch_velocity, yaw_velocity = gyro_values
    
    # Calculate velocities using NumPy
    if time > start_time + 1.0:
        velocity = (position - prev_position) / dt
        vertical_velocity = velocity[2]
    else:
        velocity = np.zeros(3)
        vertical_velocity = 0.0
    
    prev_position = position.copy()
    prev_altitude = altitude
    
    # === COMPUTE CONTROL INPUTS ===
    # Attitude control (PD controller)
    roll_input = K_ROLL_P * np.clip(roll, -1.0, 1.0) + roll_velocity
    pitch_input = K_PITCH_P * np.clip(pitch, -1.0, 1.0) + pitch_velocity
    yaw_input = 0.0  # No yaw control
    
    # Altitude control (cubic P + velocity D)
    altitude_error = target_altitude - altitude + K_VERTICAL_OFFSET
    clamped_altitude_error = np.clip(altitude_error, -1.0, 1.0)
    vertical_input = K_VERTICAL_P * clamped_altitude_error**3 - K_VERTICAL_D * vertical_velocity
    
    # === MOTOR MIXING (using matrix multiplication) ===
    # Control vector: [thrust, roll, pitch, yaw]
    control_vector = np.array([
        K_VERTICAL_THRUST + vertical_input,
        roll_input,
        pitch_input,
        yaw_input
    ])
    
    # Matrix multiplication to get motor inputs
    motor_inputs = MIXING_MATRIX @ control_vector
    
    # Apply motor direction corrections
    motor_velocities = motor_inputs * MOTOR_DIRECTIONS
    
    # === APPLY TO MOTORS ===
    for i, motor in enumerate(motors):
        motor.setVelocity(motor_velocities[i])
    
    # === DEBUG OUTPUT ===
    if int(time * 10) % 10 == 0:
        tilt = np.linalg.norm(rpy[:2])  # Magnitude of roll and pitch
        print(f"T:{time:.1f}s R:{np.degrees(roll):6.1f}° P:{np.degrees(pitch):6.1f}° " +
              f"Alt:{altitude:.3f}m→{target_altitude:.1f}m Vz:{vertical_velocity:+.2f}m/s " +
              f"V_in:{vertical_input:+.2f} Tilt:{np.degrees(tilt):.1f}°")
