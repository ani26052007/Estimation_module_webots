"""my_controller_check controller."""
from controller import Robot, InertialUnit

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# Sensors
gps = robot.getDevice("gps")
gps.enable(timestep)

imu = robot.getDevice("inertial unit")
imu.enable(timestep)

gyro = robot.getDevice("gyro")
gyro.enable(timestep)

# Get the noise value
print("IMU noise:", imu.getNoise())


# List all devices
for i in range(robot.getNumberOfDevices()):
    d = robot.getDeviceByIndex(i)
    print(d.getName(), d.getNodeType())

# Main loop
while robot.step(timestep) != -1:
    pass