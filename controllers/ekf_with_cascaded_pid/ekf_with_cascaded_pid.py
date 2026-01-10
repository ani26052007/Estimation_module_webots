from controller import Robot
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

class DroneEKF:
    def __init__(self, timestep):
        self.timestep = timestep
        self.dt = timestep / 1000.0
        
        self.x = np.zeros(9)
        self.P = np.eye(9) * 0.1
        self.Q = np.diag([0.0001, 0.0001, 0.001, 0.001, 0.01, 0.001, 0.001, 0.01, 0.01])
        self.R_imu = np.diag([0.01, 0.01, 0.05])
        self.R_gps = np.diag([0.05, 0.05, 0.05])
        
        self.last_altitude = 0.0
        self.last_x = 0.0
        self.last_y = 0.0
        self.first_update = True
        
    def state_transition(self, x, dt):
        x_next = x.copy()
        x_next[3] += x[4] * dt
        x_next[5] += x[7] * dt
        x_next[6] += x[8] * dt
        return x_next
    
    def state_jacobian(self, dt):
        F = np.eye(9)
        F[3, 4] = dt
        F[5, 7] = dt
        F[6, 8] = dt
        return F
    
    def predict(self):
        self.x = self.state_transition(self.x, self.dt)
        F = self.state_jacobian(self.dt)
        self.P = F @ self.P @ F.T + self.Q
    
    def update_imu(self, roll_meas, pitch_meas, yaw_rate_meas):
        H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
        ])
        z = np.array([roll_meas, pitch_meas, yaw_rate_meas])
        z_pred = H @ self.x
        y = z - z_pred
        S = H @ self.P @ H.T + self.R_imu
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ H) @ self.P
    
    def update_gps(self, x_meas, y_meas, alt_meas):
        if not self.first_update:
            vz_meas = (alt_meas - self.last_altitude) / self.dt
            vx_meas = (x_meas - self.last_x) / self.dt
            vy_meas = (y_meas - self.last_y) / self.dt
        else:
            vz_meas = vx_meas = vy_meas = 0.0
            self.first_update = False
        
        self.last_altitude = alt_meas
        self.last_x = x_meas
        self.last_y = y_meas
        
        H = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        
        z = np.array([alt_meas, vz_meas, x_meas, y_meas, vx_meas, vy_meas])
        z_pred = H @ self.x
        y = z - z_pred
        R = np.diag([0.05, 0.1, 0.05, 0.05, 0.1, 0.1])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ H) @ self.P
    
    def update(self, roll_meas, pitch_meas, yaw_rate_meas, position_meas):
        self.predict()
        self.update_imu(roll_meas, pitch_meas, yaw_rate_meas)
        self.update_gps(position_meas[0], position_meas[1], position_meas[2])
        return self.get_state()
    
    def get_state(self):
        return {
            'roll': self.x[0], 'pitch': self.x[1], 'yaw_rate': self.x[2],
            'altitude': self.x[3], 'vz': self.x[4],
            'x': self.x[5], 'y': self.x[6], 'vx': self.x[7], 'vy': self.x[8]
        }

class EKFPlotter:
    def __init__(self, dt):
        self.dt = dt
        
        self.full_time = []
        self.full_data = {
            'altitude_raw': [], 'altitude_ekf': [],
            'vz_raw': [], 'vz_ekf': [],
            'roll_raw': [], 'roll_ekf': [],
            'pitch_raw': [], 'pitch_ekf': [],
            'x_raw': [], 'x_ekf': [],
            'y_raw': [], 'y_ekf': [],
        }
        
        self.time = 0.0
    
    def update(self, raw_data, ekf_state):
        self.time += self.dt
        self.full_time.append(self.time)
        
        self.full_data['altitude_raw'].append(raw_data['altitude'])
        self.full_data['altitude_ekf'].append(ekf_state['altitude'])
        self.full_data['vz_raw'].append(raw_data['vz'])
        self.full_data['vz_ekf'].append(ekf_state['vz'])
        self.full_data['roll_raw'].append(np.degrees(raw_data['roll']))
        self.full_data['roll_ekf'].append(np.degrees(ekf_state['roll']))
        self.full_data['pitch_raw'].append(np.degrees(raw_data['pitch']))
        self.full_data['pitch_ekf'].append(np.degrees(ekf_state['pitch']))
        self.full_data['x_raw'].append(raw_data['x'])
        self.full_data['x_ekf'].append(ekf_state['x'])
        self.full_data['y_raw'].append(raw_data['y'])
        self.full_data['y_ekf'].append(ekf_state['y'])
    
    def show_final_summary(self):
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Complete EKF Performance Summary (Full Flight)', fontsize=16, fontweight='bold')
        
        t = self.full_time
        marker_interval = 50
        t_markers = t[::marker_interval]
        
        ax = axes[0, 0]
        ax.plot(t, self.full_data['altitude_raw'], 'b-', alpha=0.5, linewidth=1, label='Raw GPS')
        ax.plot(t_markers, self.full_data['altitude_raw'][::marker_interval], 'bo', markersize=3, alpha=0.7)
        ax.plot(t, self.full_data['altitude_ekf'], 'r-', linewidth=2, label='EKF Filtered')
        ax.set_ylabel('Altitude (m)', fontsize=11)
        ax.set_title('Altitude Estimation', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        ax.plot(t, self.full_data['vz_raw'], 'b-', alpha=0.5, linewidth=1, label='Raw GPS Speed')
        ax.plot(t_markers, self.full_data['vz_raw'][::marker_interval], 'bo', markersize=3, alpha=0.7)
        ax.plot(t, self.full_data['vz_ekf'], 'r-', linewidth=2, label='EKF Filtered')
        ax.set_ylabel('Vz (m/s)', fontsize=11)
        ax.set_title('Vertical Velocity Estimation', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        ax.plot(t, self.full_data['roll_raw'], 'b-', alpha=0.5, linewidth=1, label='Raw IMU')
        ax.plot(t_markers, self.full_data['roll_raw'][::marker_interval], 'bo', markersize=3, alpha=0.7)
        ax.plot(t, self.full_data['roll_ekf'], 'r-', linewidth=2, label='EKF Filtered')
        ax.set_ylabel('Roll (deg)', fontsize=11)
        ax.set_title('Roll Angle Estimation', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        ax.plot(t, self.full_data['pitch_raw'], 'b-', alpha=0.5, linewidth=1, label='Raw IMU')
        ax.plot(t_markers, self.full_data['pitch_raw'][::marker_interval], 'bo', markersize=3, alpha=0.7)
        ax.plot(t, self.full_data['pitch_ekf'], 'r-', linewidth=2, label='EKF Filtered')
        ax.set_ylabel('Pitch (deg)', fontsize=11)
        ax.set_title('Pitch Angle Estimation', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        ax = axes[2, 0]
        ax.plot(t, self.full_data['x_raw'], 'b-', alpha=0.5, linewidth=1, label='Raw GPS')
        ax.plot(t_markers, self.full_data['x_raw'][::marker_interval], 'bo', markersize=3, alpha=0.7)
        ax.plot(t, self.full_data['x_ekf'], 'r-', linewidth=2, label='EKF Filtered')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('X Position (m)', fontsize=11)
        ax.set_title('X Position Estimation', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        ax = axes[2, 1]
        ax.plot(t, self.full_data['y_raw'], 'b-', alpha=0.5, linewidth=1, label='Raw GPS')
        ax.plot(t_markers, self.full_data['y_raw'][::marker_interval], 'bo', markersize=3, alpha=0.7)
        ax.plot(t, self.full_data['y_ekf'], 'r-', linewidth=2, label='EKF Filtered')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Y Position (m)', fontsize=11)
        ax.set_title('Y Position Estimation', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("\n" + "="*70)
        print("EKF PERFORMANCE SUMMARY")
        print("="*70)
        print(f"Total flight time: {t[-1]:.1f} seconds")
        print(f"Data points collected: {len(t)}")
        print("\nFinal Statistics:")
        print(f"  Altitude: {self.full_data['altitude_ekf'][-1]:.3f} m")
        print(f"  Position: X={self.full_data['x_ekf'][-1]:.3f} m, Y={self.full_data['y_ekf'][-1]:.3f} m")
        print(f"  Final Roll: {self.full_data['roll_ekf'][-1]:.2f}°")
        print(f"  Final Pitch: {self.full_data['pitch_ekf'][-1]:.2f}°")
        print("="*70)

class LowPassFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.prev_value = None
    
    def update(self, new_value):
        if self.prev_value is None:
            self.prev_value = new_value
        self.prev_value = self.alpha * new_value + (1 - self.alpha) * self.prev_value
        return self.prev_value

class PID:
    def __init__(self, kp, ki, kd, beta=None, output_limits=None, integrator_limits=None, 
                 derivative_on_measurement=False):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.beta = beta if beta else 0.3
        self.output_limits = output_limits
        self.integrator_limits = integrator_limits
        self.derivative_on_measurement = derivative_on_measurement
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = None
        self.d_filter = LowPassFilter(self.beta)

    def update(self, error, dt, measurement=None):
        p = self.kp * error
        
        self.integral += error * dt
        if self.integrator_limits:
            self.integral = np.clip(self.integral, self.integrator_limits[0], self.integrator_limits[1])
        i = self.ki * self.integral
        
        if self.derivative_on_measurement and measurement is not None:
            if self.prev_measurement is not None and dt > 0:
                d_raw = -self.kd * (measurement - self.prev_measurement) / dt
                d = self.d_filter.update(d_raw)
            else:
                d = 0.0
            self.prev_measurement = measurement
        else:
            if dt > 0:
                d_raw = self.kd * (error - self.prev_error) / dt
                d = self.d_filter.update(d_raw)
            else:
                d = 0.0
        
        self.prev_error = error
        output = p + i + d
        if self.output_limits:
            output = np.clip(output, self.output_limits[0], self.output_limits[1])
        return output

robot = Robot()
timestep = int(robot.getBasicTimeStep())
dt = timestep / 1000.0

motors = [robot.getDevice(name) for name in ["front left propeller", "front right propeller", 
                                              "rear left propeller", "rear right propeller"]]
for m in motors:
    m.setPosition(float('inf'))
    m.setVelocity(0.0)

gps = robot.getDevice("gps")
gps.enable(timestep)
imu = robot.getDevice("inertial unit")
imu.enable(timestep)
gyro = robot.getDevice("gyro")
gyro.enable(timestep)

ekf = DroneEKF(timestep)
plotter = EKFPlotter(dt)
yaw_filter = LowPassFilter(0.3)

pid_pos_x = PID(0.8, 0.02, 1, 0.5, (-0.8, 0.8), (-0.15, 0.15), True)
pid_pos_y = PID(1, 0.02, 0.8, 0.5, (-0.8, 0.8), (-0.15, 0.15), True)
pid_pos_z = PID(1.5, 0.8, 1.5, 0.5, (-0.8, 0.8), (-0.3, 0.3), True)

pid_vel_x = PID(5.0, 0.2, 2.0, 0.4, (-0.35, 0.35), (-0.04, 0.04), True)
pid_vel_y = PID(5.0, 0.2, 2.0, 0.4, (-0.35, 0.35), (-0.04, 0.04), True)
pid_vel_z = PID(18.0, 2.5, 8.0, 0.5, (-25, 25), (-4, 4), True)

pid_att_roll  = PID(1.5, 0.3, 0.6, 0.5, (-0.8, 0.8), (-0.2, 0.2), False)
pid_att_pitch = PID(5, 0.1, 0.2, 0.7, (-0.8, 0.8), (-0.2, 0.2), True)
pid_att_yaw   = PID(3, 0.05, 1.2, 0.6, (-0.5, 0.5), (-0.1, 0.1), True)

pid_roll_rate  = PID(-1.5, -0.05, -0.15, 0.3, (-100, 100), (-5, 5))
pid_pitch_rate = PID(-1.5, -0.05, -0.15, 0.3, (-100, 100), (-5, 5))
pid_yaw_rate   = PID(0.8, 0.02, 0.1, 0.3, (-100, 100), (-5, 5))

K_VERTICAL_THRUST = 68.5
DRONE_MASS = 1.3
g = 9.81
MIXING_MATRIX = np.array([[1, -1, 1, -1], [1, 1, 1, 1], [1, -1, -1, 1], [1, 1, -1, -1]])
MOTOR_DIRECTIONS = np.array([1, -1, -1, 1])

target_x, target_y, target_z = None, None, 1.0
start_time = robot.getTime()

while robot.step(timestep) != -1:
    if robot.getTime() - start_time > 1.0:
        break
    if robot.getTime() > 50:
        break

print(f"Taking off to {target_z} meters with position hold...")
print("EKF + Cascaded PID Control Active")
print("Plots will appear after 50 seconds")
print("=" * 70)

while robot.step(timestep) != -1:
    time = robot.getTime()
    
    if time >= 50.0:
        break
    
    roll_raw, pitch_raw, yaw_raw = imu.getRollPitchYaw()
    position = np.array(gps.getValues())
    gps_speed = gps.getSpeedVector()
    r_rate, p_rate, y_rate = gyro.getValues()
    
    state = ekf.update(roll_raw, pitch_raw, y_rate, position)
    
    roll = state['roll']
    pitch = state['pitch']
    yaw = yaw_filter.update(yaw_raw)
    pos = [state['x'], state['y'], state['altitude']]
    velocity = {'x': state['vx'], 'y': state['vy'], 'z': state['vz']}
    
    if target_x is None and time > start_time + 3.0:
        target_x = pos[0]
        target_y = pos[1]
        print(f"\nPosition hold set at X={target_x:.2f}m, Y={target_y:.2f}m\n")
    
    if target_x is None:
        target_x, target_y = 0.0, 0.0
    
    target_vx_world = pid_pos_x.update(target_x - pos[0], dt, measurement=pos[0])
    target_vy_world = pid_pos_y.update(target_y - pos[1], dt, measurement=pos[1])
    target_vz = pid_pos_z.update(target_z - pos[2], dt, measurement=pos[2])

    cy, sy = np.cos(yaw), np.sin(yaw)
    e_vx_body = cy * (target_vx_world - velocity['x']) + sy * (target_vy_world - velocity['y'])
    e_vy_body = -sy * (target_vx_world - velocity['x']) + cy * (target_vy_world - velocity['y'])
    
    ax_body = pid_vel_x.update(e_vx_body, dt)
    ay_body = pid_vel_y.update(e_vy_body, dt)
    az = pid_vel_z.update(target_vz - velocity['z'], dt, measurement=velocity['z'])
    
    target_pitch_cmd = np.arctan2(ax_body, g)
    target_roll_cmd = np.arctan2(-ay_body, g)
    thrust = np.clip(K_VERTICAL_THRUST + DRONE_MASS * az, 0, 600)

    yaw_err = 0.0 - yaw
    while yaw_err > np.pi: yaw_err -= 2 * np.pi
    while yaw_err < -np.pi: yaw_err += 2 * np.pi

    des_rates = {
        'roll':  pid_att_roll.update(target_roll_cmd - roll, dt, measurement=roll),
        'pitch': pid_att_pitch.update(target_pitch_cmd - pitch, dt, measurement=pitch),
        'yaw':   pid_att_yaw.update(yaw_err, dt)
    }

    u_roll  = pid_roll_rate.update(des_rates['roll'] - r_rate, dt)
    u_pitch = pid_pitch_rate.update(des_rates['pitch'] - p_rate, dt)
    u_yaw   = pid_yaw_rate.update(des_rates['yaw'] - y_rate, dt)

    motor_velocities = (MIXING_MATRIX @ np.array([thrust, u_roll, u_pitch, u_yaw])) * MOTOR_DIRECTIONS
    for i, m in enumerate(motors):
        m.setVelocity(np.clip(motor_velocities[i], -576, 576))
    
    raw_data = {
        'altitude': position[2],
        'vz': gps_speed[2],
        'roll': roll_raw,
        'pitch': pitch_raw,
        'x': position[0],
        'y': position[1]
    }
    plotter.update(raw_data, state)
    
    if int(time * 10) % 10 == 0:
        if target_x is not None:
            pos_error = np.sqrt((pos[0] - target_x)**2 + (pos[1] - target_y)**2)
            print(f"Time: {time:.1f}s | "
                  f"Alt: {pos[2]:.2f}m | "
                  f"Pos: ({pos[0]:.2f}, {pos[1]:.2f}) | "
                  f"Error: {pos_error:.3f}m | "
                  f"Vel: ({velocity['x']:.2f}, {velocity['y']:.2f})m/s")
        else:
            print(f"Time: {time:.1f}s | Stabilizing... | Pos: ({pos[0]:.2f}, {pos[1]:.2f})")

print("\n" + "="*70)
print("Simulation complete! Displaying final plots...")
print("="*70)
plotter.show_final_summary()