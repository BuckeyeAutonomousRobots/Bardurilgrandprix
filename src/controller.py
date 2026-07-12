import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pymavlink import mavutil

MAVLINK_CMD_SIM_RESET = 31000

class DCLDroneEnv(gym.Env):
    """
    Gymnasium Environment wrapping MAVLink motor controls and 
    YOLO vision feedback for PPO training in DCL.
    """
    metadata = {"render_modes": []}

    def __init__(self, sim_conn, vision_rx, shared_data, system_boot_ms, control_hz=30):
        super().__init__()
        
        self.sim_conn = sim_conn
        self.vision_rx = vision_rx
        self.data = shared_data
        self.system_boot_ms = system_boot_ms
        self.dt = 1.0 / control_hz

        # Initialize default runtime states
        self.data["is_crashed"] = False

        # Action Space: 4 normalized motor speeds [0.0, 1.0]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # Observation Space:
        # [gate_visible (0/1), norm_x, norm_y, norm_depth, acc_x, acc_y, acc_z, prev_m0..m3]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float32
        )

        self.last_action = np.zeros(4, dtype=np.float32)
        self.step_count = 0
        self.max_steps = 1000  # Episode truncation limit

    def _send_motor_controls(self, motors):
        """Sends motor speeds (0.0 to 1.0) over MAVLink."""
        front_left, front_right, back_left, back_right = motors
        motor_rpms = [float(front_left), float(front_right), float(back_left), float(back_right), 0, 0, 0, 0]
        
        self.sim_conn.mav.set_actuator_control_target_send(
            int(time.time() * 1e6),
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            0,
            motor_rpms
        )

    def _get_observation(self):
        """Extracts normalized gate detection & IMU readings."""
        gates = self.data.get("gates", [])
        
        if len(gates) > 0:
            target_gate = gates[0]
            gate_visible = 1.0
            norm_x = target_gate[0] / 320.0  # Normalized relative to center
            norm_y = target_gate[1] / 180.0  # Normalized relative to center
            norm_depth = target_gate[2] / 740.0 # Bounding box scale
        else:
            gate_visible = 0.0
            norm_x, norm_y, norm_depth = 0.0, 0.0, 0.0

        obs = np.array([
            gate_visible,
            norm_x,
            norm_y,
            norm_depth,
            self.data.get("acc_x", 0.0),
            self.data.get("acc_y", 0.0),
            self.data.get("acc_z", 0.0),
            self.last_action[0],
            self.last_action[1],
            self.last_action[2],
            self.last_action[3]
        ], dtype=np.float32)

        return obs

    def _calculate_reward(self, obs, terminated):
        # High penalty on collision crash
        if terminated:
            return -100.0

        gate_visible, norm_x, norm_y, norm_depth = obs[0], obs[1], obs[2], obs[3]

        if gate_visible == 0.0:
            return -1.0  # Penalty for losing sight of the gate

        # 1. Centering reward: Keep target centered in camera view
        center_error = np.sqrt(norm_x**2 + norm_y**2)
        centering_reward = 1.0 - center_error

        # 2. Forward progress: Getting closer to gate increases bounding box size
        progress_reward = norm_depth * 2.0

        # 3. Smoothness penalty
        action_penalty = -0.01 * np.sum(np.square(self.last_action))

        return float(centering_reward + progress_reward + action_penalty)

    def arm(self):
        """Arm vehicle via MAVLink command."""
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Clear collision state
        self.data["is_crashed"] = False

        # Reset DCL simulator environment
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            MAVLINK_CMD_SIM_RESET,
            0, 0, 0, 0, 0, 0, 0, 0
        )

        self.arm()
        self.last_action = np.zeros(4, dtype=np.float32)
        self.step_count = 0

        time.sleep(0.1)
        return self._get_observation(), {}

    def step(self, action):
        self.step_count += 1
        self.last_action = action

        # Command motor speeds
        self._send_motor_controls(action)
        time.sleep(self.dt)

        # Update OpenCV stream window
        self.vision_rx.update_window(OUTPUT_MODE=False)

        obs = self._get_observation()
        
        # Check termination flags
        terminated = bool(self.data.get("is_crashed", False))
        truncated = self.step_count >= self.max_steps

        reward = self._calculate_reward(obs, terminated)

        return obs, reward, terminated, truncated, {}