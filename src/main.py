import time
import cv2
from stable_baselines3 import PPO
from pymavlink import mavutil

from vision_rx import VisionRX
from mavlink_rx import MAVLinkRX
from controller import DCLDroneEnv

SIM_SERVER_UDP_IP = "127.0.0.1"
SIM_SERVER_UDP_PORT = 14550
DEBUG_MODE = True

def main():
    system_boot_ms = int(time.time() * 1000)

    # Shared telemetry dictionary
    shared_data = {
        "current_time": 0,
        "timestep": 0.005,
        "acc_x": 0.0, "acc_y": 0.0, "acc_z": 0.0,
        "vel_x": 0.0, "vel_y": 0.0, "vel_z": 0.0,
        "gates": [],
        "is_crashed": False
    }

    print("Connecting MAVLink...", flush=True)
    sim_conn = mavutil.mavlink_connection(f"udp:{SIM_SERVER_UDP_IP}:{SIM_SERVER_UDP_PORT}")
    sim_conn.wait_heartbeat()
    print("Heartbeat received!", flush=True)

    print("Initializing MAVLink & Vision Receivers...", flush=True)
    mavlink_rx = MAVLinkRX.create_mavlink_rx(sim_conn, shared_data)
    vision_rx = VisionRX(shared_data)

    # Instantiate Gymnasium environment
    env = DCLDroneEnv(
        sim_conn=sim_conn,
        vision_rx=vision_rx,
        shared_data=shared_data,
        system_boot_ms=system_boot_ms,
        control_hz=30
    )

    if DEBUG_MODE:
        cv2.namedWindow(vision_rx.window_name, cv2.WINDOW_AUTOSIZE)

    # Initialize PPO Model with TensorBoard tracking
    print("Initializing PPO Model...", flush=True)
    model = PPO.load("ppo_dcl_drone.zip", env=env)
    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     verbose=1,
    #     learning_rate=3e-4,
    #     n_steps=1024,
    #     batch_size=64,
    #     gamma=0.99,
    #     tensorboard_log="./tensorboard_logs/"
    # )

    try:
        print("Starting PPO Training Loop...", flush=True)
        model.learn(total_timesteps=100000)
        model.save("ppo_dcl_drone")
        print("Training completed and model saved to 'ppo_dcl_drone.zip'.")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    finally:
        print("Cleaning up resources...", flush=True)
        mavlink_rx.get_thread_for_join().join(timeout=1.0)
        vision_rx.get_thread_for_join().join(timeout=1.0)

        if DEBUG_MODE:
            cv2.destroyAllWindows()

        print("Client exited successfully!", flush=True)

if __name__ == "__main__":
    main()