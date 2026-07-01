#
# Sample Python client for the AI GP controller
#

import time

import keyboard

from setup import setup_components
import cv2

# Modify these properties if you want to run the server remotely for example
SIM_SERVER_UDP_IP = "127.0.0.1"
SIM_SERVER_UDP_PORT = 14550

DEBUG_MODE = True

# time since sim started ms
system_boot_ms = int(time.time() * 1000)

# arbitrary shared data between the various components
shared_data = {
    "timestep": 0.005,
    "pos_x": 0.0,
    "pos_y": 0.0,
    "pos_z": 0.0,
    "vel_x": 0.0,
    "vel_y": 0.0,
    "vel_z": 0.0,
    "acc_x": 0.0,
    "acc_y": 0.0,
    "acc_z": 0.0,
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
    "veL_roll": 0.0,
    "vel_pitch": 0.0,
    "vel_yaw": 0.0,
    "target_rel_x": 0.0,
    "target_rel_y": 0.0,
    "target_rel_z": 0.0,
    # Realistically, everything should be lists like this instead of separately defined floats
    # One tuple per gate
    "gates": [[0.0, 0.0, 0.0]],
    }

# setup components
components = setup_components(shared_data, system_boot_ms, SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT)
controller = components['controller']
ts_loop = components['ts_loop']
mavlink_rx = components['mavlink_rx']
vision_rx = components['vision_rx']

if DEBUG_MODE:
    cv2.namedWindow(vision_rx.window_name, cv2.WINDOW_AUTOSIZE)

print("Arming drone...", flush=True)
controller.arm()
print("Starting control loop...", flush=True)

is_running = True
try:
    while is_running:
        controller.update()
        vision_rx.update_window()
        if keyboard.is_pressed('q'):
            is_running = False
except KeyboardInterrupt:
    print("\nProgram interrupted by user. Exiting...")
    is_running = False

# exit
# Timesync not yet implemented
# ts_loop.get_thread_for_join().join(timeout=1.0)
mavlink_rx.get_thread_for_join().join(timeout=1.0)
vision_rx.get_thread_for_join().join(timeout=1.0)

if DEBUG_MODE:
    vision_rx.get_thread_for_join().join()
    cv2.destroyAllWindows()

print("Client exited!", flush=True)
