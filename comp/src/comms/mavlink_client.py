from __future__ import annotations

import socket
import struct
import subprocess
import threading
import time
from typing import Optional

from pymavlink import mavutil

from src.types import AttitudeCommand, SharedState, VehicleState

ENCAPSULATED_RACE_STATUS_MSG_ID = 1
ENCAPSULATED_TRACK_INFO_MSG_ID = 2


class MavlinkClient:
    """Simulator-only MAVLink v2 UDP client."""

    def __init__(self, connection_string: str, shared_state: SharedState, logger=None) -> None:
        self.connection_string = connection_string
        self.shared_state = shared_state
        self.logger = logger
        self.connection = None
        self._rx_thread: Optional[threading.Thread] = None
        self._hb_thread: Optional[threading.Thread] = None
        self._running = False
        self._boot_monotonic_s = time.monotonic()
        self._track_chunks: dict[int, dict[int | str, bytes | float]] = {}
        self._expected_num_track_chunks: dict[int, int] = {}

    def connect(self, timeout_s: float = 15.0) -> None:
        self._preflight_udp_bind_check()
        self.connection = mavutil.mavlink_connection(
            self.connection_string,
            source_system=42,
            source_component=196,
        )
        heartbeat = self.connection.wait_heartbeat(timeout=timeout_s)
        if heartbeat is None:
            raise RuntimeError(
                f"No MAVLink heartbeat received on {self.connection_string} within {timeout_s:.1f}s"
            )

    def _preflight_udp_bind_check(self) -> None:
        if not self.connection_string.startswith("udpin:"):
            return
        _prefix, host, port_str = self.connection_string.split(":")
        port = int(port_str)
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.bind((host, port))
        except OSError as exc:
            holders = self._find_udp_port_holders(port)
            holder_txt = f" Current holder(s): {', '.join(holders)}." if holders else ""
            raise RuntimeError(
                f"Cannot bind MAVLink UDP {host}:{port}; the port is already in use.{holder_txt} "
                f"Close the stale pilot process and retry."
            ) from exc
        finally:
            probe.close()

    @staticmethod
    def _find_udp_port_holders(port: int) -> list[str]:
        try:
            out = subprocess.check_output(["netstat", "-ano", "-p", "udp"], text=True)
        except Exception:
            return []
        holders: list[str] = []
        needle = f":{port}"
        for line in out.splitlines():
            if needle not in line:
                continue
            parts = line.split()
            if not parts:
                continue
            pid = parts[-1]
            if pid not in holders:
                holders.append(pid)
        return holders

    def start(self) -> None:
        if self.connection is None:
            self.connect()
        self._running = True
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._rx_thread.start()
        self._hb_thread.start()

    def stop(self) -> None:
        self._running = False
        for thread in (self._rx_thread, self._hb_thread):
            if thread is not None:
                thread.join(timeout=1.0)

    def send_attitude_target(self, command: AttitudeCommand) -> None:
        assert self.connection is not None
        elapsed_ms = int((time.monotonic() - self._boot_monotonic_s) * 1000.0)
        # This simulator responds to body-rate + thrust control and ignores the
        # quaternion attitude fields in SET_ATTITUDE_TARGET.
        self.connection.mav.set_attitude_target_send(
            elapsed_ms,
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE,
            [1.0, 0.0, 0.0, 0.0],
            command.roll_rate_rps,
            -command.pitch_rate_rps,
            -command.yaw_rate_rps,
            command.thrust,
        )

    def arm(self) -> None:
        assert self.connection is not None
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,
            0, 0, 0, 0, 0, 0,
        )

    def reset_sim(self, repeats: int = 3, interval_s: float = 0.3) -> None:
        assert self.connection is not None
        for _ in range(max(1, repeats)):
            self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                31000,
                0,
                0, 0, 0, 0, 0, 0, 0,
            )
            time.sleep(interval_s)

    def _heartbeat_loop(self) -> None:
        while self._running:
            self._send_heartbeat()
            self._send_timesync()
            time.sleep(0.25)

    def _send_heartbeat(self) -> None:
        assert self.connection is not None
        self.connection.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0,
            0,
            mavutil.mavlink.MAV_STATE_ACTIVE,
        )

    def _send_timesync(self) -> None:
        assert self.connection is not None
        self.connection.mav.timesync_send(int(time.time_ns()), 0)

    def _rx_loop(self) -> None:
        while self._running:
            try:
                msg = self.connection.recv_match(blocking=False)  # type: ignore[union-attr]
            except ConnectionResetError:
                break
            if msg is None:
                time.sleep(0.001)
                continue
            if msg.get_type() == "BAD_DATA":
                continue
            self._handle_message(msg)

    def _handle_message(self, msg) -> None:
        vehicle = self.shared_state.get_vehicle()
        vehicle.wall_time_s = time.monotonic()
        vehicle.raw[msg.get_type()] = msg.to_dict()
        msg_type = msg.get_type()

        if msg_type == "HEARTBEAT":
            vehicle.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            vehicle.system_status = int(msg.system_status)
            vehicle.heartbeat_wall_time_s = vehicle.wall_time_s
        elif msg_type == "ATTITUDE":
            vehicle.roll_rad = float(msg.roll)
            vehicle.pitch_rad = float(msg.pitch)
            vehicle.yaw_rad = float(msg.yaw)
            vehicle.roll_rate_rps = float(msg.rollspeed)
            vehicle.pitch_rate_rps = float(msg.pitchspeed)
            vehicle.yaw_rate_rps = float(msg.yawspeed)
        elif msg_type == "HIGHRES_IMU":
            vehicle.accel_mps2 = (float(msg.xacc), float(msg.yacc), float(msg.zacc))
            vehicle.gyro_rps = (float(msg.xgyro), float(msg.ygyro), float(msg.zgyro))
        elif msg_type == "LOCAL_POSITION_NED":
            vehicle.position_ned_m = (float(msg.x), float(msg.y), float(msg.z))
            vehicle.velocity_ned_mps = (float(msg.vx), float(msg.vy), float(msg.vz))
            vehicle.position_wall_time_s = vehicle.wall_time_s
        elif msg_type == "COLLISION":
            collision = {
                "id": int(msg.id),
                "threat_level": int(msg.threat_level),
                "impact": float(msg.horizontal_minimum_delta),
                "wall_time_s": vehicle.wall_time_s,
            }
            vehicle.raw["last_collision"] = collision
            if self.logger is not None:
                self.logger.log("collision", collision)
        elif msg_type == "ODOMETRY":
            pos_stale = (
                vehicle.position_wall_time_s is None
                or (vehicle.wall_time_s - vehicle.position_wall_time_s) > 0.35
            )
            if pos_stale:
                vehicle.position_ned_m = (float(msg.x), float(msg.y), float(msg.z))
                vehicle.velocity_ned_mps = (float(msg.vx), float(msg.vy), float(msg.vz))
                vehicle.position_wall_time_s = vehicle.wall_time_s
        elif msg_type == "TIMESYNC" and int(getattr(msg, "ts1", 0)) != 0:
            vehicle.timesync_offset_ns = float(msg.tc1 - msg.ts1)
        elif msg_type == "DATA_TRANSMISSION_HANDSHAKE":
            transfer_id = int(msg.width)
            self._track_chunks[transfer_id] = {"_created_wall_time": time.time()}
            self._expected_num_track_chunks[transfer_id] = int(msg.packets)
        elif msg_type == "ENCAPSULATED_DATA":
            self._handle_encapsulated_data(msg)

        self.shared_state.update_vehicle(vehicle)
        if self.logger is not None and msg_type in {"HEARTBEAT", "LOCAL_POSITION_NED"}:
            self.logger.log(
                "telemetry_rx",
                {
                    "msg_type": msg_type,
                    "wall_time_s": vehicle.wall_time_s,
                    "armed": vehicle.armed,
                    "position_ned_m": vehicle.position_ned_m,
                    "velocity_ned_mps": vehicle.velocity_ned_mps,
                    "attitude_rad": [vehicle.roll_rad, vehicle.pitch_rad, vehicle.yaw_rad],
                },
            )

    def _handle_encapsulated_data(self, msg) -> None:
        raw_payload = bytes(msg.data)
        if not raw_payload:
            return
        data_type = raw_payload[0]
        if data_type == ENCAPSULATED_RACE_STATUS_MSG_ID:
            self._handle_race_status(raw_payload)
        elif data_type == ENCAPSULATED_TRACK_INFO_MSG_ID:
            self._handle_track_data_packet(msg, raw_payload)

    def _handle_race_status(self, raw_payload: bytes) -> None:
        data_type, sim_boot_time_ms, race_start_boot_time_ms, race_finish_time_ns, active_gate_index, last_gate_race_time = struct.unpack_from(
            "<BQqqIq", raw_payload
        )
        _ = data_type
        race_status = {
            "sim_boot_time_ms": sim_boot_time_ms,
            "race_start_boot_time_ms": race_start_boot_time_ms,
            "race_finish_time_ns": race_finish_time_ns,
            "active_gate_index": active_gate_index,
            "last_gate_race_time": last_gate_race_time,
            "updated_wall_time": time.time(),
        }
        self.shared_state.update_race_status(race_status)
        if self.logger is not None:
            self.logger.log("race_status", race_status)

    def _handle_track_data_packet(self, msg, raw_payload: bytes) -> None:
        _data_type, transfer_id = struct.unpack_from("<BH", raw_payload)
        if transfer_id not in self._expected_num_track_chunks:
            return
        payload = raw_payload[3:]
        chunk_map = self._track_chunks.setdefault(transfer_id, {"_created_wall_time": time.time()})
        chunk_map[int(msg.seqnr)] = payload
        expected = self._expected_num_track_chunks[transfer_id]
        chunk_count = len([key for key in chunk_map.keys() if isinstance(key, int)])
        if chunk_count != expected:
            return
        full_payload = bytes().join(chunk_map[i] for i in range(expected) if i in chunk_map)
        if len(full_payload) == 0:
            return
        del self._track_chunks[transfer_id]
        del self._expected_num_track_chunks[transfer_id]
        self._handle_track_data(full_payload)

    def _handle_track_data(self, payload: bytes) -> None:
        (num_gates,) = struct.unpack_from("<H", payload)
        payload = payload[2:]
        gates: list[dict[str, object]] = []
        for _ in range(num_gates):
            gate_id, px, py, pz, ow, ox, oy, oz, width, height = struct.unpack_from("<Hfffffffff", payload)
            payload = payload[38:]
            gates.append(
                {
                    "gate_id": gate_id,
                    "position_ned": [px, py, pz],
                    "orientation_wxyz": [ow, ox, oy, oz],
                    "width": width,
                    "height": height,
                }
            )
        gates.sort(key=lambda gate: int(gate["gate_id"]))
        self.shared_state.update_track_gates(gates)
        if self.logger is not None:
            self.logger.log("track_gates", {"count": len(gates)})
