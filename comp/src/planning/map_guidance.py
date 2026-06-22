"""Map-based gate geometry for sim gate stack."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class GatePlaneMetrics:
    signed_dist_m: float
    within_bounds: bool
    lateral_m: float
    vertical_m: float


@dataclass(frozen=True)
class GateCommitInfo:
    active: bool
    strength: float
    plane_m: float
    dist_center_m: float
    within_bounds: bool
    through_ned: Vec3
    exit_dir: Vec3
    drive_dir: Vec3
    commit_speed_mps: float


def _quat_forward_ned(q_wxyz) -> Vec3:
    if q_wxyz is None or len(q_wxyz) != 4:
        return (1.0, 0.0, 0.0)
    w, x, y, z = [float(v) for v in q_wxyz]
    fx = 1.0 - 2.0 * (y * y + z * z)
    fy = 2.0 * (x * y + w * z)
    fz = 2.0 * (x * z - w * y)
    norm = math.hypot(fx, fy)
    if norm < 1e-6:
        return (1.0, 0.0, 0.0)
    return (fx / norm, fy / norm, fz)


def course_direction_ned(track_gates: list[dict], gate_idx: int) -> Vec3:
    gate = track_gates[gate_idx]
    quat = gate.get("orientation_wxyz")
    if quat is not None:
        fx, fy, fz = _quat_forward_ned(quat)
        if len(track_gates) > 1:
            gx, gy, _gz = [float(v) for v in gate["position_ned"]]
            nx, ny, _nz = [float(v) for v in track_gates[(gate_idx + 1) % len(track_gates)]["position_ned"]]
            to_next_x, to_next_y = nx - gx, ny - gy
            if math.hypot(to_next_x, to_next_y) > 1e-3 and (fx * to_next_x + fy * to_next_y) < 0.0:
                fx, fy = -fx, -fy
        return (fx, fy, fz)

    gx, gy, gz = [float(v) for v in gate["position_ned"]]
    if len(track_gates) > 1:
        px, py, _pz = [float(v) for v in track_gates[(gate_idx - 1) % len(track_gates)]["position_ned"]]
        nx, ny, _nz = [float(v) for v in track_gates[(gate_idx + 1) % len(track_gates)]["position_ned"]]
        course_x, course_y = nx - px, ny - py
    else:
        course_x, course_y = 1.0, 0.0
    norm = math.hypot(course_x, course_y)
    if norm < 1e-6:
        return (1.0, 0.0, 0.0)
    return (course_x / norm, course_y / norm, 0.0)


def turn_dirs_ned(track_gates: list[dict], gate_idx: int) -> tuple[Vec3, Vec3]:
    if len(track_gates) <= 1:
        exit_dir = course_direction_ned(track_gates, gate_idx)
        return exit_dir, exit_dir

    gate_pos = [float(v) for v in track_gates[gate_idx]["position_ned"]]
    prev_gate = [float(v) for v in track_gates[(gate_idx - 1) % len(track_gates)]["position_ned"]]
    next_gate = [float(v) for v in track_gates[(gate_idx + 1) % len(track_gates)]["position_ned"]]
    in_vec = (gate_pos[0] - prev_gate[0], gate_pos[1] - prev_gate[1], 0.0)
    out_vec = (next_gate[0] - gate_pos[0], next_gate[1] - gate_pos[1], 0.0)
    in_norm = math.hypot(in_vec[0], in_vec[1])
    out_norm = math.hypot(out_vec[0], out_vec[1])
    if in_norm < 1e-6:
        in_dir = course_direction_ned(track_gates, gate_idx)
    else:
        in_dir = (in_vec[0] / in_norm, in_vec[1] / in_norm, 0.0)
    if out_norm < 1e-6:
        out_dir = course_direction_ned(track_gates, gate_idx)
    else:
        out_dir = (out_vec[0] / out_norm, out_vec[1] / out_norm, 0.0)
    return in_dir, out_dir


def exit_direction_ned(track_gates: list[dict], gate_idx: int) -> Vec3:
    if len(track_gates) <= 1:
        return course_direction_ned(track_gates, gate_idx)
    _in_dir, out_dir = turn_dirs_ned(track_gates, gate_idx)
    return out_dir


def gate_target_ned(track_gates: list[dict], gate_idx: int) -> Vec3:
    gate = track_gates[gate_idx]
    gx, gy, gz = [float(v) for v in gate["position_ned"]]
    gate_width = float(gate.get("width", 0.0) or 0.0)
    if len(track_gates) <= 1 or gate_width <= 0.05:
        return (gx, gy, gz)

    in_dir, out_dir = turn_dirs_ned(track_gates, gate_idx)
    exit_dir = course_direction_ned(track_gates, gate_idx)
    turn_signed = in_dir[0] * out_dir[1] - in_dir[1] * out_dir[0]
    turn_mag = abs(turn_signed)
    if turn_mag < 0.05:
        return (gx, gy, gz)

    right_x, right_y = -exit_dir[1], exit_dir[0]
    offset_mag = min(max(gate_width * 0.22, 0.35), 1.5) * min(turn_mag, 1.0)
    sign = -1.0 if turn_signed >= 0.0 else 1.0
    return (gx + sign * right_x * offset_mag, gy + sign * right_y * offset_mag, gz)


def gate_normal_ned(track_gates: list[dict], gate_idx: int, pos_ned: Vec3) -> Vec3:
    gate = track_gates[gate_idx]
    quat = gate.get("orientation_wxyz")
    if quat is not None:
        nx, ny, nz = _quat_forward_ned(quat)
    else:
        nx, ny, nz = course_direction_ned(track_gates, gate_idx)
    gx, gy, _gz = [float(v) for v in gate["position_ned"]]
    rel_x, rel_y = pos_ned[0] - gx, pos_ned[1] - gy
    if rel_x * nx + rel_y * ny > 0.0:
        nx, ny = -nx, -ny
    return (nx, ny, nz)


def gate_plane_metrics(
    track_gates: list[dict],
    gate_idx: int,
    pos_ned: Vec3,
    *,
    gate_pass_radius: float = 1.5,
) -> Optional[GatePlaneMetrics]:
    if not track_gates:
        return None

    gate = track_gates[gate_idx]
    gx, gy, gz = [float(v) for v in gate["position_ned"]]
    px, py, pz = pos_ned
    rel_x, rel_y, rel_z = px - gx, py - gy, pz - gz

    exit_dir = exit_direction_ned(track_gates, gate_idx)
    signed_dist = rel_x * exit_dir[0] + rel_y * exit_dir[1]
    right_x, right_y = -exit_dir[1], exit_dir[0]
    lateral = rel_x * right_x + rel_y * right_y
    vertical = rel_z

    gate_width = float(gate.get("width", 0.0) or 0.0)
    gate_height = float(gate.get("height", 0.0) or 0.0)
    half_width = max(gate_width * 0.5, gate_pass_radius * 0.55)
    half_height = max(gate_height * 0.5, 1.0)
    lateral_margin = max(0.4, gate_width * 0.12)
    vertical_margin = max(0.35, gate_height * 0.12)
    within_bounds = (
        abs(lateral) <= half_width + lateral_margin
        and abs(vertical) <= half_height + vertical_margin
    )
    return GatePlaneMetrics(signed_dist, within_bounds, lateral, vertical)


def gate_approach_target_ned(
    track_gates: list[dict],
    gate_idx: int,
    pos_ned: Vec3,
) -> Vec3:
    gate_center = [float(v) for v in track_gates[gate_idx]["position_ned"]]
    dist_center = math.hypot(gate_center[0] - pos_ned[0], gate_center[1] - pos_ned[1])
    if dist_center < 2.0:
        return (gate_center[0], gate_center[1], gate_center[2])

    in_dir, out_dir = turn_dirs_ned(track_gates, gate_idx)
    plane = gate_plane_metrics(track_gates, gate_idx, pos_ned)
    signed_dist = plane.signed_dist_m if plane is not None else 0.0

    if dist_center > 8.0:
        standoff = max(4.0, min(10.0, dist_center * 0.36))
        return (
            gate_center[0] + in_dir[0] * standoff,
            gate_center[1] + in_dir[1] * standoff,
            gate_center[2],
        )

    if signed_dist <= 0.0 and dist_center < 14.0:
        return (gate_center[0], gate_center[1], gate_center[2])

    if signed_dist > 0.6:
        back_dist = min(4.0, signed_dist + 1.0)
        return (
            gate_center[0] - out_dir[0] * back_dist,
            gate_center[1] - out_dir[1] * back_dist,
            gate_center[2],
        )

    return (gate_center[0], gate_center[1], gate_center[2])


def _quad_bezier(p0: Vec3, p1: Vec3, p2: Vec3, t: float) -> Vec3:
    u = 1.0 - t
    return (
        u * u * p0[0] + 2.0 * u * t * p1[0] + t * t * p2[0],
        u * u * p0[1] + 2.0 * u * t * p1[1] + t * t * p2[1],
        u * u * p0[2] + 2.0 * u * t * p1[2] + t * t * p2[2],
    )


def path_target_ned(
    track_gates: list[dict],
    gate_idx: int,
    pos_ned: Vec3,
    *,
    horizontal_speed_mps: float = 0.0,
    cruise_speed_mps: float = 6.0,
    slowdown_radius_m: float = 12.0,
) -> Vec3:
    current = gate_target_ned(track_gates, gate_idx)
    if len(track_gates) <= 1:
        return current

    prev_target = gate_target_ned(track_gates, (gate_idx - 1) % len(track_gates))
    next_target = gate_target_ned(track_gates, (gate_idx + 1) % len(track_gates))
    p0, p1, p2 = prev_target, current, next_target

    seg_x, seg_y = p1[0] - p0[0], p1[1] - p0[1]
    seg_len2 = seg_x * seg_x + seg_y * seg_y
    if seg_len2 < 1e-6:
        return p1

    progress = max(0.0, min(1.0, ((pos_ned[0] - p0[0]) * seg_x + (pos_ned[1] - p0[1]) * seg_y) / seg_len2))
    dist_to_gate = math.hypot(p1[0] - pos_ned[0], p1[1] - pos_ned[1])
    if dist_to_gate > 12.0:
        return gate_approach_target_ned(track_gates, gate_idx, pos_ned)

    in_dir, out_dir = turn_dirs_ned(track_gates, gate_idx)
    turn_mag = abs(in_dir[0] * out_dir[1] - in_dir[1] * out_dir[0])
    speed_ratio = max(0.0, min(1.35, horizontal_speed_mps / max(cruise_speed_mps, 1.0)))
    near_gate = max(0.0, min(1.0, 1.0 - dist_to_gate / max(slowdown_radius_m, 1.0)))
    lookahead = 0.18 + 0.16 * near_gate + 0.18 * speed_ratio - 0.12 * min(turn_mag, 1.0)
    t = max(0.12, min(0.92, progress + lookahead))
    curve = _quad_bezier(p0, p1, p2, t)
    curve = (curve[0], curve[1], p1[2])
    curve_dist = math.hypot(curve[0] - pos_ned[0], curve[1] - pos_ned[1])
    if dist_to_gate > 10.0 and curve_dist > dist_to_gate * 1.35 + 4.0:
        return p1
    return curve


def gate_commit_info(
    track_gates: list[dict],
    gate_idx: int,
    pos_ned: Vec3,
    *,
    min_speed_mps: float = 2.5,
    cruise_speed_mps: float = 6.0,
) -> Optional[GateCommitInfo]:
    plane = gate_plane_metrics(track_gates, gate_idx, pos_ned)
    if plane is None:
        return None

    signed_dist = plane.signed_dist_m
    within_bounds = plane.within_bounds
    gate_center = [float(v) for v in track_gates[gate_idx]["position_ned"]]
    dist_center = math.hypot(gate_center[0] - pos_ned[0], gate_center[1] - pos_ned[1])
    exit_dir = exit_direction_ned(track_gates, gate_idx)
    exit_norm = max(math.hypot(exit_dir[0], exit_dir[1]), 1e-6)
    exit_dir = (exit_dir[0] / exit_norm, exit_dir[1] / exit_norm, exit_dir[2])

    in_front = signed_dist < 0.35
    aligned = within_bounds or dist_center < 4.5
    active = (
        dist_center < 18.0
        and abs(signed_dist) < 12.0
        and (aligned or in_front or dist_center < 10.0)
    )
    close = max(0.0, min(1.0, 1.0 - dist_center / 18.0))
    front = max(0.0, min(1.0, 1.0 - (abs(signed_dist) - 0.12) / 6.0))
    strength = max(0.40, close * front) if active else 0.0
    drive_dir = exit_dir if signed_dist <= 0.0 else (-exit_dir[0], -exit_dir[1], -exit_dir[2])
    through = (
        gate_center[0] + drive_dir[0] * 2.5,
        gate_center[1] + drive_dir[1] * 2.5,
        gate_center[2],
    )
    commit_speed = min_speed_mps + 1.5 + strength * (cruise_speed_mps - min_speed_mps) * 0.65
    return GateCommitInfo(
        active=active,
        strength=strength,
        plane_m=signed_dist,
        dist_center_m=dist_center,
        within_bounds=within_bounds,
        through_ned=through,
        exit_dir=exit_dir,
        drive_dir=drive_dir,
        commit_speed_mps=commit_speed,
    )


def bearing_from_vehicle(
    pos_ned: Vec3,
    yaw_rad: float,
    target_ned: Vec3,
) -> tuple[float, float, float]:
    dx = target_ned[0] - pos_ned[0]
    dy = target_ned[1] - pos_ned[1]
    dz = target_ned[2] - pos_ned[2]
    body_x = math.cos(yaw_rad) * dx + math.sin(yaw_rad) * dy
    body_y = -math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dy
    horiz = max(math.hypot(dx, dy), 1e-3)
    bearing_x = math.atan2(body_y, body_x)
    bearing_y = math.atan2(dz, horiz)
    return bearing_x, bearing_y, math.sqrt(dx * dx + dy * dy + dz * dz)


def exit_yaw_ned(track_gates: list[dict], gate_idx: int) -> float:
    exit_dir = exit_direction_ned(track_gates, gate_idx)
    return math.atan2(exit_dir[1], exit_dir[0])
