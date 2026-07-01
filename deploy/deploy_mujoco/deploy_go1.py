import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

PATH_PARENT = Path(__file__).parent
sys.path.append(str(PATH_PARENT))
from utils import MujocoRenderUtils

import imageio
import mujoco
import mujoco.viewer
import numpy as np
import pygame
import yaml
from legged_gym import LEGGED_GYM_ROOT_DIR
from matplotlib import pyplot as plt

OBS_TERM_DIMS = [3, 3, 3, 12, 12, 12]


class OnnxPolicyRunner:
    def __init__(self, policy_path, num_obs):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("Go1 MuJoCo deploy requires onnxruntime for ONNX policy inference.") from exc

        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Go1 ONNX policy not found: {policy_path}")

        self.num_obs = num_obs
        self.session = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.input_dim = self._resolve_input_dim(model_input.shape)
        if self.input_dim % num_obs != 0:
            raise ValueError(f"ONNX input dim {self.input_dim} is not a multiple of num_obs {num_obs}")
        self.history = np.zeros((self.input_dim // num_obs, num_obs), dtype=np.float32)

    def _resolve_input_dim(self, shape):
        if len(shape) != 2:
            raise ValueError(f"Expected ONNX policy input rank 2, got shape {shape}")
        dim = shape[1]
        if isinstance(dim, int):
            return dim
        if isinstance(dim, str) and dim.isdigit():
            return int(dim)
        return self.num_obs

    def _make_input(self, obs):
        self.history[:-1] = self.history[1:]
        self.history[-1] = obs

        chunks = []
        start = 0
        for dim in OBS_TERM_DIMS:
            chunks.append(self.history[:, start : start + dim].reshape(-1))
            start += dim
        return np.concatenate(chunks).astype(np.float32).reshape(1, -1)

    def __call__(self, obs):
        outputs = self.session.run(None, {self.input_name: self._make_input(obs)})
        action = np.asarray(outputs[0], dtype=np.float32).squeeze()
        if action.shape != (12,):
            raise ValueError(f"Expected 12-D action output, got shape {action.shape}")
        return action, outputs[1:]


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def quat_rotate_inverse(q, v):
    q = np.array(q, np.float32)
    v = np.array(v, np.float32)
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def get_xbox_command(joystick, max_cmd):
    pygame.event.pump()
    dead_zone = 0.1
    lx = joystick.get_axis(0)
    ly = joystick.get_axis(1)
    rx = joystick.get_axis(3)
    if abs(lx) < dead_zone:
        lx = 0
    if abs(ly) < dead_zone:
        ly = 0
    if abs(rx) < dead_zone:
        rx = 0
    cmd_x = -ly * max_cmd[0]
    cmd_y = -lx * max_cmd[1]
    cmd_yaw = -rx * max_cmd[2]
    return np.array([cmd_x, cmd_y, cmd_yaw], dtype=np.float32)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="go1.yaml", help="Go1 config file under deploy/deploy_mujoco/configs.")
    parser.add_argument("--save-video", action="store_true", help="Whether to save video of the simulation.")
    parser.add_argument("--visualize-moe-weights", action="store_true", help="Whether to visualize mixture of experts weights.")
    parser.add_argument("--save-moe-latent", action="store_true", help="Whether to save mixture of experts latent vectors.")
    args = parser.parse_args()
    save_video = args.save_video
    visualize_moe_weights = args.visualize_moe_weights
    save_moe_latent = args.save_moe_latent

    pygame.init()
    use_joystick = False
    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        use_joystick = True
        print(f"Detected Joystick: {joystick.get_name()}")
    else:
        print("No Joystick detected. Using default commands from config.")

    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{args.config}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        clip_observations = config["clip_observations"]
        clip_actions = config["clip_actions"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        cmd = np.array(config["cmd_init"], dtype=np.float32)

        idx_model2mj = idx_mj2model = list(range(num_actions))
        if "mujoco_joint_names" in config and "model_joint_names" in config:
            mujoco_joint_names = config["mujoco_joint_names"]
            model_joint_names = config["model_joint_names"]
            idx_model2mj = [model_joint_names.index(joint) for joint in mujoco_joint_names]
            idx_mj2model = [mujoco_joint_names.index(joint) for joint in model_joint_names]

    video_save_dir = str(PATH_PARENT / "videos")
    os.makedirs(video_save_dir, exist_ok=True)

    model_name = os.path.basename(policy_path).split(".")[0]
    cmd_str = f"cmd_{cmd[0]}_{cmd[1]}_{cmd[2]}"

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    renderer = mujoco.Renderer(m, height=360, width=640)
    policy = OnnxPolicyRunner(policy_path, num_obs)

    video_fps = 50
    if save_video:
        video_filename = f"{model_name}_{cmd_str}.mp4"
        video_path = os.path.join(video_save_dir, video_filename)
        sim_fps = 1.0 / m.opt.timestep
        frame_skip = max(1, int(sim_fps / video_fps))
        writer = imageio.get_writer(video_path, fps=video_fps)
        print(f"Video recording will be saved to: {video_path}")

    mujoco_render_utils = MujocoRenderUtils(video_fps, m.opt.timestep)

    if visualize_moe_weights:
        plt.ion()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.set_title(f"Command: Vx={cmd[0]:.2f}, Vy={cmd[1]:.2f}, Wz={cmd[2]:.2f}")
        bars = None

    if save_moe_latent:
        latent_save_dir = str(PATH_PARENT / "data_latents")
        os.makedirs(latent_save_dir, exist_ok=True)
        latent_filename = f"{model_name}_{cmd_str}_latents.npy"
        latent_path = os.path.join(latent_save_dir, latent_filename)
        all_latents = []

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -20.0
        viewer.cam.azimuth = 60.0

        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            vel = d.qvel[:3]
            ang_vel = d.qvel[3:6]
            local_vel = quat_rotate_inverse(d.qpos[3:7], vel)
            local_ang_vel = quat_rotate_inverse(d.qpos[3:7], ang_vel)
            show_str = f"Speed: Vx={local_vel[0]:.2f}, Vy={local_vel[1]:.2f}, Wz={local_ang_vel[2]:.2f}, "

            if use_joystick and counter % control_decimation == 0:
                cmd = get_xbox_command(joystick, config["max_cmd"])
                show_str += f"Cmd: Vx={cmd[0]:.2f}, Vy={cmd[1]:.2f}, Wz={cmd[2]:.2f}"
                print(show_str, end="\r")

            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            mujoco_render_utils.update(cmd, d)

            if save_video and counter % frame_skip == 0:
                renderer.update_scene(d, camera=viewer.cam)
                mujoco_render_utils.update_external_rendering(renderer, ctype="renderer")
                writer.append_data(renderer.render())

            counter += 1
            if counter % control_decimation == 0:
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                ang_vel = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                ang_vel = ang_vel * ang_vel_scale

                obs[:3] = ang_vel
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj[idx_mj2model]
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj[idx_mj2model]
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action[idx_mj2model]
                obs = np.clip(obs, -clip_observations, clip_observations)

                action_result, aux = policy(obs)
                action = np.clip(action_result, -clip_actions, clip_actions)[idx_model2mj]
                if aux:
                    weights = np.asarray(aux[0]).squeeze()
                    latent = np.asarray(aux[-1]).squeeze() if len(aux) > 1 else None
                    if visualize_moe_weights:
                        if bars is None:
                            x = np.arange(len(weights))
                            bars = ax.bar(x, weights)
                            ax.set_ylim(0, 1)
                        else:
                            for bar, w in zip(bars, weights):
                                bar.set_height(w)
                        plt.draw()
                        plt.pause(0.001)
                    if save_moe_latent and latent is not None:
                        all_latents.append(latent)

                target_dof_pos = action * action_scale + default_angles

            mujoco_render_utils.update_external_rendering(viewer, ctype="viewer")
            viewer.sync()

    if save_video:
        print(f"Video saved successfully to {video_path}")
        writer.close()
    if save_moe_latent and len(all_latents) > 0:
        all_latents = np.array(all_latents)
        np.save(latent_path, all_latents)
        print(f"Latent vectors saved successfully to {latent_path}")
