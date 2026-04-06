# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real_time", action="store_true", default=True, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import numpy as np

# from rsl_rl.runners import OnPolicyRunner
from modules import OnPolicyRunnerCustom
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

# from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_fault.envs.vecenv_wrapper import CustomRslRlVecEnvWrapper
from isaaclab_fault_tasks.go2.agents.rsl_rl_ppo_cfg import RslRlOnPolicyRunnerCfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)

import isaacsim.core.utils.stage as stage_utils
from pxr import UsdPhysics, UsdGeom, Gf, Sdf
import torch
if not args_cli.headless:
    import omni.ui as ui 
def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = CustomRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunnerCustom(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    # hist_latent = []
    # hist_latent_label = []
    episode_log = []
    base_vel_log = []
    cmd_vel_log = []
    base_vel_eps_log = []
    cmd_vel_eps_log = []
    task_log = []
    done_log = []
    # reset environment
    obs, extras = env.get_observations()
    obs_hist = extras["observations"]["history"]
    obs, obs_hist = obs.to(agent_cfg.device), obs_hist.to(agent_cfg.device)
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
            # breakpoint()
        asset = env.unwrapped.scene["robot"]
        # breakpoint()
        # Vis debug joint fault
        if (asset.faulty_joint_idx > 0).sum() > 0:
            
            stage = stage_utils.get_current_stage()
            dof_paths = asset.root_physx_view.dof_paths[0]  # joint prim paths
            body_name_to_id = {n: i for i, n in enumerate(asset.body_names)}

            env_idx = asset.faulty_joint_idx.nonzero()[:,0]
            fault_idx = asset.faulty_joint_idx.nonzero()[:,1]
            parent_ids = []
            child_ids = []
            for j in fault_idx:
                joint_prim = UsdPhysics.Joint.Get(stage, dof_paths[j])
                body0 = joint_prim.GetBody0Rel().GetTargets()[0]  # parent
                body1 = joint_prim.GetBody1Rel().GetTargets()[0]  # child
                parent_name = body0.pathString.split("/")[-1]
                child_name = body1.pathString.split("/")[-1]
                parent_ids.append(body_name_to_id[parent_name])
                child_ids.append(body_name_to_id[child_name])
            parent_ids = torch.tensor(parent_ids, device=asset.device)
            child_ids = torch.tensor(child_ids, device=asset.device)
            _link_idx = child_ids
            # breakpoint()
            # try:
            pos = asset.data.body_pos_w[env_idx, _link_idx, :]
            # except:
            #     breakpoint()
            env.unwrapped._fault_marker.visualize(translations=pos)
            # breakpoint()
            # if timestep % 15 == 0:
            #     print(np.array(asset.joint_names)[(asset.faulty_joint_idx[asset.faulty_joint_idx >= 0]).tolist()])
            #     print(asset.motors_strength[_env_idx])        
        
        # env step
        with torch.inference_mode():
            actions, hist_code = policy(obs, obs_hist)
            obs, rew, done, infos = env.step(actions)
            obs_hist = infos["observations"]["history"]
            obs = ppo_runner.obs_normalizer(obs)
            obs_hist = ppo_runner.obs_hist_normalizer(obs_hist)

        # # command error 
        # cmd_vel = env.unwrapped.command_manager.get_command('base_velocity')
        # base_vel = asset.data.root_ang_vel_b
        # base_vel_eps_log.append(base_vel)
        # cmd_vel_eps_log.append(cmd_vel)

        # success/failure 
        # die = env.unwrapped.termination_manager.get_term("base_contact")
        if done.sum() > 0:
            final_alive_time_s = (infos["episode_duration"] * env.unwrapped.step_dt).clone()
            task_log += final_alive_time_s[done.bool()].tolist()
            done_log += done.nonzero().flatten().tolist() 
            # base_vel_log.append(torch.stack(base_vel_eps_log, dim = 0)[:,done.bool(),:]).detach().cpu().numpy()
            # cmd_vel_log.append(torch.stack(cmd_vel_eps_log, dim = 0)[:,done.bool(),:]).detach().cpu().numpy()
            # base_vel_eps_log = []
            # cmd_vel_eps_log = []
            # breakpoint()
        timestep += 1    
        if len(done_log) >= 128:
            # breakpoint()
            base_vel_log = base_vel_log
            cmd_vel_log = cmd_vel_log
            np.savez("evaluation_result_SR", 
                     epdur = task_log,
                    #  base_vel = base_vel_log,
                    #  cmd_vel = cmd_vel_log
                     )
            break
        # if timestep % 100 == 0:
        #     # print(timestep)   
        #     episode_data = torch.stack(episode_data, dim = 0) 
        #     dataset.append(episode_data)
        #     episode_data = []
        #     print(f"Trajectory {len(dataset)}. collection time: {(time.time() - start_time)*150:.2f}")
        # if len(dataset) == 50:
        #     dataset = torch.stack(dataset, dim = 0) 
        #     np.save("quad_dynamic_dataset.npy", dataset.detach().cpu().numpy())
        #     print("Saving dataset ...")
        #     break
        # breakpoint()
        # if len(hist_latent) == 40:
        #     hist_latent = torch.concat(hist_latent).detach().cpu().numpy()
        #     hist_latent_label = torch.concat(hist_latent_label).detach().cpu().numpy()
        #     print("Saving latent vector data ...")
        #     np.savez("latent_vector.npz", latent_vec = hist_latent, latent_label = hist_latent_label, allow_pickle = True)
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
