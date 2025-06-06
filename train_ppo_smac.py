import os
import torch
import torch.optim as optim
import numpy as np
import time
import datetime
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union
from torch.utils.tensorboard import SummaryWriter

from smac_env_wrapper import SMACTextWrapper
from model import QwenActorCritic
from ppo import PPO, PPOBuffer
import config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PPO on SMAC with Qwen")

    # Environment settings
    parser.add_argument(
        "--map_name",
        type=str,
        default=config.SMAC_MAP_NAME,
        help="Name of the SMAC map",
    )
    parser.add_argument("--seed", type=int, default=config.SEED, help="Random seed")

    # Model settings
    parser.add_argument(
        "--model_path",
        type=str,
        default=config.QWEN_MODEL_PATH,
        help="Path to the Qwen model",
    )
    parser.add_argument(
        "--device_map",
        type=Dict[str, str],
        default=config.DEVICE_MAP,
        help="Device map for the model (e.g., 'auto', 'balanced', etc.)",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether to use LoRA for fine-tuning",
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to load (optional)",
    )

    # PPO settings
    parser.add_argument(
        "--lr", type=float, default=config.LEARNING_RATE, help="Learning rate"
    )
    parser.add_argument(
        "--ppo_epochs", type=int, default=config.PPO_EPOCHS, help="Number of PPO epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size"
    )
    parser.add_argument(
        "--rollout_length",
        type=int,
        default=config.ROLLOUT_LENGTH,
        help="Number of steps to collect before an update",
    )
    parser.add_argument(
        "--gamma", type=float, default=config.GAMMA, help="Discount factor"
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=config.GAE_LAMBDA,
        help="GAE lambda parameter",
    )
    parser.add_argument(
        "--clip_eps", type=float, default=config.CLIP_EPS, help="PPO clipping parameter"
    )
    parser.add_argument(
        "--ent_coef", type=float, default=config.ENT_COEF, help="Entropy coefficient"
    )
    parser.add_argument(
        "--vf_coef",
        type=float,
        default=config.VF_COEF,
        help="Value function coefficient",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=config.MAX_GRAD_NORM,
        help="Maximum gradient norm",
    )
    parser.add_argument(
        "--target_kl", type=float, default=config.TARGET_KL, help="Target KL divergence"
    )

    # Training settings
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=config.TOTAL_TIMESTEPS,
        help="Total timesteps for training",
    )
    parser.add_argument(
        "--log_interval", type=int, default=10, help="Log interval (in episodes)"
    )
    parser.add_argument(
        "--save_interval", type=int, default=1000, help="Save interval (in updates)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./runs",
        help="Directory to save TensorBoard logs",
    )

    return parser.parse_args()


def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(f"可训练参数量: {trainable_params}")
    print(f"总参数量: {all_params}")
    print(f"可训练参数占比: {100 * trainable_params / all_params:.2f}%")


def train():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Create TensorBoard log directory
    init_timestamp = time.strftime("%Y%m%d-%H%M%S")
    tb_log_dir = os.path.join(args.log_dir, f"{args.map_name}_{init_timestamp}")
    if not os.path.exists(tb_log_dir):
        os.makedirs(tb_log_dir)
    writer = SummaryWriter(tb_log_dir)

    # Log hyperparameters
    # backup device map because it is a dict
    device_map_dict = args.device_map
    args.device_map = str(device_map_dict)
    writer.add_hparams(vars(args), {})

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        torch.cuda.manual_seed(args.seed)
    except Exception as e:
        print(f"Error setting CUDA seed: {e}")

    # Create environment
    env = SMACTextWrapper(map_name=args.map_name, seed=args.seed)

    # Create model
    model = QwenActorCritic(
        model_path=args.model_path,
        n_agents=env.n_agents,
        n_actions=env.n_actions,
        use_lora=args.use_lora,
        device_map=device_map_dict,
    )

    print_trainable_parameters(model=model)

    # Load checkpoint if specified
    if args.load_checkpoint:
        if args.use_lora:
            model.load_lora_weights(args.load_checkpoint)
            print(f"Loaded LoRA weights from {args.load_checkpoint}")
        else:
            checkpoint = torch.load(args.load_checkpoint)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"Loaded checkpoint from {args.load_checkpoint}")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create PPO agent
    ppo_agent = PPO(
        model=model,
        optimizer=optimizer,
        clip_eps=args.clip_eps,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
    )

    # Create buffer
    buffer = PPOBuffer(
        size=args.rollout_length,
        n_agents=env.n_agents,
    )

    # Training variables
    total_timesteps = args.total_timesteps
    num_updates = total_timesteps // args.rollout_length
    global_step = 0

    # Logging variables
    episode_rewards = []
    episode_lengths = []
    episode_wins = []
    update_metrics = {
        "value_loss": [],
        "policy_loss": [],
        "entropy": [],
        "kl": [],
        "clip_fraction": [],
        "approx_kl": [],
        "total_loss": [],
    }

    # Training loop
    start_time = time.time()
    print(f"Starting training for {total_timesteps} timesteps...")

    # Initial observation
    static_prompt = (
        config.SYSTEM_PROMPT + "\n" + env.map_config + "\n" + env.unit_config + "\n"
    )
    with torch.no_grad():
        static_cache = model.get_static_kv_cache(static_text=static_prompt)

    text_obs = env.reset()

    episode_reward = 0
    episode_length = 0
    episode_count = 0

    # Progress bar for updates
    progress_bar = tqdm(range(num_updates), desc="Updates")

    for update in range(num_updates):
        # Collect rollout
        for step in range(args.rollout_length):
            global_step += 1
            # Get action mask
            action_mask = env.get_action_mask()
            action_mask_tensor = torch.tensor(
                action_mask, dtype=torch.bool, device=model.out_device
            )

            # Get action and value
            with torch.no_grad():
                action, action_log_prob, _, value = model.get_action_and_value(
                    text_obs=text_obs,
                    action_masks=action_mask_tensor,
                    deterministic=False,
                )

            # Convert tensors to numpy
            action_cpu = action.cpu()
            action_log_prob_cpu = action_log_prob.cpu()
            value_cpu = value.cpu().flatten()
            action_mask_cpu = action_mask_tensor.cpu()

            # Take a step in the environment
            next_text_obs, reward, done, info = env.step(action_cpu)

            # Add to episode statistics
            episode_reward += reward
            episode_length += 1

            # Add to buffer
            buffer.add(
                text_obs=text_obs,
                action=action_cpu,
                action_log_prob=action_log_prob_cpu,
                reward=reward,
                done=done,
                value=value_cpu,
                action_mask=action_mask_cpu,
            )

            # Update observation
            text_obs = next_text_obs

            # Handle episode termination
            if done:
                # Log episode statistics
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                current_win = 1 if info.get("battle_won", False) else 0
                episode_wins.append(current_win)

                writer.add_scalar(
                    "charts/episode_reward", episode_reward, episode_count
                )
                writer.add_scalar(
                    "charts/episode_length", episode_length, episode_count
                )
                writer.add_scalar(
                    "charts/win_rate",
                    np.mean(episode_wins) if episode_wins else 0,
                    episode_count,
                )
                writer.add_scalar(
                    "charts/epsilon_greedy",
                    env.epsilon if hasattr(env, "epsilon") else 0,
                    episode_count,
                )  # If epsilon exists

                # Log episode
                if episode_count % args.log_interval == 0:
                    print(
                        f"Episode {episode_count} | "
                        f"Reward: {episode_reward:.2f} | "
                        f"Length: {episode_length} | "
                        f"Win: {info.get('battle_won', False)}"
                    )

                # Reset episode statistics
                episode_reward = 0
                episode_length = 0
                episode_count += 1

                # Reset environment
                text_obs = env.reset()

        # Compute 1 extra step to advantages and returns
        # This step is done outside the loop to avoid double counting
        with torch.no_grad():
            next_action_mask = env.get_action_mask()
            next_action_mask_tensor = torch.tensor(
                next_action_mask, dtype=torch.bool, device=model.out_device
            )
            _, _, _, next_value = model.get_action_and_value(
                text_obs=text_obs,
                action_masks=next_action_mask_tensor,
                deterministic=False,
            )
            next_value_cpu = next_value.cpu().flatten()

        # Finish the trajectory
        buffer.finish_trajectory(
            last_value=next_value_cpu,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )

        # Update the model
        metrics = ppo_agent.update(
            buffer=buffer,
            n_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
        )

        # Log metrics to TensorBoard
        writer.add_scalar("losses/value_loss", metrics["value_loss"], update)
        writer.add_scalar("losses/policy_loss", metrics["policy_loss"], update)
        writer.add_scalar("losses/entropy", metrics["entropy"], update)
        writer.add_scalar("losses/approx_kl", metrics["approx_kl"], update)
        writer.add_scalar("losses/clip_fraction", metrics["clip_fraction"], update)
        writer.add_scalar("losses/total_loss", metrics["total_loss"], update)

        # Update metrics
        for k, v in metrics.items():
            update_metrics[k].append(v)

        # Log update
        if update % 10 == 0:
            elapsed_time = time.time() - start_time
            print(
                f"Update {update}/{num_updates} | "
                f"FPS: {int((update+1) * args.rollout_length / elapsed_time)} | "
                f"Value Loss: {metrics['value_loss']:.4f} | "
                f"Policy Loss: {metrics['policy_loss']:.4f} | "
                f"Entropy: {metrics['entropy']:.4f} | "
                f"KL: {metrics['approx_kl']:.4f}"
            )

        # Save model
        if update % args.save_interval == 0 and update >= args.save_interval:
            # get timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            checkpoint_path = os.path.join(
                args.save_dir,
                f"{args.map_name}_{init_timestamp}",
                (
                    f"model_{update}@{num_updates}"
                    + f"_Lv{metrics['value_loss']:.4f}_Lp{metrics['policy_loss']:.4f}"
                    + f"_En{metrics['entropy']:.4f}_KL{metrics['approx_kl']:.4f}"
                    + f"_{timestamp}"
                    + ".pt"
                ),
            )
            if args.use_lora:
                model.save_lora_weights(checkpoint_path)
            else:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "update": update,
                        "args": vars(args),
                    },
                    checkpoint_path,
                )
            print(f"Saved checkpoint to {checkpoint_path}")

        # Clear buffer
        buffer.clear()

        # Update progress bar
        progress_bar.update(1)

    # Final save
    # get timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    final_checkpoint_path = os.path.join(
        args.save_dir,
        f"{args.map_name}_{init_timestamp}",
        (
            f"model_final{update}@{num_updates}"
            + f"_Rmax{np.max(episode_rewards):.2f}_WR{np.mean(episode_wins):.2f}"
            + f"_{timestamp}"
            + ".pt"
        ),
    )
    if args.use_lora:
        model.save_lora_weights(final_checkpoint_path)
    else:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "update": num_updates,
                "args": vars(args),
            },
            final_checkpoint_path,
        )
    print(f"Saved final checkpoint to {final_checkpoint_path}")

    # Close environment
    env.close()
    writer.close()

    # Final statistics
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    print(f"Average episode reward: {np.mean(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.2f}")
    print(f"Win rate: {np.mean(episode_wins):.2f}")


if __name__ == "__main__":
    train()
