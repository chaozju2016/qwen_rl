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

# 新增DDP相关导入
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from smac_env_wrapper import SMACTextWrapper
from model import QwenActorCritic
from ppo import PPO, PPOBuffer
import config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PPO on SMAC with Qwen")

    # 新增DDP相关参数
    parser.add_argument(
        "--world_size",
        type=int,
        default=torch.cuda.device_count(),
        help="Number of GPUs to use for training",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for distributed training"
    )
    parser.add_argument(
        "--dist_url",
        type=str,
        default="env://",
        help="URL used to set up distributed training",
    )

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
        "--device",
        type=str,
        default=config.DEVICE,
        help="Device to run the model on ('cuda' or 'cpu')",
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


def setup_distributed(args):
    """初始化分布式训练环境"""
    # 设置CUDA和随机种子
    if args.local_rank == -1:  # 非分布式训练
        device = torch.device(args.device)
    else:  # 分布式训练
        # 初始化进程组
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            os.environ["RANK"] = str(args.local_rank)
            os.environ["WORLD_SIZE"] = str(args.world_size)
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f"cuda:{args.local_rank}")

        # 为每个进程设置不同的随机种子
        args.seed = args.seed + dist.get_rank()

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if "cuda" in str(device):
        torch.cuda.manual_seed(args.seed)

    return device


def train_distributed(args, device):
    """分布式训练主函数"""
    # 确定当前进程的rank
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    is_main_process = local_rank == 0

    # 只在主进程创建保存目录和日志
    if is_main_process:
        # 创建保存目录
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # 创建TensorBoard日志目录
        tb_log_dir = os.path.join(
            args.log_dir, f"{args.map_name}_{time.strftime('%Y%m%d-%H%M%S')}"
        )
        if not os.path.exists(tb_log_dir):
            os.makedirs(tb_log_dir)
        writer = SummaryWriter(tb_log_dir)

        # 记录超参数
        writer.add_hparams(vars(args), {})
    else:
        writer = None

    # 创建环境 - 每个进程都需要自己的环境
    env = SMACTextWrapper(map_name=args.map_name, seed=args.seed)

    # 创建模型
    model = QwenActorCritic(
        model_path=args.model_path,
        n_agents=env.n_agents,
        n_actions=env.n_actions,
        device=device,  # 使用当前进程的设备
        use_lora=args.use_lora,
    )

    # 将模型移动到对应设备
    model.to(device)

    # 创建DDP模型
    if dist.is_initialized():
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )  # 有些参数可能在某些前向传播中未使用

    if is_main_process:
        print_trainable_parameters(model.module if dist.is_initialized() else model)

    # 加载检查点（如果指定）
    if args.load_checkpoint:
        if is_main_process:
            print(f"Loading checkpoint from {args.load_checkpoint}")
        if args.use_lora:
            if isinstance(model, DDP):
                model.module.load_lora_weights(args.load_checkpoint)
            else:
                model.load_lora_weights(args.load_checkpoint)
        else:
            checkpoint = torch.load(args.load_checkpoint, map_location=device)
            if isinstance(model, DDP):
                model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint["model_state_dict"])

    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 创建PPO代理
    ppo_agent = PPO(
        model=model,
        optimizer=optimizer,
        device=device,
        clip_eps=args.clip_eps,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
    )

    # 创建缓冲区
    buffer = PPOBuffer(
        size=args.rollout_length,
        n_agents=env.n_agents,
        device=device,
    )

    # 训练变量
    total_timesteps = args.total_timesteps
    num_updates = total_timesteps // args.rollout_length
    global_step = 0

    # 日志变量
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

    # 训练循环
    start_time = time.time()
    if is_main_process:
        print(f"Starting training for {total_timesteps} timesteps...")

    # 初始观察
    text_obs = env.reset()
    prompt = (
        config.SYSTEM_PROMPT
        + "\n"
        + env.map_config
        + "\n"
        + env.unit_config
        + "\n"
        + text_obs
    )

    episode_reward = 0
    episode_length = 0
    episode_count = 0

    # 进度条（仅在主进程显示）
    if is_main_process:
        progress_bar = tqdm(range(num_updates), desc="Updates")

    for update in range(num_updates):
        # 收集rollout
        for step in range(args.rollout_length):
            global_step += 1
            # 获取动作掩码
            action_mask = env.get_action_mask()
            action_mask_tensor = torch.tensor(
                action_mask, dtype=torch.bool, device=device
            )

            # 获取动作和价值
            with torch.no_grad():
                action, action_log_prob, _, value = (
                    model.module.get_action_and_value(
                        text_obs=prompt,
                        action_masks=action_mask_tensor,
                        deterministic=False,
                    )
                    if isinstance(model, DDP)
                    else model.get_action_and_value(
                        text_obs=prompt,
                        action_masks=action_mask_tensor,
                        deterministic=False,
                    )
                )

            # 转换张量为numpy
            action_np = action.cpu().numpy()
            action_log_prob_np = action_log_prob.cpu().numpy()
            value_np = value.cpu().numpy().flatten()

            # 在环境中执行一步
            next_text_obs, reward, done, info = env.step(action_np)

            # 添加到episode统计信息
            episode_reward += reward
            episode_length += 1

            # 添加到缓冲区
            buffer.add(
                text_obs=prompt,
                action=action_np,
                action_log_prob=action_log_prob_np,
                reward=reward,
                done=done,
                value=value_np,
                action_mask=action_mask,
            )

            # 更新观察
            text_obs = next_text_obs
            prompt = (
                config.SYSTEM_PROMPT
                + "\n"
                + env.map_config
                + "\n"
                + env.unit_config
                + "\n"
                + text_obs
            )

            # 处理episode终止
            if done:
                # 记录episode统计信息
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                current_win = 1 if info.get("battle_won", False) else 0
                episode_wins.append(current_win)

                if is_main_process and writer is not None:
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

                # 记录episode
                if is_main_process and episode_count % args.log_interval == 0:
                    print(
                        f"Episode {episode_count} | "
                        f"Reward: {episode_reward:.2f} | "
                        f"Length: {episode_length} | "
                        f"Win: {info.get('battle_won', False)}"
                    )

                # 重置episode统计信息
                episode_reward = 0
                episode_length = 0
                episode_count += 1

                # 重置环境
                text_obs = env.reset()
                prompt = (
                    config.SYSTEM_PROMPT
                    + "\n"
                    + env.map_config
                    + "\n"
                    + env.unit_config
                    + "\n"
                    + text_obs
                )

        # 计算优势和回报
        with torch.no_grad():
            next_action_mask = env.get_action_mask()
            next_action_mask_tensor = torch.tensor(
                next_action_mask, dtype=torch.bool, device=device
            )
            _, _, _, next_value = (
                model.module.get_action_and_value(
                    text_obs=prompt,
                    action_masks=next_action_mask_tensor,
                    deterministic=False,
                )
                if isinstance(model, DDP)
                else model.get_action_and_value(
                    text_obs=prompt,
                    action_masks=next_action_mask_tensor,
                    deterministic=False,
                )
            )
            next_value_np = next_value.cpu().numpy().flatten()

        # 完成轨迹
        buffer.finish_trajectory(
            last_value=next_value_np,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )

        # 更新模型
        metrics = ppo_agent.update(
            buffer=buffer,
            n_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
        )

        # 记录指标到TensorBoard（仅在主进程）
        if is_main_process and writer is not None:
            writer.add_scalar("losses/value_loss", metrics["value_loss"], update)
            writer.add_scalar("losses/policy_loss", metrics["policy_loss"], update)
            writer.add_scalar("losses/entropy", metrics["entropy"], update)
            writer.add_scalar("losses/approx_kl", metrics["approx_kl"], update)
            writer.add_scalar("losses/clip_fraction", metrics["clip_fraction"], update)
            writer.add_scalar("losses/total_loss", metrics["total_loss"], update)

        # 更新指标
        for k, v in metrics.items():
            update_metrics[k].append(v)

        # 记录更新（仅在主进程）
        if is_main_process and update % 10 == 0:
            elapsed_time = time.time() - start_time
            print(
                f"Update {update}/{num_updates} | "
                f"FPS: {int((update+1) * args.rollout_length / elapsed_time)} | "
                f"Value Loss: {metrics['value_loss']:.4f} | "
                f"Policy Loss: {metrics['policy_loss']:.4f} | "
                f"Entropy: {metrics['entropy']:.4f} | "
                f"KL: {metrics['approx_kl']:.4f}"
            )

        # 保存模型（仅在主进程）
        if is_main_process and update % args.save_interval == 0:
            # 获取时间戳
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            checkpoint_path = os.path.join(
                args.save_dir,
                f"model_{update}@{num_updates}"
                + f"_Lv{metrics['value_loss']:.4f}_Lp{metrics['policy_loss']:.4f}"
                + f"_Rmax{np.max(episode_rewards):.2f}_WR{np.mean(episode_wins):.2f}"
                + f"_{timestamp}"
                + ".pt",
            )
            if args.use_lora:
                if isinstance(model, DDP):
                    model.module.save_lora_weights(checkpoint_path)
                else:
                    model.save_lora_weights(checkpoint_path)
            else:
                torch.save(
                    {
                        "model_state_dict": (
                            model.module.state_dict()
                            if isinstance(model, DDP)
                            else model.state_dict()
                        ),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "update": update,
                        "args": vars(args),
                    },
                    checkpoint_path,
                )
            print(f"Saved checkpoint to {checkpoint_path}")

        # 清空缓冲区
        buffer.clear()

        # 更新进度条（仅在主进程）
        if is_main_process:
            progress_bar.update(1)

    # 最终保存（仅在主进程）
    if is_main_process:
        # 获取时间戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        final_checkpoint_path = os.path.join(
            args.save_dir,
            f"model_final{update}@{num_updates}"
            + f"_Lv{metrics['value_loss']:.4f}_Lp{metrics['policy_loss']:.4f}"
            + f"_Rmax{np.max(episode_rewards):.2f}_WR{np.mean(episode_wins):.2f}"
            + f"_{timestamp}"
            + ".pt",
        )
        if args.use_lora:
            if isinstance(model, DDP):
                model.module.save_lora_weights(final_checkpoint_path)
            else:
                model.save_lora_weights(final_checkpoint_path)
        else:
            torch.save(
                {
                    "model_state_dict": (
                        model.module.state_dict()
                        if isinstance(model, DDP)
                        else model.state_dict()
                    ),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "update": num_updates,
                    "args": vars(args),
                },
                final_checkpoint_path,
            )
        print(f"Saved final checkpoint to {final_checkpoint_path}")

    # 关闭环境
    env.close()
    if is_main_process and writer is not None:
        writer.close()

    # 最终统计（仅在主进程）
    if is_main_process:
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
        print(f"Average episode reward: {np.mean(episode_rewards):.2f}")
        print(f"Average episode length: {np.mean(episode_lengths):.2f}")
        print(f"Win rate: {np.mean(episode_wins):.2f}")


def train():
    """主训练函数"""
    # 解析参数
    args = parse_args()

    # 检查是否使用torchrun启动
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    # 设置分布式环境
    device = setup_distributed(args)

    # 开始分布式训练
    train_distributed(args, device)


if __name__ == "__main__":
    train()
