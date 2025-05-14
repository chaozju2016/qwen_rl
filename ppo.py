import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union
from torch.nn.parallel import DistributedDataParallel as DDP


class PPOBuffer:
    """Buffer for storing trajectories experienced by a PPO agent."""

    def __init__(self, size: int, n_agents: int):
        """Initialize the PPO buffer."""
        self.size = size
        self.n_agents = n_agents
        self.clear()

    def clear(self):
        """Clear the buffer."""
        self.text_obs = []  # 每个时间步一个环境观察
        self.actions = []  # 形状为[timestep, n_agents]
        self.action_log_probs = []  # 形状为[timestep, n_agents]
        self.rewards = []  # 形状为[timestep]（SMAC给出团队奖励）
        self.dones = []  # 形状为[timestep]
        self.values = []  # 形状为[timestep, n_agents]
        self.action_masks = []  # 形状为[timestep, n_agents, n_actions]
        self.ptr = 0
        self.trajectory_start_indices = []

    def add(
        self,
        text_obs: str,
        action: np.ndarray,
        action_log_prob: np.ndarray,
        reward: float,
        done: bool,
        value: np.ndarray,
        action_mask: np.ndarray,
    ):
        """
        Add a transition to the buffer.
        action: 形状为[n_agents]，每个智能体的动作
        action_log_prob: 形状为[n_agents]，每个动作的对数概率
        reward: 标量，整个团队的奖励
        value: 形状为[n_agents]，每个智能体的值估计
        """
        if self.ptr == 0:
            self.trajectory_start_indices.append(0)

        # Add the transition to the buffer
        self.text_obs.append(text_obs)
        self.actions.append(action)
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.action_masks.append(action_mask)

        # Increment the pointer
        self.ptr += 1

        # Record next trajectory start index if episode ended
        if done and self.ptr < self.size:
            self.trajectory_start_indices.append(self.ptr)

    def finish_trajectory(
        self, last_value: np.ndarray, gamma: float = 0.99, gae_lambda: float = 0.95
    ):
        """Compute advantages and returns for the current trajectory."""
        # Convert lists to arrays
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        # Get the length of the buffer
        trajectory_length = len(rewards)

        # Initialize arrays for advantages and returns
        # In PPO, we use global critic, so there is only one value/advantage for each timestep
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)

        if isinstance(last_value, np.ndarray):
            assert last_value.shape == (1,)
            values = np.append(values, last_value[None, :], axis=0)
            returns = np.append(returns, last_value, axis=0)
        else:
            assert isinstance(last_value, values.dtype)
            # Extend the values array with the last value
            values = np.append(values, last_value)
            returns = np.append(returns, last_value)

        next_advantage = 0.0
        # Loop through the trajectory backwards
        for t in reversed(range(trajectory_length)):
            # 获取下一状态是否终止
            next_non_terminal = 1.0 - dones[t]

            # 计算TD误差
            # values has been extended with last_value, so we can access t+1
            delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]

            # 计算GAE优势
            advantages[t] = (
                delta + gamma * gae_lambda * next_non_terminal * next_advantage
            )
            next_advantage = advantages[t]

            # 计算回报（用于值函数损失）
            returns[t] = rewards[t] + gamma * returns[t + 1] * next_non_terminal

        # Store the computed advantages and returns
        self.advantages = advantages
        self.returns = returns

    def get(self) -> Dict[str, Union[List, np.ndarray]]:
        """Get the buffer data."""
        data = {
            "text_obs": self.text_obs,
            "actions": np.array(self.actions),
            "action_log_probs": np.array(self.action_log_probs),
            "rewards": np.array(self.rewards),
            "dones": np.array(self.dones),
            "values": np.array(self.values),
            "action_masks": np.array(self.action_masks),
            "advantages": self.advantages,
            "returns": self.returns,
            "trajectory_start_indices": self.trajectory_start_indices,
        }
        return data

    def __len__(self) -> int:
        """Get the current size of the buffer."""
        return self.ptr


class PPO:
    """Proximal Policy Optimization algorithm."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
    ):
        """Initialize the PPO algorithm."""
        self.model = model
        self.optimizer = optimizer
        self.device = self.model.out_device
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

    def update(
        self, buffer: PPOBuffer, n_epochs: int = 4, batch_size: int = 64
    ) -> Dict[str, float]:
        """Update the model using the data in the buffer."""
        # Get data from the buffer
        data = buffer.get()
        text_obs = data["text_obs"]
        actions = torch.tensor(data["actions"], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(
            data["action_log_probs"], dtype=torch.float32, device=self.device
        )
        advantages = torch.tensor(
            data["advantages"], dtype=torch.float32, device=self.device
        )
        returns = torch.tensor(data["returns"], dtype=torch.float32, device=self.device)
        action_masks = torch.tensor(
            data["action_masks"], dtype=torch.bool, device=self.device
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training metrics
        metrics = {
            "value_loss": 0,
            "policy_loss": 0,
            "entropy": 0,
            "kl": 0,
            "clip_fraction": 0,
            "approx_kl": 0,
            "total_loss": 0,
        }

        # Perform multiple epochs of training
        for epoch in range(n_epochs):
            # Generate random indices
            indices = np.random.permutation(len(text_obs))

            # Create mini-batches
            n_batches = len(indices) // batch_size
            for start in range(0, len(indices), batch_size):
                # Get batch indices
                batch_indices = indices[start : start + batch_size]

                # Get batch data
                batch_text_obs = [text_obs[i] for i in batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_action_masks = action_masks[batch_indices]

                # Evaluate the batch with the current model
                new_log_probs, entropy, values = (
                    self.model.module.evaluate_actions(
                        batch_text_obs, batch_actions, batch_action_masks
                    )
                    if isinstance(self.model, DDP)
                    else self.model.evaluate_actions(
                        batch_text_obs, batch_actions, batch_action_masks
                    )
                )

                # Calculate the ratio between new and old action probabilities
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Compute the policy loss using clipped objective
                policy_loss1 = -batch_advantages * ratio
                policy_loss2 = -batch_advantages * torch.clamp(
                    ratio, 1 - self.clip_eps, 1 + self.clip_eps
                )
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                # Compute the value loss
                value_loss = F.mse_loss(values.squeeze(-1), batch_returns)

                # Compute the entropy bonus
                entropy_loss = -entropy.mean()

                # Compute the total loss
                total_loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                )

                # Compute the approximate KL divergence
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()

                # Break early if KL divergence is too high
                if self.target_kl is not None and approx_kl > self.target_kl:
                    break

                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                # Clip gradients
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update metrics
                metrics["value_loss"] += value_loss.item()
                metrics["policy_loss"] += policy_loss.item()
                metrics["entropy"] += entropy.mean().item()
                metrics["approx_kl"] += approx_kl
                metrics["clip_fraction"] += (
                    ((ratio - 1).abs() > self.clip_eps).float().mean().item()
                )
                metrics["total_loss"] += total_loss.item()

        # Average the metrics
        for k in metrics:
            metrics[k] /= n_epochs * n_batches

        return metrics
