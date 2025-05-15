import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, DynamicCache
from typing import Dict, List, Tuple, Optional, Union
from peft import LoraConfig, TaskType, get_peft_model
import re


class QwenActorCritic(nn.Module):
    """Actor-Critic model based on Qwen3 language model."""

    def __init__(
        self,
        model_path: str,
        n_agents: int,
        n_actions: int,
        use_lora: bool = False,
        lora_r: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        device_map: str = "balanced",
        out_device: str = None,
    ):
        """Initialize the Qwen-based Actor-Critic model."""
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        # self.qwen_model = AutoModelForCausalLM.from_pretrained(model_path)
        self.use_lora = use_lora

        # Apply LoRA to the model if specified
        # Use AutoModel instead of AutoModelForCausalLM to avoid using LM head
        if use_lora:
            print("using lora")
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                ],
                modules_to_save=[],
                lora_dropout=lora_dropout,
                bias="none",
                peft_type="LORA",
                inference_mode=False,
                task_type=TaskType.FEATURE_EXTRACTION,
                init_lora_weights="gaussian",
            )

            self.qwen_model = get_peft_model(
                AutoModel.from_pretrained(
                    model_path,
                    device_map=device_map,
                    torch_dtype=torch.bfloat16,
                ),
                lora_config,
            )
            self.qwen_model.config.use_cache = False

            print(f"LoRA config: {self.qwen_model.peft_config}")

            self.embedding_device = (
                self.qwen_model.base_model.model.embed_tokens.weight.device
            )
            self.last_layer_device = next(
                self.qwen_model.base_model.model.norm.parameters()
            ).device
        else:
            print("using full finetune")
            self.qwen_model = AutoModel.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
            )
            self.qwen_model.config.use_cache = False

            self.embedding_device = self.qwen_model.embed_tokens.weight.device
            self.last_layer_device = next(self.qwen_model.norm.parameters()).device
        print(f"Model config use_cache: {self.qwen_model.config.use_cache}")

        if out_device is not None:
            self.out_device = out_device
        else:
            self.out_device = self.last_layer_device
        if device_map == "balanced_low_0":
            self.out_device = "cuda:0"

        print(f"Embedding device: {self.embedding_device}")
        print(f"Last layer device: {self.last_layer_device}")
        print(f"Output device: {self.out_device}")

        self.layer_devices = {}
        for name, param in self.qwen_model.named_parameters():
            layer_match = re.match(r".*layers\.(\d+)\..*", name)
            if layer_match:
                layer_idx = int(layer_match.group(1))
                self.layer_devices[layer_idx] = param.device
        print(f"Layer devices:")
        for layer_idx, device in self.layer_devices.items():
            print(f"Layer {layer_idx}: {device}")

        self.static_kv_cache = None

        # Get the embedding dimension from the model
        self.hidden_size = self.qwen_model.config.hidden_size
        # Create action head (policy network)
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, n_actions),
        ).to(self.out_device, dtype=torch.bfloat16)

        # Create value head (value network)
        self.value_head = nn.Linear(self.hidden_size, 1).to(
            self.out_device, dtype=torch.bfloat16
        )

    def _get_qwen_output(
        self,
        text_obs: Union[str, List[str]],
        try_to_use_static_kv_cache: Optional[bool] = True,
        past_key_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process the text observation through the Qwen model."""
        # Tokenize the input text
        inputs = self.tokenizer(
            text_obs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,  # Adjust based on observation length
        ).to(self.embedding_device)
        batch_size, seq_len = inputs["input_ids"].shape

        attention_mask = inputs["attention_mask"].to(self.out_device)

        if past_key_values is not None or (
            try_to_use_static_kv_cache and self.static_kv_cache is not None
        ):
            past_key_values = past_key_values or self.static_kv_cache

            static_seq_len = past_key_values.get_seq_length()

            expanded_cache = self.expand_cache_for_batch(past_key_values, batch_size)

            position_ids = (
                torch.arange(static_seq_len, static_seq_len + seq_len)
                .unsqueeze(0)
                .expand(batch_size, -1)
                .to(self.embedding_device)
            )

            static_attention_mask = (
                attention_mask[..., -1]
                .unsqueeze(-1)
                .expand(*attention_mask.shape[:-1], static_seq_len)
            )

            inputs["attention_mask"] = (
                torch.concat([static_attention_mask, attention_mask], dim=-1)
                .to(torch.bfloat16)
                .to(self.embedding_device)
            )
            inputs["position_ids"] = position_ids
            inputs["past_key_values"] = expanded_cache

            hidden_states = (
                self.qwen_model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                )
                .hidden_states[-1]
                .to(self.out_device)
            )
            attention_mask = attention_mask.to(self.out_device)
        else:
            hidden_states = (
                self.qwen_model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                )
                .hidden_states[-1]
                .to(self.out_device)
            )

        # Extract the last hidden state for the last token
        # last_token_hidden_states = hidden_states[:, -1, :]
        last_token_hidden_states = self.mean_pooling(hidden_states, attention_mask)

        return last_token_hidden_states

    def mean_pooling(self, hidden_states, attention_mask=None):

        # Convert to bfloat16 for consistency
        hidden_states = hidden_states.to(torch.bfloat16)
        # Expand attention mask to hidden_states dimensions
        expanded_mask = (
            attention_mask.unsqueeze(-1).expand(hidden_states.size()).to(torch.bfloat16)
        )
        # Apply mask and calculate average
        sum_hidden = torch.sum(hidden_states * expanded_mask, dim=1)
        sum_mask = torch.sum(expanded_mask, dim=1)
        # Avoid division by zero
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_hidden / sum_mask

    def forward(
        self,
        text_obs: Union[str, List[str]],
        action_masks: Optional[torch.Tensor] = None,
        try_to_use_static_kv_cache: Optional[bool] = True,
        past_key_values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model."""
        # Get the hidden states from the Qwen model
        # print(f"Prompt: {text_obs}")
        # try:
        if isinstance(text_obs, str):
            hidden_states = self._get_qwen_output(
                text_obs=[
                    text_obs + f"\nGenerate most proper action for agent {i}:"
                    for i in range(self.n_agents)
                ],
                try_to_use_static_kv_cache=try_to_use_static_kv_cache,
                past_key_values=past_key_values,
            )
        elif isinstance(text_obs, list):
            hidden_states = self._get_qwen_output(
                text_obs=[
                    to + f"\nGenerate most proper action for agent {i}:"
                    for i in range(self.n_agents)
                    for to in text_obs
                ],
                try_to_use_static_kv_cache=try_to_use_static_kv_cache,
                past_key_values=past_key_values,
            ).view(len(text_obs), self.n_agents, self.hidden_size)
        # except TypeError as e:
        #     print(f"Error: {e}")
        #     print(f"past_key_values: {type(past_key_values)} {past_key_values}")
        #     print(f"text_obs: {type(text_obs)} {text_obs}")
        #     exit()

        # Compute action logits
        action_logits = self.action_head(hidden_states)

        # Apply action masks if provided
        if action_masks is not None:
            # Set the logits of invalid actions to a large negative value
            action_logits = action_logits.masked_fill(action_masks == 0, -1e9)

        # Compute value estimate
        value = self.value_head(hidden_states)
        # In PPO, value should be global
        # apply mean pooling across all agents
        value = torch.mean(value, dim=-2)

        return action_logits, value

    def get_action_and_value(
        self,
        text_obs: Union[str, List[str]],
        action_masks: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        try_to_use_static_kv_cache: Optional[bool] = True,
        past_key_values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action and value for the given observation."""
        # Forward pass through the model
        action_logits, value = self.forward(
            text_obs=text_obs,
            action_masks=action_masks,
            try_to_use_static_kv_cache=try_to_use_static_kv_cache,
            past_key_values=past_key_values,
        )

        # Compute the action distribution
        action_probs = F.softmax(action_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)

        # Sample actions from the distribution
        if deterministic:
            action = torch.argmax(action_probs, dim=-1).to(torch.bfloat16)
        else:
            action = action_dist.sample().to(torch.bfloat16)

        # Compute the log probability of the actions
        action_log_prob = action_dist.log_prob(action)
        # In PPO, sum of products of log probability and advantage
        # is equivalent to products of advantage sum of log probability
        action_log_prob = torch.sum(action_log_prob.unsqueeze(-2), dim=-1)

        # Compute the entropy of the action distribution
        entropy = action_dist.entropy()
        # In PPO, entropy should be global
        # apply mean pooling across all agents
        # divide by square root of number of agents
        entropy = torch.sum(entropy.unsqueeze(-2), dim=-1) / torch.sqrt(
            torch.tensor(self.n_agents, dtype=entropy.dtype, device=entropy.device)
        )

        return action, action_log_prob, entropy, value

    def evaluate_actions(
        self,
        text_obs: Union[str, List[str]],
        actions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        try_to_use_static_kv_cache: Optional[bool] = True,
        past_key_values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate the log probability, entropy, and value of actions."""
        # Forward pass through the model
        action_logits, value = self.forward(
            text_obs=text_obs,
            action_masks=action_masks,
            try_to_use_static_kv_cache=try_to_use_static_kv_cache,
            past_key_values=past_key_values,
        )

        # Compute the action distribution
        action_probs = F.softmax(action_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)

        # Compute the log probability of the actions
        action_log_prob = action_dist.log_prob(actions)
        action_log_prob = torch.sum(action_log_prob.unsqueeze(-2), dim=-1)

        # Compute the entropy of the action distribution
        entropy = action_dist.entropy()
        entropy = torch.sum(entropy.unsqueeze(-2), dim=-1) / torch.sqrt(
            torch.tensor(self.n_agents, dtype=entropy.dtype, device=entropy.device)
        )

        return action_log_prob, entropy, value

    def save_lora_weights(self, path: str):
        """Save the LoRA adapter weights."""
        self.qwen_model.save_pretrained(path)

    def load_lora_weights(self, path: str):
        """Load the LoRA adapter weights."""
        self.qwen_model.load_adapter(path, adapter_name="default")

    def get_static_kv_cache(self, static_text: str, sync: bool = True) -> DynamicCache:
        """获取静态KV缓存"""
        static_inputs = self.tokenizer(
            static_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,  # Adjust based on observation length
        ).to(self.embedding_device)

        static_hidden_states = self.qwen_model(
            **static_inputs,
            output_hidden_states=True,
            # Note: Need to set use_cache to True to get past_key_values
            use_cache=True,
        )

        past_key_values = static_hidden_states.past_key_values

        for layer_idx in range(len(past_key_values)):
            target_device = self.layer_devices.get(layer_idx)
            past_key_values.key_cache[layer_idx] = (
                past_key_values.key_cache[layer_idx]
                .to(target_device)
                .to(torch.bfloat16)
            )
            past_key_values.value_cache[layer_idx] = (
                past_key_values.value_cache[layer_idx]
                .to(target_device)
                .to(torch.bfloat16)
            )

        if sync:
            self.static_kv_cache = past_key_values

        return past_key_values

    def expand_cache_for_batch(
        self, cache: DynamicCache, batch_size: int
    ) -> DynamicCache:
        """扩展单样本缓存到批量大小"""
        if batch_size == 1:
            return cache  # 如果批量大小为1，无需扩展

        expanded_cache = DynamicCache()
        for key_layer_cache, value_layer_cache in cache:
            # key_cache: [1, num_heads, seq_len, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
            expanded_key = key_layer_cache.expand(
                batch_size, *key_layer_cache.shape[1:]
            ).to(key_layer_cache.device)

            # value_cache: [1, num_heads, seq_len, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
            expanded_value = value_layer_cache.expand(
                batch_size, *value_layer_cache.shape[1:]
            ).to(value_layer_cache.device)

            # 更新扩展后的缓存
            expanded_cache.key_cache.append(expanded_key.to(torch.bfloat16))
            expanded_cache.value_cache.append(expanded_value.to(torch.bfloat16))

        return expanded_cache


if __name__ == "__main__":
    # Example usage
    model = QwenActorCritic(
        model_path="Qwen/Qwen3-0.6B",
        n_agents=5,
        n_actions=11,
        use_lora=False,
        device_map={
            "embed_tokens": "cuda:0",
            "layers.0": "cuda:0",
            "layers.1": "cuda:0",
            "layers.2": "cuda:0",
            "layers.3": "cuda:0",
            "layers.4": "cuda:0",
            "layers.5": "cuda:0",
            "layers.6": "cuda:0",
            "layers.7": "cuda:0",
            "layers.8": "cuda:0",
            "layers.9": "cuda:0",
            "layers.10": "cuda:0",
            "layers.11": "cuda:0",
            "layers.12": "cuda:0",
            "layers.13": "cuda:0",
            "layers.14": "cuda:0",
            "layers.15": "cuda:1",
            "layers.16": "cuda:1",
            "layers.17": "cuda:1",
            "layers.18": "cuda:1",
            "layers.19": "cuda:1",
            "layers.20": "cuda:1",
            "layers.21": "cuda:1",
            "layers.22": "cuda:1",
            "layers.23": "cuda:1",
            "layers.24": "cuda:1",
            "layers.25": "cuda:1",
            "layers.26": "cuda:1",
            "layers.27": "cuda:1",
            "norm": "cuda:1",
            "rotary_emb": "cuda:1",
        },
    )
    static_text = """Map Config: The map is 2s3z of 32*32 sized square map.
The available area of x axis is from 0 to 32, and the y axis is from 7 to 25.
The enemy units are at (23, 16) point and your units are at (9, 16) point initially.
The ememy controls all the units to move and attack (9, 16) point along the way.
There is no terrain advantages nor choke points in this map.
You can control 2 Stalker units and 3 Zealot units.
The enemy controls 2 Stalker units and 3 Zealot units.

Unit Config: 
    The information of the units are:
    {'Unit': 'zealot', 'Attack': {'Targets': 'Ground', 'Damage': '8 (+1) (x2)', 'DPS': '18.6 (+2.33)', 'Cooldown': '0.86', 'Bonus': 'Not found', 'Bonus DPS': 'Not found', 'Range': '0.1'}, 'Unit stats': {'Defense': '100 50 1 (+1)', 'Attributes': 'Light, Biological', 'Sight': '9', 'Speed': '3.154.725 (+5.67 with Charge)', 'Cargo size': '2'}}
{'Unit': 'stalker', 'Attack': {'Targets': 'Ground / Air', 'Damage': '13 (+1)', 'DPS': '9.7 (+0.75)', 'Cooldown': '1.34', 'Bonus': '+5 (+1) vs Armored', 'Bonus DPS': '+3.7 (+0.75) vs Armored', 'Range': '6'}, 'Unit stats': {'Defense': '80 80 1 (+1)', 'Attributes': 'Armored, Mechanical', 'Sight': '10', 'Speed': '4.13', 'Cargo size': '2'}}
All the units has no abilities such as blinking or equipments."""

    past_key_values = model.get_static_kv_cache(static_text)

    dynamic_text = """Initial Observation: Current timestep: 0    
Ally units:
Agent 0 stalker is at (9.00, 14.81) with 80/80 Health and 80/80 Shield
Agent 1 stalker is at (9.00, 16.00) with 80/80 Health and 80/80 Shield
Agent 2 zealot is at (8.25, 16.75) with 100/100 Health and 50/50 Shield
Agent 3 zealot is at (9.75, 16.75) with 100/100 Health and 50/50 Shield
Agent 4 zealot is at (9.94, 15.51) with 100/100 Health and 50/50 Shield
No visible enemy units."""

    print("=" * 20 + "Get Action and Value" + "=" * 20)

    action_masks = torch.ones((5, 11), device=model.out_device)  # Example action mask

    # time the time taken for the forward pass
    import time

    start = time.time()
    _action, _log_prob, _entropy, _value = model.get_action_and_value(
        text_obs=static_text + dynamic_text,
        action_masks=action_masks,
        deterministic=False,
        try_to_use_static_kv_cache=False,
        past_key_values=None,
    )
    end = time.time()
    print(f"Time taken for Full forward pass: {end - start:.4f} seconds")

    start = time.time()
    action, log_prob, entropy, value = model.get_action_and_value(
        text_obs=dynamic_text,
        action_masks=action_masks,
        deterministic=False,
        # past_key_values=past_key_values,
    )
    end = time.time()
    print(f"Time taken for Dyn forward pass: {end - start:.4f} seconds")

    print("Action:", action.shape)
    print("Log Probability:", log_prob.shape)
    print("Entropy:", entropy.shape)
    print("Value:", value.shape)

    print("=" * 20 + "Evaluate Actions" + "=" * 20)
    batch_size = 2
    # expand to match batch size
    text_obs = [dynamic_text] * batch_size
    actions = action.unsqueeze(0).expand(batch_size, -1)
    action_masks = action_masks.unsqueeze(0).expand(batch_size, -1, -1)

    full_text_obs = [static_text + dynamic_text] * batch_size
    start = time.time()
    log_prob, entropy, value = model.evaluate_actions(
        text_obs=full_text_obs,
        actions=actions,
        action_masks=action_masks,
        try_to_use_static_kv_cache=False,
        past_key_values=None,
    )
    end = time.time()
    print(f"Time taken for Full forward pass: {end - start:.4f} seconds")

    start = time.time()
    log_prob, entropy, value = model.evaluate_actions(
        text_obs=text_obs,
        actions=actions,
        action_masks=action_masks,
        # past_key_values=past_key_values,
    )
    end = time.time()
    print(f"Time taken for Dyn forward pass: {end - start:.4f} seconds")

    print("Actions:", actions.shape)
    print("Log Probability:", log_prob.shape)
    print("Entropy:", entropy.shape)
    print("Value:", value.shape)
