import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional, Union
from peft import LoraConfig, TaskType, get_peft_model


class QwenActorCritic(nn.Module):
    """Actor-Critic model based on Qwen3 language model."""

    def __init__(
        self,
        model_path: str,
        n_agents: int,
        n_actions: int,
        device: str = "cuda",
        use_lora: bool = False,
        lora_r: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        """Initialize the Qwen-based Actor-Critic model."""
        super().__init__()

        # Load the Qwen model and tokenizer
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # self.qwen_model = AutoModelForCausalLM.from_pretrained(model_path)
        self.use_lora = use_lora

        # Apply LoRA to the model if specified
        if use_lora:
            print("using lora")
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj"],
                modules_to_save=[],
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.qwen_model = get_peft_model(
                AutoModelForCausalLM.from_pretrained(model_path), lora_config
            )
        else:
            print("using full finetune")
            self.qwen_model = AutoModelForCausalLM.from_pretrained(model_path)

        # Get the embedding dimension from the model
        self.hidden_size = self.qwen_model.config.hidden_size

        # Create action head (policy network)
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.action_head = nn.Linear(self.hidden_size, n_actions)

        # Create value head (value network)
        self.value_head = nn.Linear(self.hidden_size, 1)

        # Move model to the specified device
        self.to(device)

    def _get_qwen_output(self, text_obs: Union[str, List[str]]) -> torch.Tensor:
        """Process the text observation through the Qwen model."""
        # Tokenize the input text
        inputs = self.tokenizer(
            text_obs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,  # Adjust based on observation length
        ).to(self.device)

        # Get the output from the Qwen model
        outputs = self.qwen_model(**inputs, output_hidden_states=True)

        # Extract the last hidden state for the last token
        last_token_hidden_states = outputs.hidden_states[-1][:, -1, :]

        return last_token_hidden_states

    def forward(
        self,
        text_obs: Union[str, List[str]],
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model."""
        # Get the hidden states from the Qwen model
        # print(f"Prompt: {text_obs}")
        try:
            if isinstance(text_obs, str):
                hidden_states = self._get_qwen_output(
                    [
                        text_obs + f"\nGenerate most proper action for agent {i}:"
                        for i in range(self.n_agents)
                    ]
                )
            elif isinstance(text_obs, list):
                hidden_states = self._get_qwen_output(
                    [
                        to + f"\nGenerate most proper action for agent {i}:"
                        for i in range(self.n_agents)
                        for to in text_obs
                    ]
                ).view(len(text_obs), self.n_agents, self.hidden_size)
        except TypeError:
            print(f"text_obs: {type(text_obs)} {text_obs}")
            exit()
        # print(f"hidden_states: {hidden_states.shape}")

        # Compute action logits
        action_logits = self.action_head(hidden_states)

        # Apply action masks if provided
        if action_masks is not None:
            # Set the logits of invalid actions to a large negative value
            # print(f"action_logits: {action_logits.shape}")
            # print(f"action_masks: {action_masks.shape}")
            action_logits = action_logits.masked_fill(action_masks == 0, -1e9)

        # exit()
        # Compute value estimate
        value = self.value_head(hidden_states)

        return action_logits, value

    def get_action_and_value(
        self,
        text_obs: Union[str, List[str]],
        action_masks: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action and value for the given observation."""
        # Forward pass through the model
        action_logits, value = self.forward(text_obs, action_masks)

        # Compute the action distribution
        action_probs = F.softmax(action_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)

        # Sample actions from the distribution
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            action = action_dist.sample()

        # Compute the log probability of the actions
        action_log_prob = action_dist.log_prob(action)

        # Compute the entropy of the action distribution
        entropy = action_dist.entropy()

        return action, action_log_prob, entropy, value

    def evaluate_actions(
        self,
        text_obs: Union[str, List[str]],
        actions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate the log probability, entropy, and value of actions."""
        # Forward pass through the model
        action_logits, value = self.forward(text_obs, action_masks)

        # Compute the action distribution
        action_probs = F.softmax(action_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)

        # Compute the log probability of the actions
        action_log_prob = action_dist.log_prob(actions)

        # Compute the entropy of the action distribution
        entropy = action_dist.entropy()

        return action_log_prob, entropy, value

    def save_lora_weights(self, path: str):
        """Save the LoRA adapter weights."""
        self.qwen_model.save_pretrained(path)

    def load_lora_weights(self, path: str):
        """Load the LoRA adapter weights."""
        self.qwen_model.load_adapter(path, adapter_name="default")


if __name__ == "__main__":
    # Example usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = QwenActorCritic(model_path="Qwen/Qwen3-0.6B", n_actions=11, device=device)
    text_obs = """Map Config: The map is 2s3z of 32*32 sized square map.
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
All the units has no abilities such as blinking or equipments.
    
Initial Observation: Current timestep: 0
Ally units:
Agent 0 stalker is at (9.00, 14.81) with 80/80 Health and 80/80 Shield
Agent 1 stalker is at (9.00, 16.00) with 80/80 Health and 80/80 Shield
Agent 2 zealot is at (8.25, 16.75) with 100/100 Health and 50/50 Shield
Agent 3 zealot is at (9.75, 16.75) with 100/100 Health and 50/50 Shield
Agent 4 zealot is at (9.94, 15.51) with 100/100 Health and 50/50 Shield
No visible enemy units."""
    action_masks = torch.ones((5, 11), device=device)  # Example action mask
    action, log_prob, entropy, value = model.get_action_and_value(
        text_obs, action_masks
    )
    print("Action:", action)
    print("Log Probability:", log_prob)
    print("Entropy:", entropy)
    print("Value:", value)
