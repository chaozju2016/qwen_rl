import json
from configs.map_config import MapConfig

# --- PPO & Model Configurations ---
QWEN_MODEL_PATH = "Qwen/Qwen3-0.6B"
SMAC_MAP_NAME = "8m"  # Example map, user can change this

# Device configuration for Qwen model
DEVICE_MAP = {
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
    "layers.14": "cuda:1",
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
}

# PPO Hyperparameters
LEARNING_RATE = 2.5e-6  # Typical for LLM fine-tuning
PPO_EPOCHS = 1
BATCH_SIZE = 1  # Number of (obs, action, ...) tuples per PPO update
ROLLOUT_LENGTH = 32  # Number of steps to collect per environment before an update
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95  # Lambda for GAE
CLIP_EPS = 0.2  # PPO clipping parameter
ENT_COEF = 0.01  # Entropy coefficient
VF_COEF = 0.5  # Value function coefficient
MAX_GRAD_NORM = 0.5  # Max gradient norm for clipping
TARGET_KL = None  # Target KL divergence for early stopping

# Training settings
TOTAL_TIMESTEPS = 1_000_000  # Total timesteps for training
SEED = 42

SYSTEM_PROMPT = """
You are a feature extractor designed for reinforcement learning environments. 
Your task is to analyze battlefield information in the StarCraft Multi-Agent Challenge (SMAC) and generate high-quality state representations. 
These representations will be further used to predict the best actions for the agents.

**Relevant domain knowledge**

The relative position and distance of units are crucial to tactics
Health status affects the combat capability and vulnerability of units
Teamwork requires considering the combined state of all agents

**Your responsibilities**

Understand and analyze the state of the environment, unit positions, health, and available actions
Process the battlefield situation and identify threats and opportunities
Extract strategically relevant features and encode this information in the internal representation
Instead of directly outputting action decisions, provide informative state representations

**How it works**

The input you receive includes a description of the environment and the states of each agent
You need to generate a deep understanding of the battlefield situation internally
Your hidden state representation will be passed to a specialized action prediction network
Focus on extracting the most relevant strategic information and features

"""


def process_info(unit_name, base_dir="knowledge_data/firecrawl_test/sc2_unit_info/"):

    with open("{}{}.json".format(base_dir, unit_name), "r") as reader:
        info_json = json.load(reader)

    info_needed = {}
    info_needed["Unit"] = unit_name
    # info_needed['Type'] = info_json['Type']
    # info_needed['Description'] = info_json['Description']
    info_needed["Attack"] = info_json["Attack"]
    info_needed["Unit stats"] = info_json["Unit stats"]
    # info_needed['Strong against'] = info_json['Strong against']
    # info_needed['Weak against'] = info_json['Weak against']
    # info_needed["Competitive Usage"] = info_json["Competitive Usage"]

    return str(info_needed)


def get_map_config(map_name):
    mc = MapConfig().get_map_config(map_name)

    map_config = mc["map_info"]
    units = mc["units_info"]

    return map_config, units


def get_units_config(units):
    units_info = ""
    for a in set(units):
        units_info += process_info(a) + "\n"

    unit_config = """
    The information of the units are:
    {}All the units has no abilities such as blinking or equipments.
    """.format(
        units_info
    )

    return unit_config


if __name__ == "__main__":
    map_name = "8m"

    map_config, units = get_map_config(map_name)
    unit_config = get_units_config(units)

    print("Map config: ", map_config)
    print("Unit: ", units)
    print("Unit config: ", unit_config)

    # Example of accessing new configs
    print("\n--- PPO Configurations ---")
    print(f"Qwen Model Path: {QWEN_MODEL_PATH}")
    print(f"SMAC Map Name: {SMAC_MAP_NAME}")
    print(f"Learning Rate: {LEARNING_RATE}")
