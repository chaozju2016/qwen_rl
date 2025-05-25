# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
import flashbax as fbx
from flashbax.vault import Vault
import torch
import tqdm
import numpy as np
import pickle
import json

from smac_env_wrapper import SMACTextWrapper
import config


# %%
rel_dir = "/mnt/HDD/wangchao/smac_v1"
vlt_name = input()
vault_name_list = [vlt_name]
vault_uid_list = ["Good", "Medium", "Poor"]
pkl_path = "/mnt/HDD/wangchao/smac_v1_origin/"
json_path = "/mnt/HDD/wangchao/smac_v1_json/"

# %%
instruct_prompt = "Given these information, please give integer-indexed action for each agent, for example, if you control 3 agents that should perform action 5, action 6 and action 2, response 5,6,2\n\n"

output_format = {
    "LongCat": {
        "system": ["", ""],
        "user": ["[Round 0] USER:", ""],
        "assistant": ["", instruct_prompt + " ASSISTANT:"],
        "target": ["", ""],
    },
    "Qwen": {
        "system": ["<|im_start|>system:\n", "<|im_end|>\n\n"],
        "user": ["<|im_start|>user:\n", "<|im_end|>\n\n"],
        "assistant": ["", instruct_prompt + "<|im_start|>assistant:\n"],
        "target": ["", "<|im_end|>\n\n"],
    },
}


# %%
output_type = "LongCat"
prompt_format = output_format[output_type]


# %%
for vault_name in vault_name_list:
    for vault_uid in vault_uid_list:
        pkl_data = pickle.load(
            open(os.path.join(pkl_path, f"{vault_name}_{vault_uid}.pkl"), "rb")
        )
        num_episodes = len(pkl_data["actions"])
        json_file = os.path.join(
            json_path, f"{vault_name}_{vault_uid}_{num_episodes}.json"
        )
        fd = open(json_file, "a", encoding="utf-8")
        env = SMACTextWrapper(map_name=vault_name, lazy=False)
        static_prompt = (
            config.SYSTEM_PROMPT + "\n" + env.map_config + "\n" + env.unit_config + "\n"
        )
        print(
            f"vault_name: {vault_name}, vault_uid: {vault_uid}, num_episodes: {num_episodes}"
        )
        for episode in tqdm.tqdm(
            range(num_episodes), position=0, desc=f"{vault_name}_{vault_uid}"
        ):
            # TODO: select episode with reward

            # process episode
            len_episode = len(pkl_data["actions"][episode])
            for step in tqdm.tqdm(range(len_episode), position=1, leave=False):
                text_obs = env._obs_to_text(
                    obs=pkl_data["observations"][episode][step],
                    state=pkl_data["state"][episode][step],
                    avail_actions=pkl_data["legals"][episode][step],
                    step=step,
                )
                text_act = env._act_to_text(
                    act=np.asarray(pkl_data["actions"][episode][step], dtype=str)
                )

                prompt = {
                    "input": (static_prompt + text_obs)
                    .join(prompt_format["user"])
                    .join(prompt_format["assistant"]),
                    "target": text_act,
                }

                json.dump(prompt, fd, ensure_ascii=False)
                fd.write("\n")

                # break
            # break
        fd.close()
        print(f"Finish updating {json_file}")
        # break
    # break
