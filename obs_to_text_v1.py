import os
import pickle
import yaml
import torch
import json
import tqdm

from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
from smacv2.env.starcraft2.starcraft2 import StarCraft2Env

system_prompt = {
    "introduction": (
        "SMACv2 (StarCraft Multi-Agent Challenge v2) 是一个实时战略游戏环境，"
        "你将作为中央控制器同时指挥多个友方单位与敌方作战。"
        "\n"
    ),
    "map_info": {
        "3m": "友方有3个单位，敌方有3个单位\n",
        "8m": "友方有8个单位，敌方有8个单位\n",
        "5m_vs_6m": "友方有5个单位，敌方有6个单位\n",
    },
    "race": {
        "terran": (
            "敌我均会包含下列单位："
            "Marine - 基础步兵单位，血量45，护甲0，攻击力6，射程5。"
            "\n"
        ),
    },
    "coordination": (
        "所有地图的标准尺寸为32x32游戏单位，在相对坐标系统中对应[-1,1]x[-1,1]的范围"
        "所有单位位置使用相对坐标系统："
        "以地图中心为原点(0,0)，坐标值范围[-1,1]。"
        "格式为(rel_x, rel_y)，其中rel_x正值=右侧/负值=左侧，rel_y正值=上方/负值=下方。"
        "例如Marine位于(0.15, -0.22)表示其在地图中心右侧0.15、下方0.22处。"
        "\n"
    ),
    "task": (
        "你是一个多智能体强化学习环境中的中央控制器，需要同时控制所有友方单位进行实时战略作战。"
        "主要目标：消灭所有敌方单位，同时尽可能保护友方单位存活。\n"
        "输出格式：为每个友方单位选择一个最合适的有效动作，格式为字符串 'action_0,action_1, ...'"
        "例如，如果你控制3个友方单位，分别选择动作5、6和2，"
        "则输出 '5,6,2'。"
        "如果某个友方单位已死亡，则只能为其选择动作0。\n"
        "\n"
    ),
}

instruct_prompt = (
    "根据以上信息，为每个友方单位选择一个最合适的有效动作，"
    "格式为字符串 'action_0,action_1, ...'。"
    "如果某个友方单位已死亡，则只能为其选择动作0。\n\n"
)


def build_ally_prompt(
    unit_name,
    health_in_percent,
    rel_x,
    rel_y,
    energy_or_cooldown,
    sheild=None,
    is_cooldown_in_sec=True,
    avail_actions_mask=None,
):
    assert (
        avail_actions_mask is not None and avail_actions_mask.dim() == 1
    ), "avail_actions_mask should be a 1D tensor"

    if health_in_percent <= 0:
        return f"{unit_name}已死亡，可用动作为[0]。\n"

    horizontal = "右侧" if rel_x > 0 else "左侧" if rel_x < 0 else "中央"
    vertical = "上方" if rel_y > 0 else "下方" if rel_y < 0 else "中央"

    prompt = (
        f"{unit_name}当前拥有{health_in_percent:.0f}%的生命值{f'和{sheild*100:.0f}%的护盾' if sheild else ''}，"
        f"位于地图{horizontal}{abs(rel_x):.2f}, {vertical}{abs(rel_y):.2f}位置，"
        f"其{'技能冷却时间'if is_cooldown_in_sec else '剩余能量'}为{energy_or_cooldown:.1f}{'秒'if is_cooldown_in_sec else '点'}，"
        f"可用动作为{torch.where(avail_actions_mask)[0].tolist()}。"
        "\n"
    )
    return prompt


def build_enemy_prompt(
    unit_name,
    health_in_percent,
    rel_x,
    rel_y,
    sheild=None,
):
    if health_in_percent <= 0:
        return f"{unit_name}已死亡，可用动作为[0]。\n"

    horizontal = "右侧" if rel_x > 0 else "左侧" if rel_x < 0 else "中央"
    vertical = "上方" if rel_y > 0 else "下方" if rel_y < 0 else "中央"

    prompt = (
        f"{unit_name}当前拥有{health_in_percent:.0f}%的生命值{f'和{sheild*100:.0f}%的护盾' if sheild else ''}，"
        f"位于地图{horizontal}{abs(rel_x):.2f}, {vertical}{abs(rel_y):.2f}位置。"
        "\n"
    )
    return prompt


def get_prompt_step(
    state,
    avail_actions_masks,
    env_wrapper,
    n_agents,
    n_enemies,
    ally_num_attributes,
    enemy_num_attributes,
    max_x,
    max_y,
    unit_type_bits,
    unit_names,
    game_sight_range_map,
    max_cd,
    ally_has_shield=False,
    enemy_has_shield=False,
):

    agent_abs_positions = []
    game_sight_range_map_agent = {}

    ally_prompt = ""
    for agent_id in range(n_agents):
        shield = None
        if ally_has_shield:
            health, energy_or_cooldown, rel_x, rel_y, shield = state[
                agent_id * ally_num_attributes : (agent_id + 1) * ally_num_attributes
                - unit_type_bits
            ]
        else:
            health, energy_or_cooldown, rel_x, rel_y = state[
                agent_id * ally_num_attributes : (agent_id + 1) * ally_num_attributes
                - unit_type_bits
            ]

        unit_type = unit_names[
            (
                torch.argmax(
                    state[
                        (agent_id + 1) * ally_num_attributes
                        - unit_type_bits : (agent_id + 1) * ally_num_attributes
                    ]
                ).item()
                if unit_type_bits > 0
                else 0
            )
        ]
        game_sight_range_map_agent[agent_id] = game_sight_range_map.get(unit_type, 9)
        agent_abs_positions.append((rel_x * max_x / 2, rel_y * max_y / 2))
        is_cooldown_in_sec = unit_type != "medivac"
        energy_or_cooldown = energy_or_cooldown * max_cd[unit_type]
        ally_prompt += build_ally_prompt(
            unit_name=f"友方{unit_type}-{agent_id}",
            health_in_percent=health * 100,
            rel_x=rel_x,
            rel_y=rel_y,
            energy_or_cooldown=energy_or_cooldown,
            sheild=shield,
            is_cooldown_in_sec=is_cooldown_in_sec,
            avail_actions_mask=avail_actions_masks[agent_id],
        )

    enemy_prompt = ""

    for enemy_id in range(n_enemies):
        shield = None
        if enemy_has_shield:
            health, rel_x, rel_y, shield = state[
                n_agents * ally_num_attributes
                + enemy_id * enemy_num_attributes : n_agents * ally_num_attributes
                + (enemy_id + 1) * enemy_num_attributes
                - unit_type_bits
            ]
        else:
            health, rel_x, rel_y = state[
                n_agents * ally_num_attributes
                + enemy_id * enemy_num_attributes : n_agents * ally_num_attributes
                + (enemy_id + 1) * enemy_num_attributes
                - unit_type_bits
            ]
        is_visible = False
        for agent_id in range(n_agents):
            sight_ragne = game_sight_range_map_agent[agent_id]
            distance = env_wrapper.env.distance(
                x1=agent_abs_positions[agent_id][0],
                y1=agent_abs_positions[agent_id][1],
                x2=rel_x * max_x / 2,
                y2=rel_y * max_y / 2,
            )
            if distance <= sight_ragne:
                is_visible = True
                break
        if not is_visible:
            continue

        unit_type = unit_names[
            (
                torch.argmax(
                    state[
                        n_agents * ally_num_attributes
                        + (enemy_id + 1) * enemy_num_attributes
                        - unit_type_bits : n_agents * ally_num_attributes
                        + (enemy_id + 1) * enemy_num_attributes
                    ]
                ).item()
                if unit_type_bits > 0
                else 0
            )
        ]
        enemy_prompt += build_enemy_prompt(
            unit_name=f"敌方{unit_type}-{enemy_id}",
            health_in_percent=health * 100,
            rel_x=rel_x,
            rel_y=rel_y,
            sheild=shield,
        )

    return ally_prompt, enemy_prompt


if __name__ == "__main__":

    pkl_path = "/mnt/HDD/wangchao/smac_v1_origin/"
    json_path = "/mnt/HDD/wangchao/smac_v1_json/"
    os.makedirs(json_path, exist_ok=True)
    map_config_path = "/home/wangchao/work/marl-ppo-suite/envs/smacv2/config/"

    pkl_files = os.listdir(pkl_path)
    pkl_files = [f for f in pkl_files if f.endswith(".pkl")]
    pkl_files.sort()
    for pkl_file in pkl_files:
        game_name, algo_name = pkl_file.rsplit(".", 1)[0].rsplit("_", 1)
        race = game_name.split("_")[0]
        if algo_name not in ["Good"]:
            # print(f"Skipping {algo_name} for {game_name}")
            continue
        print(f"Game: {game_name}, Algorithm: {algo_name}")

        json_file = os.path.join(json_path, f"{game_name}_{algo_name}.json")
        fd = open(json_file, "a", encoding="utf-8")

        # Load map configuration
        map_config = {"map_name": game_name, "capability_config": {}}
        env_wrapper = StarCraftCapabilityEnvWrapper(**map_config)
        total_actions = env_wrapper.env.get_total_actions()
        state_size = env_wrapper.env.get_state_size()
        ally_num_attributes = env_wrapper.env.get_ally_num_attributes()
        enemy_num_attributes = env_wrapper.env.get_enemy_num_attributes()

        n_agents = env_wrapper.env.n_agents
        n_enemies = env_wrapper.env.n_enemies
        print(
            f"Number of agents: {n_agents}, Number of enemies: {n_enemies}, "
            f"State size: {state_size}, Total actions: {total_actions}, "
            f"Ally attributes: {ally_num_attributes}, Enemy attributes: {enemy_num_attributes}"
        )

        max_cd = {
            "marine": 15,
            "marauder": 25,
            "medivac": 200,  # max energy
            "stalker": 35,
            "zealot": 22,
            "colossus": 24,
            "hydralisk": 10,
            "zergling": 11,
            "baneling": 1,
        }
        sight_range_map = {
            "stalker": 10,
            "zealot": 9,
            "colossus": 10,
            "zergling": 8,
            "baneling": 8,
            "hydralisk": 9,
            "marine": 9,
            "marauder": 10,
            "medivac": 11,
        }

        max_x = 32
        max_y = 32

        assert (
            state_size
            == ally_num_attributes * n_agents
            + enemy_num_attributes * n_enemies
            + total_actions * n_agents
        ), "State size does not match the expected size based on attributes and actions"

        unit_type_bits = env_wrapper.env.unit_type_bits
        ally_has_shield = env_wrapper.env.shield_bits_ally > 0
        enemy_has_shield = env_wrapper.env.shield_bits_enemy > 0
        unit_names = ["marine"]
        game_sight_range_map = {
            unit_name: sight_range_map.get(unit_name, 9) for unit_name in unit_names
        }

        static_prompt = (
            system_prompt["introduction"]
            + f"当前游戏地图为{game_name}"
            + system_prompt["map_info"].get(game_name, "")
            + system_prompt["race"].get(race, "")
            + system_prompt["coordination"]
            + system_prompt["task"]
        )

        # Load trajectory
        evaluation_data = pickle.load(open(os.path.join(pkl_path, pkl_file), "rb"))
        # assert isinstance(evaluation_data, list), "Evaluation data should be a list"
        assert isinstance(
            evaluation_data, dict
        ), "smac_v1 Evaluation data should be a dict"
        num_episodes = len(evaluation_data["actions"])
        for episode_index in tqdm.tqdm(range(num_episodes)):
            episode = {k: v[episode_index] for k, v in evaluation_data.items()}
            episode_length = len(episode["state"])
            for step in range(episode_length):
                state = episode["state"][step]
                avail_actions_masks = episode["legals"][step]
                actions = episode["actions"][step]

                text_act = ",".join([str(int(a)) for a in actions[:n_agents]])

                ally_prompt, enemy_prompt = get_prompt_step(
                    state=state,
                    avail_actions_masks=avail_actions_masks,
                    env_wrapper=env_wrapper,
                    n_agents=n_agents,
                    n_enemies=n_enemies,
                    ally_num_attributes=ally_num_attributes,
                    enemy_num_attributes=enemy_num_attributes,
                    max_x=max_x,
                    max_y=max_y,
                    unit_type_bits=unit_type_bits,
                    unit_names=unit_names,
                    game_sight_range_map=game_sight_range_map,
                    max_cd=max_cd,
                    ally_has_shield=ally_has_shield,
                    enemy_has_shield=enemy_has_shield,
                )

                step_prompt = (
                    f"当前进行到第{step}个step\n"
                    f"友方的状态:\n{ally_prompt}\n"
                    f"敌方的状态:\n{enemy_prompt}\n"
                )

                prompt = {
                    "input": [
                        {"role": "system", "content": static_prompt},
                        {
                            "role": "user",
                            "content": step_prompt + instruct_prompt,
                        },
                    ],
                    "target": text_act,
                }
                # print(prompt)
                json.dump(prompt, fd, ensure_ascii=False)
                fd.write("\n")
                # break
            # break
        fd.close()
        print(f"Finish updating {json_file}")
        # break
