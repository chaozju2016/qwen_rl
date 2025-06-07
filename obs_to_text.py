import os
import pickle
import yaml
import torch
import json
import tqdm

from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

direction_info = {
    "abs": {
        0: "北",
        1: "南",
        2: "东",
        3: "西",
    },
    "rel": {
        0: "上方",
        1: "下方",
        2: "右侧",
        3: "左侧",
    },
}
direction_type = "rel"  # "abs" or "rel"
assert (
    direction_type in direction_info.keys()
), f"direction_type should be in {list(direction_info.keys())}"

system_prompt = {
    "introduction": {
        "central_controller": (
            "SMACv2 (StarCraft Multi-Agent Challenge v2) 是一个实时战略游戏环境,"
            "你将作为中央控制器同时指挥多个友方单位与敌方作战。"
            "\n"
        ),
        "individual_agent": (
            "SMACv2 (StarCraft Multi-Agent Challenge v2) 是一个实时战略游戏环境,"
            "你与友方单位一起与敌方作战。"
            "\n"
        ),
    },
    "map_info": {
        "protoss_5_vs_5": "友方有5个单位,敌方有5个单位\n",
        "protoss_10_vs_10": "友方有10个单位,敌方有10个单位\n",
        "terran_5_vs_5": "友方有5个单位,敌方有5个单位\n",
        "terran_10_vs_10": "友方有10个单位,敌方有10个单位\n",
        "zerg_5_vs_5": "友方有5个单位,敌方有5个单位\n",
        "zerg_10_vs_10": "友方有10个单位,敌方有10个单位\n",
        "zerg_10_vs_11": "友方有10个单位,敌方有11个单位\n",
    },
    "race": {
        "terran": (
            "敌我均会包含下列三种单位："
            "Marine - 基础步兵单位,血量45,护甲0,攻击力6,射程5游戏单位。"
            "Marauder - 重装步兵,血量125,护甲1,攻击力10,射程6游戏单位。"
            "Medivac - 医疗运输船,血量150,护甲1,无攻击能力但可治疗友军。"
            "\n"
        ),
        "zerg": (
            "敌我均会包含下列三种单位："
            "Zergling - 近战单位,血量35,护甲0,攻击力5,射程1游戏单位。"
            "Hydralisk - 远程单位,血量80,护甲0,攻击力12,射程5游戏单位。"
            "Baneling - 自爆单位,血量30,护甲0,爆炸伤害80,射程2.2游戏单位。"
            "\n"
        ),
        "protoss": (
            "敌我均会包含下列三种单位："
            "Zealot - 近战单位,血量100+50护盾,护甲1,攻击力8×2,射程1游戏单位。"
            "Stalker - 远程单位,血量80+80护盾,护甲1,攻击力10,射程6游戏单位。"
            "Colossus - 重型单位,血量200+150护盾,护甲1,攻击力15×2,射程7游戏单位。"
            "\n"
        ),
    },
    "coordination": (
        "所有地图的标准尺寸为32x32游戏单位,在相对坐标系统中对应[-1,1]x[-1,1]的范围"
        "所有单位位置使用相对坐标系统："
        "以地图中心为原点(0,0),坐标值范围[-1,1]。"
        f"格式为(rel_x, rel_y),其中rel_x正值={direction_info[direction_type][2]}/负值={direction_info[direction_type][3]},rel_y正值={direction_info[direction_type][0]}/负值={direction_info[direction_type][1]}。"
        f"例如Marine位于(0.15, -0.22)表示其在地图中心右侧0.15、下方0.22处。"
        "\n"
    ),
    "task": {
        "central_controller": (
            "你的任务是根据友方单位和敌方单位的状态信息,为每个友方单位选择一个最合适的有效动作。"
            "你需要考虑友方单位的生命值、位置等信息,并在每个step中做出决策。"
            "目标是最大化友方单位的生存时间和击败敌方单位。"
            "如果某个友方单位已死亡,则只能为其选择动作0(无动作)。"
            "\n"
        ),
        "individual_agent": (
            "你的任务是根据友方单位和敌方单位的状态信息,为你自己选择一个最合适的有效动作。"
            "你需要考虑自己的生命值、位置等信息,并在每个step中做出决策。"
            "目标是最大化友方单位的生存时间和击败敌方单位。"
            "如果你已死亡,则只能选择动作0(无动作)。"
            "\n"
        ),
    },
}
role_type = "central_controller"
assert (
    role_type in system_prompt["introduction"].keys()
), f"role_type should be in {list(system_prompt['introduction'].keys())}"

instruct_prompt = {
    "central_controller": (
        "根据以上信息,为每个友方单位选择一个最合适的有效动作,"
        "格式为可行动作编号int构成的字符串 'action_0,action_1, ...'。"
        "如果某个友方单位已死亡,则只能为其选择动作0。\n\n"
    ),
    "individual_agent": (
        "根据以上信息,为你自己选择一个最合适的有效动作,"
        "格式为可行动作编号int构成的字符串 'action'。"
        "如果你已死亡,则只能选择动作0。\n\n"
    ),
}

action_info = {
    0: "无动作(单位已死亡)",
    1: "原地不动",
    2: f"向{direction_info[direction_type][0]}移动",
    3: f"向{direction_info[direction_type][1]}移动",
    4: f"向{direction_info[direction_type][2]}移动",
    5: f"向{direction_info[direction_type][3]}移动",
}


def get_static_action_info(
    num_actions: int = 6, num_self_actions: int = 6, is_medivac=False
) -> str:
    """
    根据可用动作的索引列表,返回对应的动作信息字符串。
    """
    action_info_prompt = ""
    for action_index in range(num_actions):
        action_str = action_info.get(
            action_index,
            f"攻击敌方{'(对于medivac则为治疗友方)' if is_medivac else ''}单位{action_index-num_self_actions}",
        )
        action_info_prompt += f"动作{action_index}含义为{action_str}\n"
    return action_info_prompt


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
        return f"{unit_name}已死亡,可用动作为[0]。\n"

    horizontal = (
        f"{direction_info[direction_type][2]}"
        if rel_x > 0
        else f"{direction_info[direction_type][3]}" if rel_x < 0 else "中央"
    )
    vertical = (
        f"{direction_info[direction_type][0]}"
        if rel_y > 0
        else f"{direction_info[direction_type][1]}" if rel_y < 0 else "中央"
    )

    prompt = (
        f"{unit_name}当前拥有{health_in_percent:.0f}%的生命值{f'和{sheild*100:.0f}%的护盾' if sheild else ''},"
        f"位于地图{horizontal}{abs(rel_x):.2f}, {vertical}{abs(rel_y):.2f}位置,"
        # f"其{'技能冷却时间'if is_cooldown_in_sec else '剩余能量'}为{energy_or_cooldown:.1f}{'秒'if is_cooldown_in_sec else '点'},"
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
        return f"{unit_name}已死亡,可用动作为[0]。\n"

    horizontal = (
        f"{direction_info[direction_type][2]}"
        if rel_x > 0
        else f"{direction_info[direction_type][3]}" if rel_x < 0 else "中央"
    )
    vertical = (
        f"{direction_info[direction_type][0]}"
        if rel_y > 0
        else f"{direction_info[direction_type][1]}" if rel_y < 0 else "中央"
    )

    prompt = (
        f"{unit_name}当前拥有{health_in_percent:.0f}%的生命值{f'和{sheild*100:.0f}%的护盾' if sheild else ''},"
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
            torch.argmax(
                state[
                    (agent_id + 1) * ally_num_attributes
                    - unit_type_bits : (agent_id + 1) * ally_num_attributes
                ]
            ).item()
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
            torch.argmax(
                state[
                    n_agents * ally_num_attributes
                    + (enemy_id + 1) * enemy_num_attributes
                    - unit_type_bits : n_agents * ally_num_attributes
                    + (enemy_id + 1) * enemy_num_attributes
                ]
            ).item()
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

    pkl_path = "/mnt/HDD/wangchao/smac_v2/"
    json_path = "/mnt/HDD/wangchao/smac_v2_json/"
    os.makedirs(json_path, exist_ok=True)
    map_config_path = "/home/wangchao/work/marl-ppo-suite/envs/smacv2/config/"
    max_episodes = 2000

    pkl_files = os.listdir(pkl_path)
    pkl_files = [f for f in pkl_files if f.endswith(".pkl")]
    pkl_files.sort()
    print(f"Found {len(pkl_files)} pkl files in {pkl_path}")
    for pkl_file in pkl_files:
        game_name, algo_name = pkl_file.rsplit(".", 1)[0].rsplit("_", 1)
        race = game_name.split("_")[0]
        print(f"Game: {game_name}, Algorithm: {algo_name}")

        json_file = os.path.join(json_path, f"{game_name}_{algo_name}.json")
        fd = open(json_file, "a", encoding="utf-8")

        # Load map configuration
        map_config = yaml.load(
            open(
                os.path.join(map_config_path, f"{game_name}.yaml"),
                "r",
                encoding="utf-8",
            ),
            Loader=yaml.FullLoader,
        )
        env_wrapper = StarCraftCapabilityEnvWrapper(**map_config)
        total_actions = env_wrapper.env.get_total_actions()
        state_size = env_wrapper.env.get_state_size()
        ally_num_attributes = env_wrapper.env.get_ally_num_attributes()
        enemy_num_attributes = env_wrapper.env.get_enemy_num_attributes()

        n_agents = env_wrapper.env.n_agents
        n_enemies = env_wrapper.env.n_enemies

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
        sight_range_map = (
            {
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
            if map_config["use_unit_ranges"]
            else {}
        )

        max_x = map_config["capability_config"]["start_positions"]["map_x"]
        max_y = map_config["capability_config"]["start_positions"]["map_y"]

        assert (
            state_size
            == ally_num_attributes * n_agents
            + enemy_num_attributes * n_enemies
            + total_actions * n_agents
        ), "State size does not match the expected size based on attributes and actions"

        unit_type_bits = env_wrapper.env.unit_type_bits
        ally_has_shield = env_wrapper.env.shield_bits_ally > 0
        enemy_has_shield = env_wrapper.env.shield_bits_enemy > 0
        unit_names = map_config["capability_config"]["team_gen"]["unit_types"]
        game_sight_range_map = {
            unit_name: sight_range_map.get(unit_name, 9) for unit_name in unit_names
        }

        static_prompt = (
            system_prompt["introduction"][role_type]
            + f"当前游戏地图为{game_name},"
            + system_prompt["map_info"].get(game_name, "")
            + system_prompt["race"].get(race, "")
            + system_prompt["coordination"]
            + system_prompt["task"][role_type]
            + get_static_action_info(
                num_actions=total_actions,
                num_self_actions=total_actions - n_enemies,
                is_medivac=(race == "terran"),
            )
        )

        # Load trajectory
        evaluation_data = pickle.load(open(os.path.join(pkl_path, pkl_file), "rb"))
        assert isinstance(evaluation_data, list), "Evaluation data should be a list"
        num_episodes = min(len(evaluation_data), max_episodes)

        for episode in tqdm.tqdm(evaluation_data[:num_episodes]):
            assert isinstance(episode, dict), "Each episode should be a dict"
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
                            "content": step_prompt + instruct_prompt[role_type],
                        },
                    ],
                    "target": text_act,
                }
                # print(json.dumps(prompt, indent=2, ensure_ascii=False))
                json.dump(prompt, fd, ensure_ascii=False)
                fd.write("\n")
                # break
            # break
        fd.close()
        print(f"Finish updating {json_file}")
        # break
