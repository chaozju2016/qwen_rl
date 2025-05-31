import numpy as np
from smac.env import StarCraft2Env
from knowledge_data.unit_species import unit_species
from typing import Dict, List, Tuple, Optional


class SMACTextWrapper:
    """
    Wrapper for SMAC environment that converts observations to text format
    and processes actions from the model.
    """

    def __init__(self, map_name: str, seed: int = 42):
        """Initialize the SMAC environment wrapper."""
        self.env = StarCraft2Env(map_name=map_name, seed=seed)
        self.env.reset()
        self.n_agents = self.env.n_agents
        self.n_enemies = self.env.n_enemies
        self.n_actions = self.env.n_actions

        # Get map and unit configurations
        from config import get_map_config, get_units_config

        self.map_config, self.unit_types = get_map_config(map_name)
        self.unit_config = get_units_config(self.unit_types)

        # check if the map is heterogeneous, but all allies or enemies are within same type
        self.is_hetrogeneous = (self.env.unit_type_bits == 0) and (
            len(self.unit_types) > 1
        )
        # check if the map contains >2 types of units within same side
        self.is_hybrid = (
            self.env.map_type
            in [
                "MMM",
                "bane",
                "colossi_stalkers_zealots",
                "stalkers_and_zealots",
            ]
            and len(self.unit_types) > 1
        )

        if self.is_hetrogeneous:
            agent_type = self.env.map_type
            if agent_type != "colossus":
                # match map_config
                agent_type = agent_type[:-1]
            self.units = [agent_type] * self.n_agents

            enemy_type = self.unit_types[
                np.where(np.asarray(self.unit_types) != agent_type)[0][0]
            ]
            self.enemy_units = [enemy_type] * self.n_enemies
        elif self.is_hybrid:
            # Get ally unit type
            self.unit_type_ids = [
                self.env.get_unit_type_id(self.env.agents[i], ally=True)
                for i in range(self.n_agents)
            ]
            self.units = [self.unit_types[i] for i in self.unit_type_ids]

            # Get enemy unit type
            self.enemy_unit_type_ids = [
                self.env.get_unit_type_id(self.env.enemies[i], ally=False)
                for i in range(self.n_enemies)
            ]
            # Only for XsXz scenarios
            if self.env.map_type == "stalkers_and_zealots":
                self.enemy_unit_type_ids = 1 - np.array(self.enemy_unit_type_ids)
            self.enemy_units = [self.unit_types[i] for i in self.enemy_unit_type_ids]
        else:
            agent_type = self.unit_types[0]
            self.units = [agent_type] * self.n_agents
            self.enemy_units = [agent_type] * self.n_enemies

    def reset(self) -> str:
        """Reset the environment and return the initial observation as text."""
        if self.env._episode_steps != 0:
            obs, state = self.env.reset()
        else:
            obs = self.env.get_obs()
            state = self.env.get_state()
        state = self.env.get_state()
        avail_actions = self.env.get_avail_actions()

        # Convert the observation to text
        text_obs = self._obs_to_text(obs, state, avail_actions, step=0)
        return text_obs

    def step(self, actions: List[int]) -> Tuple[str, List[float], bool, Dict]:
        """Take a step in the environment using the provided actions."""
        # Take a step in the environment
        reward, done, info = self.env.step(actions)

        # Get the next observation
        obs = self.env.get_obs()
        state = self.env.get_state()
        avail_actions = self.env.get_avail_actions()

        # Convert the observation to text
        text_obs = self._obs_to_text(
            obs, state, avail_actions, step=self.env._episode_steps
        )

        return text_obs, reward, done, info

    def _act_to_text(self, act: np.ndarray) -> str:
        return ",".join(np.asarray(act, dtype=str))

    def _obs_to_text(
        self,
        obs: List[np.ndarray],
        state: np.ndarray,
        avail_actions: List[np.ndarray],
        step: int,
    ) -> str:
        """Convert SMAC observation to text format."""
        # Initialize the text observation
        text = ""
        text += f"Current timestep: {step}\n"

        # Add information about friendly units
        text += "Ally units:\n"
        for i in range(self.n_agents):
            unit = self.env.get_unit_by_id(i)
            unit_type = self.units[i]  # Unit type
            health = float(unit.health)  # Health
            health_max = float(unit.health_max)  # Health
            if health <= 0:
                text += f"Agent {i} {unit_type} is dead.\n"
                continue
            try:
                shield = float(unit.shield)  # Shield
                shield_max = float(unit.shield_max)  # Shield
            except:
                shield = None  # only protoss has shield
            x_pos = float(unit.pos.x)  # X position
            y_pos = float(unit.pos.y)  # Y position

            text += (
                f"Agent {i} {unit_type} is at ({x_pos:.2f}, {y_pos:.2f}) "
                + f"with {health:.0f}/{health_max:.0f} Health"
                + (
                    f" and {shield:.0f}/{shield_max:.0f} Shield"
                    if shield is not None and shield_max != 0
                    else ""
                )
                + "\n"
            )

            # Add available actions for this agent
            available_action_indices = np.where(avail_actions[i])[0]
            text += f" Available actions for Agent {i}: {available_action_indices}\n"

        visibility_matrix = self.env.get_visibility_matrix()
        visible_enemies = np.arange(self.n_enemies)[
            np.any(visibility_matrix[:, self.n_agents :], axis=0)
        ]

        # Add information about enemy units from state
        if len(visible_enemies) == 0:
            text += "No visible enemy units.\n"
        else:
            text += "Visible Enemy units:\n"
            for i in visible_enemies:
                unit = self.env.enemies[i]
                unit_type = self.enemy_units[i]
                health = float(unit.health)  # Health
                health_max = float(unit.health_max)  # Health
                if health <= 0:
                    text += f"Enemy {i} {unit_type} is dead.\n"
                    continue
                try:
                    shield = float(unit.shield)  # Shield
                    shield_max = float(unit.shield_max)  # Shield
                except:
                    shield = None  # only protoss has shield
                x_pos = float(unit.pos.x)  # X position
                y_pos = float(unit.pos.y)  # Y position

                text += (
                    f"Enemy {i} {unit_type} is at ({x_pos:.2f}, {y_pos:.2f}) "
                    + f"with {health:.0f}/{health_max:.0f} Health"
                    + (
                        f" and {shield:.0f}/{shield_max:.0f} Shield"
                        if shield is not None and shield_max != 0
                        else ""
                    )
                    + "\n"
                )

        return text

    def get_action_mask(self) -> List[np.ndarray]:
        """Get the action mask for each agent (which actions are available)."""
        return self.env.get_avail_actions()

    def close(self):
        """Close the environment."""
        self.env.close()

    @property
    def action_space(self):
        """Get the action space of the environment."""
        return self.env.action_space

    @property
    def observation_space(self):
        """Get the observation space of the environment."""
        return self.env.observation_space


if __name__ == "__main__":
    # Example usage
    from configs.map_config import MapConfig

    mc = MapConfig()
    for map_name in list(mc.config.keys()):
        print("=" * 16)
        smac_env = SMACTextWrapper(map_name=map_name)
        print(f"Environment {map_name} initialized.")
        print(f"Map Config: {smac_env.map_config}")
        print(f"Unit Config: {smac_env.unit_config}")
        initial_obs = smac_env.reset()
        print("Initial Observation:", initial_obs)
        action_mask = smac_env.env.get_avail_actions()
        print("Action Mask:", action_mask)

        actions = [1] * smac_env.n_agents  # Example actions
        next_obs, reward, done, info = smac_env.step(actions)
        print("Step Result:")
        print("Next Observation:", next_obs)
        print("Reward:", reward)
        print("Done:", done)
        print("Info:", info)

        smac_env.close()
        print(f"Environment {map_name} closed.")
        print("=" * 16)
