"""Overloads the FrozenLake environment to make the rewards and dynamics explicit

FrozenLake is described here: https://gym.openai.com/envs/FrozenLake-v0/

Base implementation is here:
https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
"""

import numpy as np
import interface

from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

from explicit_env.envs.utils import discrete2explicit
from explicit_env.envs.explicit import IExplicitEnv, ExplicitEnvGetters


class ExplicitFrozenLakeEnv(
    FrozenLakeEnv, ExplicitEnvGetters, interface.implements(IExplicitEnv)
):
    """Explicit FrozenLake Environment"""

    reward_range = (0.0, 1.0)

    human_actions = ["←", "↓", "→", "↑"]

    def __init__(self, *args, **kwargs):
        """C-tor"""

        # Call super-constructor
        super().__init__(*args, **kwargs)

        # Populate IExplicitEnv terms from DiscreteEnv
        discrete2explicit(self)


def demo():
    """Demo function"""
    # Constructing tests that it meets IExplicitEnv requirements
    env = ExplicitFrozenLakeEnv()
    print(env)


if __name__ == "__main__":
    demo()
