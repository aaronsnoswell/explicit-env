from gym.envs.registration import register

# Bring imports to root level for convenience
from explicit_env.envs.explicit import IExplicitEnv, ExplicitEnvGetters
from explicit_env.envs.explicit_frozen_lake import ExplicitFrozenLakeEnv
from explicit_env.envs.explicit_linear import ExplicitLinearEnv
from explicit_env.envs.explicit_nchain import ExplicitNChainEnv
from explicit_env.envs.explicit_taxi import ExplicitTaxiEnv

# Add environments to the Gym register
register(
    id="ExplicitFrozenLakeEnv-v0",
    entry_point="explicit_env.envs.explicit_frozen_lake:ExplicitFrozenLakeEnv",
)

register(
    id="ExplicitLinearEnv-v0",
    entry_point="explicit_env.envs.explicit_linear:ExplicitLinearEnv",
)

register(
    id="ExplicitNChainEnv-v0",
    entry_point="explicit_env.envs.explicit_nchain:ExplicitNChainEnv",
)

register(
    id="ExplicitTaxiEnv-v0",
    entry_point="explicit_env.envs.explicit_taxi:ExplicitTaxiEnv",
)
