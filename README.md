# Explicit Env

A more rigorous OpenAI Gym
[`Env`](https://github.com/openai/gym/blob/master/gym/core.py) interface for MDPs with
discrete and explicit dynamics.

Using the `IExplicitEnv` interface, discrete state and action MDPs are easier to work
with.

## Installation

This package is not distributed on PyPI - you'll have to install from source.

```bash
git clone https://github.com/aaronsnoswell/explicit-env.git
cd explicit-env
pip install -e .
```

## Usage

### Pre-packaged environments

Importing the library will add Explicit versions of the following MDPs to the gym
register;

```python
import gym
from explicit_env import *
env = gym.make("ExplicitFrozenLakeEnv-v0")
```

 * [`FrozenLakeEnv`](https://gym.openai.com/envs/FrozenLake-v0/) -> [`ExplicitFrozenLakeEnv`](envs/explicit_frozen_lake.py)
 * [`NChainEnv`](https://gym.openai.com/envs/NChain-v0/) -> [`ExplicitNChainEnv`](envs/explicit_nchain.py)
 * [`TaxiEnv`](https://gym.openai.com/envs/Taxi-v3/) -> [`ExplicitTaxiEnv`](envs/explicit_taxi.py)
 * [`ExplicitLinearEnv`](envs/explicit_linear.py) - A simple linear MDP

### Using the `IExplicitEnv` interface with your own Environment

To use this with your own MDPs, extend the [`IExplicitEnv`](envs/explicit.py) interface.

A simple way to meet the criteria of this interface is to add underscore-prefixed
properties to your `Env` object during `__init__()`, then simply make sure the `Env`
object extends [`ExplicitEnvGetters`](envs/explicit.py).

For example, you have an Environment definition;

```python
import gym


class MyCustomEnv(gym.Env):
    
    def __init_(self):
        # Do construction stuff etc.
        pass
```

To convert this to an ExplicitEnv, you would make the following changes;

```python
import gym
import numpy as np
from explicit_env.envs import IExplicitEnv, ExplicitEnvGetters


class MyCustomEnv(gym.Env, IExplicitEnv, ExplicitEnvGetters):
    
    def __init_(self):
        # Do construction stuff etc.
        
        # Add IExplicitEnv members privately
        self._states = np.arange(self.observation_space.n)
        self._actions = np.arange(self.action_space.n)
        # ...etc. for other IExplicitEnv properties
```

### Solving IExplicitEnv environments

This library also includes Numba-optimized functions for solving explicit environments.

E.g. to solve for an optimal deterministic policy that matches the `stable-baselines`
`.predict()` method signature;

```python
from explicit_env.soln import value_iteration, q_from_v, EpsilonGreedyPolicy


env._gamma = 0.99       # N.b. can't solve with VI if discount is 1.0 
v_star = value_iteration(env)
q_star = q_from_v(v_star, env)
pi_star = EpsilonGreedyPolicy(q_star, epsilon=0.0)

s0 = env.reset()
action, _ = pi_star.predict(s0)
```
