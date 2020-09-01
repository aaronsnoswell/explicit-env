"""Find optimal Value and Policy functions for explicit Gym environments
"""

import pickle
import warnings
import numpy as np
import itertools as it

from numba import jit


def nonb_value_iteration(env, eps=1e-6, verbose=False, max_iter=None):
    """Value iteration to find the optimal value function
    
    Args:
        env (.envs.explicit.IExplicitEnv) Explicit Gym environment
        
        eps (float): Value convergence tolerance
        verbose (bool): Extra logging
        max_iter (int): If provided, iteration will terminate regardless of convergence
            after this many iterations.
    
    Returns:
        (numpy array): |S| vector of state values
    """

    if env.gamma == 1.0:
        warnings.warn(
            "Environment discount factor is 1.0 - value iteration will only converge if all paths terminate in a finite number of steps"
        )

    value_fn = np.zeros(env.t_mat.shape[0])

    # Prepare linear reward arrays
    _state_rewards = env.state_rewards
    if _state_rewards is None:
        _state_rewards = np.zeros(env.t_mat.shape[0])
    _state_action_rewards = env.state_action_rewards
    if _state_action_rewards is None:
        _state_action_rewards = np.zeros(env.t_mat.shape[0 : 1 + 1])
    _state_action_state_rewards = env.state_action_state_rewards
    if _state_action_state_rewards is None:
        _state_action_state_rewards = np.zeros(env.t_mat.shape)

    for _iter in it.count():
        delta = 0
        for s1 in env.states:
            v = value_fn[s1]
            value_fn[s1] = np.max(
                [
                    np.sum(
                        [
                            env.t_mat[s1, a, s2]
                            * (
                                _state_action_rewards[s1, a]
                                + _state_action_state_rewards[s1, a, s2]
                                + _state_rewards[s2]
                                + env.gamma * value_fn[s2]
                            )
                            for s2 in env.states
                        ]
                    )
                    for a in env.actions
                ]
            )
            delta = max(delta, np.abs(v - value_fn[s1]))

        if max_iter is not None and _iter >= max_iter:
            if verbose:
                print("Terminating before convergence at {} iterations".format(_iter))
                break

        # Check value function convergence
        if delta < eps:
            break
        else:
            if verbose:
                print("Value Iteration #{}, delta={}".format(_iter, delta))

    return value_fn


def value_iteration(env, eps=1e-6, verbose=False, max_iter=None):
    """Value iteration to find the optimal value function
    
    Args:
        env (.envs.explicit.IExplicitEnv) Explicit Gym environment
        
        eps (float): Value convergence tolerance
        verbose (bool): Extra logging
        max_iter (int): If provided, iteration will terminate regardless of convergence
            after this many iterations.
    
    Returns:
        (numpy array): |S| vector of state values
    """

    # Numba has trouble typing these arguments to the forward/backward functions below.
    # We manually convert them here to avoid typing issues at JIT compile time
    rs = env.state_rewards
    if rs is None:
        rs = np.zeros(env.t_mat.shape[0], dtype=np.float)

    rsa = env.state_action_rewards
    if rsa is None:
        rsa = np.zeros(env.t_mat.shape[0:2], dtype=np.float)

    rsas = env.state_action_state_rewards
    if rsas is None:
        rsas = np.zeros(env.t_mat.shape[0:3], dtype=np.float)

    return _nb_value_iteration(
        env.t_mat, env.gamma, rs, rsa, rsas, eps=eps, verbose=verbose, max_iter=max_iter
    )


@jit(nopython=True)
def _nb_value_iteration(
    t_mat, gamma, rs, rsa, rsas, eps=1e-6, verbose=False, max_iter=None
):
    """Value iteration to find the optimal value function
    
    Args:
        eps (float): Value convergence tolerance
        verbose (bool): Extra logging
        max_iter (int): If provided, iteration will terminate regardless of convergence
            after this many iterations.
    
    Returns:
        (numpy array): |S| vector of state values
    """

    value_fn = np.zeros(t_mat.shape[0])

    _iter = 0
    while True:
        delta = 0
        for s1 in range(t_mat.shape[0]):
            v = value_fn[s1]
            action_values = np.zeros(t_mat.shape[1])
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):
                    action_values[a] += t_mat[s1, a, s2] * (
                        rsa[s1, a] + rsas[s1, a, s2] + rs[s2] + gamma * value_fn[s2]
                    )
            value_fn[s1] = np.max(action_values)
            delta = max(delta, np.abs(v - value_fn[s1]))

        if max_iter is not None and _iter >= max_iter:
            if verbose:
                print("Terminating before convergence, # iterations = ", _iter)
                break

        # Check value function convergence
        if delta < eps:
            break
        else:
            if verbose:
                print("Value Iteration #", _iter, " delta=", delta)

        _iter += 1

    return value_fn


@jit(nopython=True)
def _nb_q_from_v(
    v_star,
    t_mat,
    gamma,
    state_rewards,
    state_action_rewards,
    state_action_state_rewards,
):
    """Find Q* given V* (numba optimized version)
    
    Args:
        v_star (numpy array): |S| vector of optimal state values
        t_mat (numpy array): |S|x|A|x|S| transition matrix
        gamma (float): Discount factor
        state_rewards (numpy array): |S| array of state rewards
        state_action_rewards (numpy array): |S|x|A| array of state-action rewards
        state_action_state_rewards (numpy array): |S|x|A|x|S| array of state-action-state rewards
    
    Returns:
        (numpy array): |S|x|A| array of optimal state-action values
    """

    q_star = np.zeros(t_mat.shape[0 : 1 + 1])

    for s1 in range(t_mat.shape[0]):
        for a in range(t_mat.shape[1]):
            for s2 in range(t_mat.shape[2]):
                q_star[s1, a] += t_mat[s1, a, s2] * (
                    state_action_rewards[s1, a]
                    + state_action_state_rewards[s1, a, s2]
                    + state_rewards[s2]
                    + gamma * v_star[s2]
                )

    return q_star


def q_from_v(v_star, env):
    """Find Q* given V*
    
    Args:
        v_star (numpy array): |S| vector of optimal state values
        env (.envs.explicit.IExplicitEnv) Explicit Gym environment
    
    Returns:
        (numpy array): |S|x|A| array of optimal state-action values
    """

    # Prepare linear reward arrays
    _state_rewards = env.state_rewards
    if _state_rewards is None:
        _state_rewards = np.zeros(env.t_mat.shape[0])
    _state_action_rewards = env.state_action_rewards
    if _state_action_rewards is None:
        _state_action_rewards = np.zeros(env.t_mat.shape[0 : 1 + 1])
    _state_action_state_rewards = env.state_action_state_rewards
    if _state_action_state_rewards is None:
        _state_action_state_rewards = np.zeros(env.t_mat.shape)

    return _nb_q_from_v(
        v_star,
        env.t_mat,
        env.gamma,
        _state_rewards,
        _state_action_rewards,
        _state_action_state_rewards,
    )


@jit(nopython=True)
def _nb_policy_evaluation(
    t_mat,
    gamma,
    state_rewards,
    state_action_rewards,
    state_action_state_rewards,
    policy_vector,
    eps=1e-6,
):
    """Determine the value function of a given deterministic policy
    
    Args:
        env (.envs.explicit.IExplicitEnv) Explicit Gym environment
        policy (object): Policy object providing a deterministic .predict(s) method to
            match the stable-baselines policy API
        
        eps (float): State value convergence threshold
    
    Returns:
        (numpy array): |S| state value vector
    """

    v_pi = np.zeros(t_mat.shape[0])

    _iteration = 0
    while True:
        delta = 0
        for s1 in range(t_mat.shape[0]):
            v = v_pi[s1]
            _tmp = 0
            for a in range(t_mat.shape[1]):
                if policy_vector[s1] != a:
                    continue
                for s2 in range(t_mat.shape[2]):
                    _tmp += t_mat[s1, a, s2] * (
                        state_action_rewards[s1, a]
                        + state_action_state_rewards[s1, a, s2]
                        + state_rewards[s2]
                        + gamma * v_pi[s2]
                    )
            v_pi[s1] = _tmp
            delta = max(delta, np.abs(v - v_pi[s1]))

        if delta < eps:
            break
        _iteration += 1

    return v_pi


def policy_evaluation(env, policy, eps=1e-6, num_runs=1):
    """Determine the value function of a given policy
    
    Args:
        env (.envs.explicit.IExplicitEnv) Explicit Gym environment
        policy (object): Policy object providing a .predict(s) method to match the
            stable-baselines policy API
        
        eps (float): State value convergence threshold
        num_runs (int): Number of policy evaluations to average over - for deterministic
            policies, leave this as 1, but for stochastic policies, set to a large
            number (the function will then sample actions stochastically from the
            policy).
    
    Returns:
        (numpy array): |S| state value vector
    """

    # Prepare linear reward arrays
    _state_rewards = env.state_rewards
    if _state_rewards is None:
        _state_rewards = np.zeros(env.t_mat.shape[0])
    _state_action_rewards = env.state_action_rewards
    if _state_action_rewards is None:
        _state_action_rewards = np.zeros(env.t_mat.shape[0 : 1 + 1])
    _state_action_state_rewards = env.state_action_state_rewards
    if _state_action_state_rewards is None:
        _state_action_state_rewards = np.zeros(env.t_mat.shape)

    print("Num urns is: ", num_runs)

    policy_state_values = []
    for _ in range(num_runs):
        actions = np.array(
            [policy.predict(s)[0] for s in env.states]
        )
        print(actions)
        policy_state_values.append(
            _nb_policy_evaluation(
                env.t_mat,
                env.gamma,
                _state_rewards,
                _state_action_rewards,
                _state_action_state_rewards,
                actions,
                eps=eps,
            )
        )
    return np.mean(policy_state_values, axis=0)


class Policy:
    """A simple Policy base class
    
    Provides a .predict(s) method to match the stable-baselines policy API
    """

    def __init__(self):
        """C-tor"""
        raise NotImplementedError

    def save(self, path):
        """Save policy to file"""
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        """Load policy from file"""
        with open(path, "rb") as file:
            _self = pickle.load(file)
            return _self

    def predict(self, s):
        """Predict next action and distribution over states
        
        N.b. This function matches the API of the stabe-baselines policies.
        
        Args:
            s (int): Current state
        
        Returns:
            (int): Sampled action
            (None): Placeholder to ensure this function matches the stable-baselines
                policy interface. Some policies use this argument to return a prediction
                over future state distributions - we do not.
        """
        action = np.random.choice(np.arange(self.q.shape[1]), p=self.prob_for_state(s))
        return action, None

    def path_log_likelihood(self, p):
        """Compute log-likelihood of [(s, a), ..., (s, None)] path under this policy
        
        Args:
            p (list): List of state-action tuples
        
        Returns:
            (float): Absolute log-likelihood of the path under this policy
        """

        # We start with 1.0 probability, and log(1.0) = 0.0
        ll = 0.0

        # N.b. - final tuple is (s, None), which we skip
        for s, a in p[:-1]:
            log_action_prob = np.log(self.prob_for_state(s)[a])
            if np.isneginf(log_action_prob):
                return -np.inf
            ll += log_action_prob

        return log_action_prob

    def prob_for_state(self, s):
        """Get the action probability vector for the given state
        
        Args:
            s (int): Current state
        
        Returns:
            (numpy array): Probability distribution over actions
        """
        raise NotImplementedError


class EpsilonGreedyPolicy(Policy):
    """An Epsilon Greedy Policy wrt. a provided Q function
    
    Provides a .predict(s) method to match the stable-baselines policy API
    """

    def __init__(self, q, epsilon=0.1):
        """C-tor
        
        Args:
            q (numpy array): |S|x|A| Q-matrix
            epsilon (float): Probability of taking a random action. Set to 0 to create
                an optimal stochsatic policy. Specifically,
                    Epsilon == 0.0 will make the policy sample between equally good
                        (== Q value) actions. If a single action has the highest Q
                        value, that action will always be chosen
                    Epsilon > 0.0 will make the policy act in an epsilon greedy
                        fashion - i.e. a random action is chosen with probability
                        epsilon, and an optimal action is chosen with probability
                        (1 - epsilon) + (epsilon / |A|).
        """
        self.q = q
        self.epsilon = epsilon

    def prob_for_state(self, s):
        """Get the action probability vector for the given state
        
        Args:
            s (int): Current state
        
        Returns:
            (numpy array): Probability distribution over actions, respecting the
                self.stochastic and self.epsilon parameters
        """

        # Get a list of the optimal actions
        action_values = self.q[s]
        best_action_value = np.max(action_values)
        best_action_mask = action_values == best_action_value
        best_actions = np.where(best_action_mask)[0]

        # Prepare action probability vector
        p = np.zeros(self.q.shape[1])

        # All actions share probability epsilon
        p[:] += self.epsilon / self.q.shape[1]

        # Optimal actions share additional probability (1 - epsilon)
        p[best_actions] += (1 - self.epsilon) / len(best_actions)

        return p


class OptimalPolicy(EpsilonGreedyPolicy):
    """An optimal policy - can be deterministic or stochastic"""

    def __init__(self, q, stochastic=True):
        """C-tor
        
        Args:
            q (numpy array): |S|x|A| Q-matrix
            stochastic (bool): If true, this policy will sample amongst optimal actions.
                Otherwise, the first optimal action will always be chosen.
        """
        super().__init__(q, epsilon=0.0)

        self.stochastic = stochastic

    def prob_for_state(self, s):
        p = super().prob_for_state(s)
        if not self.stochastic:
            # Always select the first optimal action
            a_star = np.where(p != 0)[0][0]
            p *= 0
            p[a_star] = 1.0
        return p


class BoltzmannExplorationPolicy(Policy):
    """A Boltzmann exploration policy wrt. a provided Q function
    
    Provides a .predict(s) method to match the stable-baselines policy API
    """

    def __init__(self, q, scale=1.0):
        """C-tor
        
        Args:
            q (numpy array): |S|x|A| Q-matrix
            scale (float): Temperature scaling factor on the range [0, inf).
                Actions are chosen proportional to exp(scale * Q(s, a)), so...
                 * Scale > 1.0 will exploit optimal actions more often
                 * Scale < 1.0 will explore sub-optimal actions more often
                 * Scale == 0.0 will uniformly sample actions
                 * Scale < 0.0 will prefer non-optimal actions
        """
        self.q = q
        self.scale = scale

    def prob_for_state(self, s):
        """Get the action probability vector for the given state
        
        Args:
            s (int): Current state
        
        Returns:
            (numpy array): Probability distribution over actions, respecting the
                self.stochastic and self.epsilon parameters
        """
        # Prepare action probability vector
        p = np.exp(self.scale * self.q[s])
        p /= np.sum(p)
        return p
