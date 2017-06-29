### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters

import numpy as np
import gym
import time
from lake_envs import *

def compute_value(P, V, nS, action, state):
	reward = 0
	for i in range(len(P[state][action])):
		reward += P[state][action][i][0] * V[P[state][action][i][1]]
		# print("State %d, action %d, next state %d, reward %g, because prob is %g and value is %g" % (state, action, P[state][action][i][1], reward, P[state][action][i][0], V[P[state][action][i][1]]))
	return reward

def compute_im_reward(nS, nA, terminal_states):
	im_reward = [[[] for j in range(nA)] for i in range(nS)]
	for state in range(nS):
		for action in range(nA):
			if state in terminal_states.keys():
				im_reward[state][action] = terminal_states[state]
			else:
				im_reward[state][action] = 0
	return im_reward

def value_iteration(P, nS, nA, terminal_states, gamma=0.9, max_iteration=20, tol=1e-3):
	"""
	Learn value function and policy by using value iteration method for a given
	gamma and environment.

	Parameters:
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""
	V = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)
	im_reward = compute_im_reward(nS, nA, terminal_states)

	############################
	# YOUR IMPLEMENTATION HERE #
	# init terminal states of V
	for k in range(max_iteration):
		new_V = np.zeros(nS)
		for state in range(nS):
			opt_reward = 0
			for action in range(nA):
				reward = gamma * compute_value(P, V, nS, action, state) + im_reward[state][action]
				if reward >= opt_reward:
					opt_reward = reward
			new_V[state] = opt_reward
		V = new_V

	for state, value in enumerate(V):
		print("state %d: %g" % (state, value))

	return V
	############################
	return V, policy

def policy_evaluation(P, nS, nA, policy, im_reward, gamma=0.9, max_iteration=1000, tol=1e-3):
	"""Evaluate the value function from a given policy.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	policy: np.array
		The policy to evaluate. Maps states to actions.
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns
	-------
	value function: np.ndarray
		The value function from the given policy.
	"""
	############################
	# YOUR IMPLEMENTATION HERE #
	V = np.zeros(nS)
	new_V = np.zeros(nS)
	for cnt in range(max_iteration):
		for state in range(nS):
			new_V[state] = im_reward[state][policy[state]] + gamma * compute_value(P, V, nS, policy[state], state)
		V = new_V
	############################
	return V

def policy_improvement(P, nS, nA, value_from_policy, policy, im_reward, gamma=0.9):
	"""Given the value function from policy improve the policy.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

	Returns
	-------
	new policy: np.ndarray
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""    
	############################
	# YOUR IMPLEMENTATION HERE #
	############################
	return np.zeros(nS, dtype='int')

def policy_iteration(P, nS, nA, terminal_states, gamma=0.9, max_iteration=20, tol=1e-3):
	"""Runs policy iteration.

	You should use the policy_evaluation and policy_improvement methods to
	implement this method.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""
	# V = np.zeros(nS)
	Q = np.zeros((nS, nA))
	policy = np.zeros(nS, dtype=int)
	############################
	# YOUR IMPLEMENTATION HERE #
	im_reward = compute_im_reward(nS, nA, terminal_states)

	for epi in range(max_iteration):
		V = policy_evaluation(P, nS, nA, policy, im_reward)

		# below is policy improvement
		# I'm just too lazy to seperate them
		for state in range(nS):
			for action in range(nA):
				Q[state][action] = im_reward[state][action] + gamma * compute_value(P, V, nS, action, state)
		policy = np.argmax(Q, axis=1)
		# print(policy)
	############################
	return policy

def example(env):
	"""Show an example of gym
	Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
	"""
	env.seed(0); 
	from gym.spaces import prng; prng.seed(10) # for print the location
	# Generate the episode
	ob = env.reset()
	for t in range(100):
		env.render()
		a = env.action_space.sample()
		ob, rew, done, _ = env.step(a)
		if done:
			break
	assert done
	env.render();

def render_single(env, policy):
	"""Renders policy once on environment. Watch your agent play!

		Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
		Policy: np.array of shape [env.nS]
			The action to take at a given state
	"""

	episode_reward = 0
	ob = env.reset()
	for t in range(100):
		env.render()
		time.sleep(0.5) # Seconds between frames. Modify as you wish.
		a = policy[ob]
		ob, rew, done, _ = env.step(a)
		episode_reward += rew
		if done:
			break
	assert done
	env.render();
	print("Episode reward: %f" % episode_reward)


# Feel free to run your own debug code in main!
# Play around with these hyperparameters.
if __name__ == "__main__":
	env = gym.make("Deterministic-4x4-FrozenLake-v0")
	print(env.__doc__)
	print("Here is an example of state, action, reward, and next state")
	example(env)
	V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)
	V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)
	
