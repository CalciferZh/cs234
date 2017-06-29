### Episode model free learning using Q-learning and SARSA

# Do not change the arguments and output types of any of the functions provided! You may debug in Main and elsewhere.

import numpy as np
import gym
import time
from lake_envs import *

def learn_Q_QLearning(env, num_episodes=2000, gamma=0.95, lr=0.1, e=0.99, decay_rate=0.99):
  """Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy.
  Update Q at the end of every episode.

  Parameters
  ----------
  env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
  num_episodes: int 
    Number of episodes of training.
  gamma: float
    Discount factor. Number in range [0, 1)
  learning_rate: float
    Learning rate. Number in range [0, 1)
  e: float
    Epsilon value used in the epsilon-greedy method. 
  decay_rate: float
    Rate at which epsilon falls. Number in range [0, 1)

  Returns
  -------
  np.array
    An array of shape [env.nS x env.nA] representing state, action values
  """
  
  ############################
  # YOUR IMPLEMENTATION HERE #
  nA = env.nA
  nS = env.nS
  Q = np.zeros((nS, nA))
  for epi in range(num_episodes):
    # start a new episode
    state = env.reset()
    action = None
    # run until done
    # print("==================================================================")
    # print("Trying new episode...")
    # print(Q)
    while True:
      # print("At state %d" % state)
      action = 0
      epsilon = np.random.random()
      if epsilon > e:
        opt_reward = 0
        # take the action maximize Q[state][action]
        for i in range(nA):
          if Q[state, i] >= opt_reward:
            opt_reward = Q[state, i]
            action = i
        # print("Decide to exploit: e = %g, take action %d" % (e, action))
      else:
        # take a random action
        action = np.random.randint(nA)
        # print("Decide to explore: e = %g, take action %d" % (e, action))
      # take that action and observe
      new_state, im_reward, done, info = env.step(action)

      # print("Reach new state %d, get im_reward %g" % (new_state, im_reward))

      opt_Q = 0
      for i in range(nA):
        if Q[new_state][i] >= opt_Q:
          opt_Q = Q[new_state][i]
      Q_sample = im_reward + gamma * opt_Q
      
      # print("At new state %d, Q sample is %g" % (new_state, Q_sample))

      Q[state][action] = (1 - lr) * Q[state][action] + lr * Q_sample
      if done:
        break
      state = new_state
    if epi % 10 == 0:
      e *= decay_rate


  ############################
  return Q

def learn_Q_SARSA(env, num_episodes=2000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99):
  """Learn state-action values using the SARSA algorithm with epsilon-greedy exploration strategy
  Update Q at the end of every episode.

  Parameters
  ----------
  env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
  num_episodes: int 
    Number of episodes of training.
  gamma: float
    Discount factor. Number in range [0, 1)
  learning_rate: float
    Learning rate. Number in range [0, 1)
  e: float
    Epsilon value used in the epsilon-greedy method. 
  decay_rate: float
    Rate at which epsilon falls. Number in range [0, 1)

  Returns
  -------
  np.array
    An array of shape [env.nS x env.nA] representing state-action values
  """

  ############################
  # YOUR IMPLEMENTATION HERE #
  ############################

  return np.ones((env.nS, env.nA))

def render_single_Q(env, Q):
  """Renders Q function once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play Q function on. Must have nS, nA, and P as
      attributes.
    Q: np.array of shape [env.nS x env.nA]
      state-action values.
  """

  episode_reward = 0
  state = env.reset()
  done = False
  while not done:
    env.render()
    # time.sleep(0.5) # Seconds between frames. Modify as you wish.
    action = np.argmax(Q[state])
    state, reward, done, _ = env.step(action)
    episode_reward += reward
    input()

  print("Episode reward: %f" % episode_reward)

# Feel free to run your own debug code in main!
def main():
  env = gym.make('Stochastic-4x4-FrozenLake-v0')
  Q = learn_Q_QLearning(env)
  # Q = learn_Q_SARSA(env)
  render_single_Q(env, Q)

if __name__ == '__main__':
    main()
