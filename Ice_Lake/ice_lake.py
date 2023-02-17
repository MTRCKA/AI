import gymnasium as gym
import numpy as np

env = gym.make('FrozenLake-v1', is_slippery=False)
env.reset()

num_episodes = 100000
max_steps_per_episode = 100
learning_rate = 0.8
discount_factor = 0.95
epsilon = 0.2

Q = np.zeros((env.observation_space.n, env.action_space.n))

# new_state, reward, done, info, test = env.step(1)


def epsilon_greedy_policy(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        # Random action
        return env.action_space.sample()
    else:
        # Greedy action
        return np.argmax(Q[state, :])


for i in range(num_episodes):
    state, prob = env.reset()
    for t in range(max_steps_per_episode):
        # Choose action based on epsilon-greedy policy
        action = epsilon_greedy_policy(state, epsilon)
        
        # Take action and observe new state and reward
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Update Q-table
        Q[state, action] += learning_rate * (reward + discount_factor * np.max(Q[observation, :]) - Q[state, action])
        
        state = observation
