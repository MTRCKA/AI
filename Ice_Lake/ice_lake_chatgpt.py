import gym
import numpy as np

# Create the environment
env = gym.make('FrozenLake-v1', is_slippery=False)

# Set hyperparameters
num_episodes = 100000
max_steps_per_episode = 100
learning_rate = 0.8
discount_factor = 0.95
epsilon = 0.2

# Initialize Q-table with zeros
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Define epsilon-greedy policy
def epsilon_greedy_policy(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        # Random action
        return env.action_space.sample()
    else:
        # Greedy action
        return np.argmax(Q[state, :])

# Q-learning algorithm
for i in range(num_episodes):
    state = env.reset()
    for t in range(max_steps_per_episode):
        # Choose action based on epsilon-greedy policy
        action = epsilon_greedy_policy(state, epsilon)
        
        # Take action and observe new state and reward
        new_state, reward, done, info = env.step(action)
        
        # Update Q-table
        Q[state, action] += learning_rate * (reward + discount_factor * np.max(Q[new_state, :]) - Q[state, action])
        
        state = new_state
        if done:
            break

# Evaluate policy
total_reward = 0
num_episodes = 1000
for i in range(num_episodes):
    state = env.reset()
    for t in range(max_steps_per_episode):
        action = np.argmax(Q[state, :])
        state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
print("Average reward:", total_reward / num_episodes)


state = env.reset()
for t in range(max_steps_per_episode):
    env.render()
    action = np.argmax(Q[state, :])
    state, reward, done, info = env.step(action)
    if done:
        break
env.render()