"""
Following a tutorial by deeplizard (https://www.youtube.com/watch?v=QK_PP_2KgGE)
"""

import numpy as np
import gymnasium as gym 
import random 
import time 
from IPython.display import clear_output

# change render_mode to "human" to see the game in action
env = gym.make("FrozenLake-v1", render_mode="rgb_array")

action_space_size = env.action_space.n
state_space_size = env.observation_space.n

print(env.action_space, env.observation_space)

q_table = np.zeros((state_space_size, action_space_size))
print(q_table)

num_episodes = 10000 # number of times we run the environment from the beginning
max_steps_per_episode = 100 # max number of steps allowed for each episode 

# Q-learning algorithm parameters
learning_rate = 0.1 # alpha
discount_rate = 0.99 # gamma

exploration_rate = 1 # epsilon
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewards_all_episodes = []

# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset() # reset the environment to a new, random state 

    terminated = False 
    rewards_current_episode = 0 # reset rewards for new episode 

    for step in range(max_steps_per_episode):

        if type(state) is tuple:
            state = state[0]

        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0,1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:]) # take best possible action in current state
        else:
            action = env.action_space.sample() # else explore randomly

        new_state, reward, terminated, truncated, info = env.step(action) # take new action

        # Update Q-table for Q(s,a)a
        # New table entry is weighted sum of old value and new learned value 
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state 
        rewards_current_episode += reward

        if terminated == True:
            break

    # Exploration rate exploration_decay_rate
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

    rewards_all_episodes.append(rewards_current_episode)

# Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes/1000)
count = 1000 
print("\n\n****Average reward per thousand episodes****\n")
for r in rewards_per_thousand_episodes:
    print(count, ":", str(sum(r/1000)))
    count += 1000

print("\n\n****Q-table****\n")
print(q_table)

# Watch our agent play Frozen Lake by playing the best action 
# from each state according to the Q-table 
for episode in range(3):
    state = env.reset()
    terminated = False 
    print("****EPISODE ", episode+1, "****\n\n\n\n")
    time.sleep(1)

    for step in range(max_steps_per_episode):
        clear_output(wait=True)
        env.render()
        time.sleep(0.3)
        
        if type(state) is tuple:
            state = state[0]

        action = np.argmax(q_table[state,:])
        new_state, reward, terminated, truncated, info = env.step(action)

        if terminated:
            clear_output(wait=True)
            env.render()
            if reward == 1:
                print("****You reached the goal!****")
                time.sleep(3)
            else:
                print("****You fell through a hole!****")
                time.sleep(3)
            clear_output(wait=True)
            break

        state = new_state

env.close()
