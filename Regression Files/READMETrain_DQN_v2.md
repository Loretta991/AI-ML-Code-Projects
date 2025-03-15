
    <h1 style="color:#2E3A87; text-align:center;">Project Analysis</h1>
    <p style="color:#1F6B88; font-size:20px;">This project contains detailed analysis using Jupyter Notebooks. The following sections describe the steps, code implementations, and results.</p>
    <hr style="border: 2px solid #1F6B88;">
    <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;"># HW6-DQN
</div>

<div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">In this exercise you are going to implement your first keras-rl agent based on the **Acrobot** environment (https://www.gymlibrary.dev/environments/classic_control/acrobot/) <br />
The goal of this environment is to maneuver the robot arm upwards above the line with as little steps as possible</div>

<div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">**TASK: Import necessary libraries** <br /></div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            !pip install tensorflow==2.8.0
!pip install keras-rl2
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            ##TODO
!pip install gym

            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">**TASK: Create the environment** <br />
The name is: *Acrobot-v1*</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            ##TODO
import gym
import numpy as np

# Create the Acrobot environment
env = gym.make('Acrobot-v1')

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            num_actions = env.action_space.n
num_observations = env.observation_space.shape
print(f"Action Space: {env.action_space.n}")
print(f"Observation Space: {num_observations}")

assert num_actions == 3 and num_observations == (6,) , "Wrong environment!"
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">**TASK: Create the Neural Network for your Deep-Q-Agent** <br />
Take a look at the size of the action space and the size of the observation space.
You are free to chose any architecture you want! <br />
Hint: It already works with three layers, each having 64 neurons.</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            ## TODO
!pip install torch

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #Create neural networks
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define input size based on observation space and output size based on action space
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

# Create the Q-network
q_network = QNetwork(input_size, output_size)

            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">**TASK: Initialize the circular buffer**<br />
Make sure you set the limit appropriately (50000 works well)</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            ## TODO
from collections import deque

# Initialize the replay memory (circular buffer)
replay_memory_limit = 50000
replay_memory = deque(maxlen=replay_memory_limit)

            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">**TASK: Use the epsilon greedy action selection strategy with *decaying* epsilon**</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            ## TODO
# Hyperparameters
initial_epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.995

# Initialize epsilon
epsilon = initial_epsilon

# Epsilon-greedy action selection
def select_action(state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            q_values = q_network(state_tensor)
            return q_values.argmax().item()  # Exploit

num_episodes = 1000  # Define the number of episodes
print_interval = 10  # Print status every 10 episodes

# During the training loop
for episode in range(num_episodes):
    state = env.reset()  # Reset the environment for a new episode
    done = False  # Initialize the "done" flag

    while not done:
        action = select_action(state)
        next_state, reward, done, _ = env.step(action)  # Get next state and reward
        # Update the Q-values, experience replay, etc.
        state = next_state

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Print status every print_interval episodes
    if (episode + 1) % print_interval == 0:
        print(f"Episode {episode+1}, Epsilon: {epsilon:.4f}")

            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">**TASK: Create the DQNAgent** <br />
Feel free to play with the nb_steps_warump, target_model_update, batch_size and gamma parameters. <br />
Hint:<br />
You can try *nb_steps_warmup*=1000, *target_model_update*=1000, *batch_size*=32 and *gamma*=0.99 as a first guess</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.agents.dqn import DQNAgent
from keras.models import Sequential
from keras.layers import Dense

# Create the Q-network model
input_size = 6
output_size = 3
q_network = Sequential([
    Dense(64, activation='relu', input_shape=(input_size,)),
    Dense(64, activation='relu'),
    Dense(output_size, activation='linear')  # Linear activation for Q-values
])

# Parameters for DQNAgent
nb_steps_warmup = 1000
target_model_update = 1000
batch_size = 32
gamma = 0.99

# Define policy and memory
policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=replay_memory_limit, window_length=1)

# Create the DQNAgent using the q_network model
dqn_agent = DQNAgent(model=q_network, nb_actions=output_size, policy=policy, memory=memory,
                     nb_steps_warmup=nb_steps_warmup, target_model_update=target_model_update,
                     batch_size=batch_size, gamma=gamma)

            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">**TASK: Compile the model** <br />
Feel free to explore the effects of different optimizers and learning rates.
You can try Adam with a learning rate of 1e-3 as a first guess</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            ## TODO
from tensorflow.keras.optimizers import Adam

# Define optimizer and learning rate
optimizer = Adam(learning_rate=1e-3)

# Compile the DQN model
dqn_agent.compile(optimizer=optimizer, metrics=['mae'])

            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">**TASK: Fit the model** <br />
150,000 steps should be a very good starting point</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            ## TODO
from tensorflow.keras.optimizers import Adam

# Define optimizer and learning rate
optimizer = Adam(learning_rate=1e-3)

# Compile the DQN model
dqn_agent.compile(optimizer=optimizer, metrics=['mae'])

            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">**TASK: Evaluate the model**</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            ## TODO
## TODO
# Train the DQN agent
history = dqn_agent.fit(env, nb_steps=150000, visualize=False, verbose=1)


            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # Assuming you've already created the DQNAgent and defined the Q-network as shown earlier

# Import necessary libraries
import gym

# Create the environment
env = gym.make('Acrobot-v1')

# Define input size and output size
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

# Train the DQN agent
history = dqn_agent.fit(env, nb_steps=150000, visualize=False, verbose=1)

            </pre>


            
    <hr style="border: 2px solid #1F6B88;">
    <h3 style="color:#2E3A87;">Analysis and Results:</h3>
    <p style="color:#1F6B88; font-size:18px;">The notebook contains various steps for analyzing the dataset. Below you can see the results and analysis conducted during the notebook execution.</p>
    