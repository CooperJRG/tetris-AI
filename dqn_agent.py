import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import game  # Ensure this module provides the necessary environment
import os
from collections import deque

class DQNAgent(nn.Module):
    def __init__(self, input_dim=4, output_dim=1):
        """
        Deep Q-Network Agent with Fully Connected Layers.

        Args:
            input_dim (int): Dimension of the input state vector.
            output_dim (int): Dimension of the output. Typically, the number of possible actions.
            dropout_p (float): Dropout probability for regularization.
        """
        super(DQNAgent, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Define Fully Connected Layers
        self.fc1 = nn.Linear(self.input_dim, 128)  # First hidden layer
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(128, 256)  # Second hidden layer
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(256, 128)  # Third hidden layer
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(128, self.output_dim)  # Output layer

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim].
        """
        x = self.relu1(self.fc1(x))

        x = self.relu2(self.fc2(x))

        x = self.relu3(self.fc3(x))

        x = self.fc4(x)
        return x.squeeze(-1)  # Ensure output shape is [batch_size] or [batch_size, output_dim]

# Hyperparameters and Initialization
if torch.backends.mps.is_available():
    print("MPS backend is available.")
    device = torch.device('mps')
elif torch.cuda.is_available():
    print("CUDA backend is available.")
    device = torch.device('cuda')
else:
    print("No GPU backend available. Using CPU.")
    device = torch.device('cpu')

# Initialize agent, optimizer, and loss function
agent = DQNAgent().to(device)
optimizer = optim.Adam(agent.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Hyperparameters
num_episodes = 1_000
epsilon = 1.0  # Start with high exploration
epsilon_min = 0.0000
epsilon_decay = 0.995
gamma = 1.05  # Discount factor
batch_size = 32
memory = deque(maxlen=5000)  # Replay memory
replay_start_size = 1000  # Minimum memory size before training starts
target_update = 10  # Episodes between target network updates

# Tracking metrics
episode_rewards = []
losses = []

# Create a directory to save models
os.makedirs('saved_models', exist_ok=True)

# Initialize environment
env = game.TetrisEnv()

# Initialize Target Network
target_agent = DQNAgent().to(device)
target_agent.load_state_dict(agent.state_dict())
target_agent.eval()  # Set to evaluation mode

for episode in range(1, num_episodes + 1):
    state = env.reset()
    env.initialize_game()
    done = False
    total_reward = 0
    episode_loss = 0
    step = 0

    while not done and step < 10_000:
        possible_actions = env.get_possible_actions()
        if not possible_actions:
            print("No possible actions.")
            done = True
            break

        if random.random() < epsilon:
            # Explore: choose a random action
            action = random.choice(possible_actions)
        else:
            # Exploit: choose the best action
            state_inputs = []
            for action_option in possible_actions:
                # Assuming 'next_state' is a list or tuple of four features
                state_vector = action_option['next_state']
                state_inputs.append(state_vector)

            state_inputs = torch.tensor(state_inputs, dtype=torch.float32, device=device)
            with torch.no_grad():
                q_values = agent(state_inputs)  # Shape: [num_possible_actions]

            # Calculate total values (immediate reward + discounted future reward)
            rewards = torch.tensor(
                [action_option['reward'] for action_option in possible_actions],
                dtype=torch.float32,
                device=device
            )
            dones_tensor = torch.tensor(
                [action_option['done'] for action_option in possible_actions],
                dtype=torch.float32,
                device=device
            )
            total_values = rewards + gamma * target_agent(state_inputs) * (1 - dones_tensor)

            # Choose the action with the highest total value
            action_index = torch.argmax(total_values).item()
            action = possible_actions[action_index]

        # Retrieve the selected action details
        selected_action_sequence = action['action_sequence']
        reward = action['reward']
        done = action['done']

        # Perform the selected action in the environment
        next_state, reward, done, _ = env.step(selected_action_sequence, render=True)
        total_reward += reward

        # Store the experience in memory
        memory.append((state, reward, next_state, done))
        if len(memory) > memory.maxlen:
            memory.pop()  # Remove the oldest experience if memory is full

        # Update the state
        state = next_state

        # Train the agent if the memory is large enough
        if len(memory) >= replay_start_size and len(memory) >= batch_size:
            # Sample a batch of experiences
            batch = random.sample(memory, batch_size)

            # Prepare the batch
            states = torch.tensor(
                [exp[0] for exp in batch],
                dtype=torch.float32,
                device=device
            )  # Shape: [batch_size, 4]

            rewards_batch = torch.tensor(
                [exp[1] for exp in batch],
                dtype=torch.float32,
                device=device
            )  # Shape: [batch_size]

            next_states = torch.tensor(
                [exp[2] for exp in batch],
                dtype=torch.float32,
                device=device
            )  # Shape: [batch_size, 4]

            dones_batch = torch.tensor(
                [exp[3] for exp in batch],
                dtype=torch.float32,
                device=device
            )  # Shape: [batch_size]

            # Calculate the target Q-values using the target network
            with torch.no_grad():
                next_q_values = target_agent(next_states)  # Shape: [batch_size]
                target_q_values = rewards_batch + gamma * next_q_values * (1 - dones_batch)

            # Compute current Q-values
            current_q_values = agent(states)  # Shape: [batch_size]

            # Compute the loss
            loss = criterion(current_q_values, target_q_values)
            episode_loss += loss.item()

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        step += 1  # Increment step counter

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
        epsilon = max(epsilon, epsilon_min)

    # Record metrics
    episode_rewards.append(total_reward)
    if step > 0:
        losses.append(episode_loss / step)  # Average loss per step
    else:
        losses.append(0)

    # Update the target network periodically
    if episode % target_update == 0:
        target_agent.load_state_dict(agent.state_dict())
        print(f"Target network updated at episode {episode}")

    # Print metrics every 10 episodes
    if episode % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        avg_loss = np.mean(losses[-10:])
        print(f"Episode {episode}/{num_episodes}, "
              f"Average Reward: {avg_reward:.2f}, "
              f"Average Loss: {avg_loss:.4f}, "
              f"Epsilon: {epsilon:.4f}")

    # Save the model every 50 episodes
    if episode % 50 == 0:
        model_path = f'saved_models/dqn_tetris_episode_{episode}.pt'
        torch.save(agent.state_dict(), model_path)
        print(f"Model saved to {model_path}")

# Save the final model
model_path = 'saved_models/dqn_tetris_final.pt'
torch.save(agent.state_dict(), model_path)
print(f"Final model saved to {model_path}")
