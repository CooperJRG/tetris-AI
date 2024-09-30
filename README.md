# Tetris AI

## Introduction

This project is a Tetris environment implemented in Python, designed for AI research and experimentation. The environment simulates the classic Tetris game, providing a platform for developing and testing AI agents.

The project includes implementations of three different bots to play the game:

1. **Heuristic-based Bot**: **Completed**. Uses a genetic algorithm to optimize weights for features like bumpiness, height, holes, and lines cleared.
2. **Deep Q-Network (DQN) Bot**: **Completed**. Utilizes a DQN that uses heuristic values as its state.
3. **Sight-based DQN Bot**: Employs a DQN that uses the Tetris board (visual representation) as its state.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Tetris Game Manually](#running-the-tetris-game-manually)
  - [Running the Heuristic-based Bot](#running-the-heuristic-based-bot)
  - [Running the DQN Bot](#running-the-dqn-bot)
- [Environment Details](#environment-details)
- [AI Bots](#ai-bots)
  - [Heuristic-based Bot](#heuristic-based-bot)
  - [Deep Q-Network Bot](#deep-q-network-bot)
  - [Sight-based DQN Bot](#sight-based-dqn-bot)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Custom Tetris Environment**: A fully functional Tetris game environment built with Python and OpenCV.
- **OpenAI Gym Integration**: The environment is compatible with OpenAI's Gym interface, facilitating the development of reinforcement learning agents.
- **Visual Rendering**: The game includes a visual interface using OpenCV, displaying the game grid, current piece, and score panel.
- **Statistical Tracking**: Keeps track of score, high score, total lines cleared, Tetris count, and other game statistics.
- **AI Bots**: Implementation of a heuristic-based bot using a genetic algorithm and a DQN-based bot using heuristic values as state.

## Installation

### Prerequisites

- Python 3.x
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/) (`cv2`)
- [NumPy](https://numpy.org/)
- [Gym](https://www.gymlibrary.ml/)

### Installation Steps

1. **Clone the repository**:

   ```bash
   git clone https://github.com/cooperjrg/tetris-ai.git
   cd tetris-ai
   ```

2. **Create a virtual environment (optional but recommended)**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   If there is no `requirements.txt`, you can install the required packages directly:

   ```bash
   pip install torch opencv-python numpy gym
   ```

## Usage

### Running the Tetris Game Manually

You can play the Tetris game manually using keyboard controls.

```bash
python tetris_env.py
```

**Controls**:

- `A` - Move left
- `D` - Move right
- `S` - Move down
- `W` - Rotate piece
- `Space` - Drop piece
- `Q` - Quit game

### Running the Heuristic-based Bot

The heuristic-based bot uses a genetic algorithm to optimize weights for different heuristics that evaluate board states. To run the bot:

1. **Ensure the `game.py` and `genetic_algorithm.py` files are in the same directory.**

2. **Run the genetic algorithm to train the bot**:

   ```bash
   python genetic_algorithm.py
   ```

   This will start the training process, where the bot evolves over generations to find the best weights.

3. **Run the bot with the best-found heuristic weights**:

   The `genetic_algorithm.py` script, after training, will use the best heuristic found to run the Tetris bot without any limits. The bot will play the game automatically using the optimized weights.

### Running the DQN Bot

The DQN bot uses a Deep Q-Network that takes heuristic values (lines cleared, bumpiness, holes, and height) as its state representation. To run the DQN bot:

1. **Ensure the `game.py` and `dqn_agent.py` files are in the same directory.**

2. **Train the DQN agent**:

   ```bash
   python dqn_agent.py
   ```

   This script will start the training process for the DQN agent. It will:

   - Initialize the DQN agent and target network.
   - Interact with the Tetris environment to collect experiences.
   - Train the agent using experience replay and periodically update the target network.
   - Save the trained model at intervals and at the end of training.

3. **Monitor Training Progress**:

   - The script prints out metrics every 10 episodes, including average reward, average loss, and the exploration rate (epsilon).
   - Models are saved every 50 episodes in the `saved_models` directory.

**Note**: Adjust hyperparameters like the number of episodes, learning rate, batch size, etc., in the `dqn_agent.py` script to fine-tune the training process.

## Environment Details

- **Grid Dimensions**: 10 columns (width) x 20 rows (height)
- **Hidden Rows**: 2 (for piece spawning)
- **Pieces**: Standard Tetris pieces (I, O, T, S, Z, J, L)
- **Scoring**:
  - Single line clear: 100 points
  - Double line clear: 300 points
  - Triple line clear: 500 points
  - Tetris (four lines): 800 points
- **Game Statistics**: Tracks score, high score, total lines cleared, Tetris count, and game overs.

## AI Bots

### Heuristic-based Bot

**Status**: Completed

This bot uses a genetic algorithm to optimize the weights for various heuristics that determine the quality of a board state:

- **Bumpiness**: The difference in heights between adjacent columns.
- **Height**: The cumulative height of the columns.
- **Holes**: Empty spaces beneath blocks in columns.
- **Lines Cleared**: The number of complete lines removed.

#### How It Works

1. **Initialization**: A population of chromosomes (sets of heuristic weights) is randomly generated.

2. **Evaluation**: Each chromosome's fitness is evaluated by running simulations of the Tetris game using the corresponding heuristic weights.

3. **Selection**: Chromosomes are selected for reproduction based on their fitness, using tournament selection.

4. **Crossover and Mutation**: Selected chromosomes undergo crossover and mutation to produce new offspring.

5. **Iteration**: Steps 2–4 are repeated for a specified number of generations.

6. **Result**: The best-performing chromosome (heuristic weights) is used by the bot to play Tetris.

   
### Deep Q-Network Bot

**Status**: Completed

This bot implements a Deep Q-Network (DQN) that takes heuristic values as its state representation. The DQN learns to select actions that maximize the expected reward over time.

#### How It Works

1. **State Representation**: The agent uses a state representation consisting of:

   - **Lines Cleared**
   - **Bumpiness**
   - **Holes**
   - **Height**

2. **Action Selection**:

   - **Exploration**: With probability `epsilon`, the agent selects a random action.
   - **Exploitation**: Otherwise, it evaluates all possible actions using the current policy network and selects the action with the highest expected value.

3. **Experience Replay**:

   - Experiences are stored in a replay memory buffer.
   - The agent samples mini-batches from this buffer to perform gradient descent updates, breaking correlation between sequential data.

4. **Training Loop**:

   - The agent interacts with the environment, collects experiences, and updates the policy network.
   - The target network is updated periodically to stabilize learning.

5. **Model Saving**:

   - The model is saved at regular intervals and at the end of training for later use or evaluation.


#### Important Hyperparameters

- **Number of Episodes**: Controls how many games the agent will play during training.
- **Learning Rate**: Determines the step size during optimization.
- **Batch Size**: Number of samples per training batch.
- **Gamma (Discount Factor)**: How much the agent values future rewards.
- **Epsilon**: Exploration rate for the epsilon-greedy policy.

### Sight-based DQN Bot

**Status**: In Development

This bot uses the visual representation of the Tetris board as its state input to a convolutional neural network (CNN)-based DQN. The agent learns directly from the raw pixel data of the game grid.


## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- Inspired by the classic Tetris game and various AI implementations.
- Thanks to the open-source community for providing tools and libraries.

# Code Overview

The Tetris environment is implemented in Python and consists of the following main components:

- **Block Class**: Represents individual Tetris pieces (tetrominoes), handling their shapes, movements, rotations, and collision detection.
- **TetrisEnv Class**: Inherits from `gym.Env` and encapsulates the game logic, including the grid, scoring, piece generation, rendering, and game state updates.
- **Genetic Algorithm**: Implements the heuristic-based bot using a genetic algorithm to optimize heuristic weights.
- **DQN Agent**: Implements the Deep Q-Network agent that uses heuristic values as the state input.

## Key Functions and Methods

### DQN Agent

- **DQNAgent Class**: Neural network model representing the agent.

  - **Initialization**:

    ```python
    agent = DQNAgent(input_dim=4, output_dim=1)
    ```

    - `input_dim`: Size of the state vector (heuristic values).
    - `output_dim`: Size of the output, typically the number of possible actions.

  - **Forward Pass**:

    ```python
    def forward(self, x):
        # Defines the forward pass through the network layers
    ```

- **Training Loop**:

  - **Experience Replay Memory**: Uses a deque to store experiences.
  - **Epsilon-Greedy Policy**: Balances exploration and exploitation.
  - **Optimization**: Uses Adam optimizer and MSE loss function.
  - **Target Network**: A separate network to compute target Q-values for stability.

- **Hyperparameters**:

  - `epsilon`, `epsilon_min`, `epsilon_decay`: Control exploration.
  - `gamma`: Discount factor.
  - `batch_size`: Size of training batches.
  - `num_episodes`: Number of training episodes.
  - `target_update`: Frequency of target network updates.

### Tetris Environment

- **TetrisEnv.step(action, render=False)**: Advances the game state by one step based on the provided action. Returns the next state, reward, done flag, and additional info.
- **TetrisEnv.get_state()**: Retrieves the current state representation for the agent, including lines cleared, bumpiness, holes, and height.
- **TetrisEnv.handle_full_rows()**: Checks for and handles the clearing of full rows, updating the score and statistics accordingly.
- **TetrisEnv.get_possible_actions()**: Generates all possible valid placements for the current piece, useful for heuristic-based and search algorithms.

## Extending the Environment

The environment is designed to be extensible for AI research:

- **State Representations**: You can modify or extend the `get_state()` method to provide different state representations to your agents.
- **Reward Function**: Adjust the reward calculations in the `step()` method to align with your training objectives.
- **Action Space**: The action space includes moving left, right, down, rotating, and dropping the piece. You can redefine or limit the action space as needed.

## Future Plans

- **Develop Sight-based DQN Agent**: Create a deep learning model that learns to play Tetris using the visual representation of the board as input.
- **Optimize Performance**: Improve the efficiency of the environment and algorithms for faster training and evaluation.

Feel free to explore the code, experiment with different agents, and contribute improvements!

# Additional Information

## Parameter Configuration for the DQN Agent

You can adjust the parameters of the DQN agent to influence the training process:

- **Epsilon Parameters**:

  - `epsilon`: Initial exploration rate.
  - `epsilon_min`: Minimum exploration rate.
  - `epsilon_decay`: Decay rate for epsilon after each episode.

- **Learning Parameters**:

  - `gamma`: Discount factor for future rewards.
  - `learning_rate`: Step size during optimization.
  - `batch_size`: Number of experiences sampled from memory for each training step.

- **Training Parameters**:

  - `num_episodes`: Total number of episodes for training.
  - `replay_start_size`: Minimum memory size before training starts.
  - `target_update`: Number of episodes between updates of the target network.

These parameters are defined in the `dqn_agent.py` script:

```python
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
gamma = 0.99
learning_rate = 0.001
batch_size = 32
num_episodes = 1000
target_update = 10
```

Adjusting these values can help the agent converge faster or explore the solution space more thoroughly.

## Visualization

The Tetris environment includes visualization using OpenCV, allowing you to watch the bot play in real-time. This can be helpful for debugging and understanding how the bot makes decisions based on the heuristic evaluations.

## Contact

For any questions or suggestions, please contact cjrgilkey@gmail.com.

---

Enjoy experimenting with the Tetris AI Environment!
