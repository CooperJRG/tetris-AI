import cv2
import numpy as np
import random
import time
import gym
import copy
from collections import deque

# Add a blank line after the import statement

# Game Parameters
GRID_HEIGHT = 20
GRID_WIDTH = 10
HIDDEN_ROWS = 2
BLOCK_SIZE = 30
GRID_PADDING = 2  # Padding between blocks for cleaner look
SCORE_PANEL_WIDTH = 250  # Width of the side panel for score

# Define Colors
COLORS = {
    '\x00': (40, 40, 40),        # Dark gray for empty spaces (background)
    'I': (85, 205, 252),          # Soft Sky Blue
    'O': (255, 228, 181),         # Moccasin
    'T': (168, 134, 255),         # Light Lavender
    'S': (144, 238, 144),         # Light Green
    'Z': (255, 160, 122),         # Light Salmon
    'J': (173, 216, 230),         # Light Blue
    'L': (255, 182, 102),         # Light Orange
}

# Define Tetris blocks
class Block:
    def __init__(self, shape_type):
        block_shapes = {
            'I': np.array([[0, 0, 0, 0],
                          [1, 1, 1, 1],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]]),
            'O': np.array([[0, 1, 1, 0],
                          [0, 1, 1, 0],
                          [0, 0, 0, 0]]),
            'T': np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 0, 0]]),
            'S': np.array([[0, 1, 1],
                          [1, 1, 0],
                          [0, 0, 0]]),
            'Z': np.array([[1, 1, 0],
                          [0, 1, 1],
                          [0, 0, 0]]),
            'J': np.array([[1, 0, 0],
                          [1, 1, 1],
                          [0, 0, 0]]),
            'L': np.array([[0, 0, 1],
                          [1, 1, 1],
                          [0, 0, 0]])
        }
        self.shape_type = shape_type
        self.shape = block_shapes[shape_type]
        self.y = 0
        self.x = GRID_WIDTH // 2 - self.shape.shape[1] // 2
        self.position = (self.y, self.x) 
        
    def copy(self):
        new_block = Block(self.shape_type)
        new_block.shape = self.shape.copy()
        new_block.y = self.y
        new_block.x = self.x
        new_block.position = self.position
        return new_block
    
    def hash(self):
        occupied_positions = []
        for dy in range(self.shape.shape[0]):
            for dx in range(self.shape.shape[1]):
                if self.shape[dy, dx]:
                    grid_x = self.x + dx
                    grid_y = self.y + dy
                    occupied_positions.append((grid_x, grid_y))
        return frozenset(occupied_positions)
        
    def move(self, direction, placed_blocks_grid):
        if direction == 'down':
            self.y += 1
        elif direction == 'drop':
            while not self.detect_collision(placed_blocks_grid):
                self.y += 1
            self.y -= 1
        elif direction == 'left':
            self.x -= 1
        elif direction == 'right':
            self.x += 1
        elif direction == 'up':
            self.x -= 1
        elif direction == 'rotate':
            return self.rotate(placed_blocks_grid)
        if self.detect_collision(placed_blocks_grid):
            self.y, self.x = self.position
            return False
        else:
            self.position = (self.y, self.x)
            return True
        
    def rotate(self, placed_blocks_grid):
        old_shape = self.shape.copy()
        old_position = self.position
        self.shape = np.rot90(self.shape)
        
        # Check for collision after rotation
        if self.detect_collision(placed_blocks_grid):
            # Attempt wall kicks without moving upwards
            # Define possible kicks: right, left, down
            kicks = [(0, 1), (0, -1), (1, 0)]
            for dy, dx in kicks:
                self.y += dy
                self.x += dx
                if not self.detect_collision(placed_blocks_grid):
                    self.position = (self.y, self.x)
                    return True
                # Revert position if collision persists
                self.y -= dy
                self.x -= dx
            # If all kicks fail, undo rotation
            self.shape = old_shape
            self.position = old_position
            return False
        else:
            self.position = (self.y, self.x)
            return True

    def draw(self, grid):
        for y in range(self.shape.shape[0]):
            for x in range(self.shape.shape[1]):
                if self.shape[y, x] == 1:
                    grid[self.y + y, self.x + x] = ord(self.shape_type)
    
    def detect_collision(self, placed_blocks_grid):
        for y in range(self.shape.shape[0]):
            for x in range(self.shape.shape[1]):
                if self.shape[y, x] == 1:
                    grid_x = self.x + x
                    grid_y = self.y + y
                    # Check if the block is out of bounds
                    if grid_x < 0 or grid_x >= GRID_WIDTH or grid_y >= GRID_HEIGHT + HIDDEN_ROWS or grid_y < 0:
                        return True
                    # Check for collision with placed blocks
                    if placed_blocks_grid[grid_y, grid_x] != 0:
                        return True
        return False
    
    def get_current_position(self):
        return self.position
    
    def get_current_shape(self):
        return self.shape


class TetrisEnv(gym.Env):
    def __init__(self):
        self.grid = np.zeros((GRID_HEIGHT + HIDDEN_ROWS, GRID_WIDTH), dtype=int)
        self.placed_blocks_grid = np.zeros((GRID_HEIGHT + HIDDEN_ROWS, GRID_WIDTH), dtype=int)
        self.bag = []
        self.speed_factor = 1
        self.score = 0
        self.frame_count = 0
        self.fps = 60
        self.bool_new_piece = True
        self.piece = Block(self.new_piece())
        
        self.lines_cleared = 0
        self.high_score = 0
        self.total_lines_cleared = 0
        self.tetris_count = 0
        self.game_overs = 0
        # Adjust image size to accommodate the score panel
        self.img = np.zeros(((GRID_HEIGHT) * BLOCK_SIZE, GRID_WIDTH * BLOCK_SIZE + SCORE_PANEL_WIDTH, 3), dtype=np.uint8)
        
    def reset(self):
        # Reset the game to the initial state and return the initial state
        self.grid = np.zeros((GRID_HEIGHT + HIDDEN_ROWS, GRID_WIDTH), dtype=int)
        self.placed_blocks_grid = np.zeros((GRID_HEIGHT + HIDDEN_ROWS, GRID_WIDTH), dtype=int)
        self.piece = Block(self.new_piece())
        self.bag = []
        self.speed_factor = 1
        self.score = 0
        self.tetris_count = 0
        self.total_lines_cleared = 0
        self.frame_count = 0
        return self.get_state()
        
    def step(self, action, render=False):
        # Take a step in the environment based on the action
        score = self.score
        self.handle_input(self.piece, action, render)
        placed_height = self.piece.y
        self.piece = self.handle_falling(self.piece)
        self.frame_count += 1
        # Check if the game is over
        if not self.piece:
            # If unable to spawn a new piece, the game is over
            done = True
            self.game_overs += 1
        else:
            done = False
            self.update_grid()
        done_reward = -100 if done else 0
        reward = self.score - score + done_reward + 5
        #reward = self.lines_cleared**2 * GRID_WIDTH + 1 + done_reward
        return self.get_state(), reward, done, {}
    
    def step_heuristic(self, action):
        # Take a step in the environment based on the action
        self.handle_input(self.piece, action)
        
        self.piece = self.handle_falling(self.piece)
        # Check if the game is over
        if not self.piece:
            # If unable to spawn a new piece, the game is over
            done = True
            self.game_overs += 1
        else:
            done = False
            self.update_grid()
        self.render()
        
        return done
    
    def get_height(self):
        # take the sum of the heights of each column
        heights = []
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT + HIDDEN_ROWS):
                if self.placed_blocks_grid[y, x] != 0:
                    heights.append(GRID_HEIGHT - y)
                    break
        return sum(heights)
    
    def get_bumpiness(self):
        # take the sum of the absolute differences between adjacent columns
        heights = []
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT + HIDDEN_ROWS):
                if self.placed_blocks_grid[y, x] != 0:
                    heights.append(GRID_HEIGHT - y)
                    break
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return bumpiness
    
    def get_holes(self):
        # take the sum of the number of holes in each column (holes are empty spaces with a block above them in the same column)
        holes = 0
        for x in range(GRID_WIDTH):
            hole = False
            for y in range(GRID_HEIGHT + HIDDEN_ROWS):
                if self.placed_blocks_grid[y, x] != 0:
                    hole = True
                if hole and self.placed_blocks_grid[y, x] == 0:
                    holes += 1
        return holes
    
    def get_state_sight(self):
        """
        Generate a comprehensive state representation for the agent.

        The state includes:
        - Placed blocks grid (excluding the current piece).
        - Current piece grid (shows the current piece's position).
        - Current piece type (as a one-hot encoded vector or normalized index).

        Returns:
            state (numpy.ndarray): A multi-channel tensor representing the game state.
        """
        # Get the grid of placed blocks, excluding the hidden rows
        placed_blocks = (self.placed_blocks_grid[HIDDEN_ROWS:] != 0).astype(np.float32)

        # Map piece types to indices
        piece_type_dict = {'I': 0, 'O': 1, 'T': 2, 'S': 3, 'Z': 4, 'J': 5, 'L': 6}
        if self.piece == False:
            piece_type_index = -1
        else:
            piece_type_index = piece_type_dict[self.piece.shape_type]

        # One-hot encode the piece type (optional)
        piece_type_one_hot = np.zeros(7, dtype=np.float32)
        piece_type_one_hot[piece_type_index] = 1.0

        # Alternatively, create a grid filled with the normalized piece type index
        piece_type_grid = np.full((GRID_HEIGHT, GRID_WIDTH), piece_type_index / 6.0, dtype=np.float32)

        # Stack the grids to create the state tensor
        state = np.stack([placed_blocks, piece_type_grid], axis=0)
        # Shape: (2, GRID_HEIGHT, GRID_WIDTH)

        return state
    
    def get_state(self):
        return [self.lines_cleared, self.get_bumpiness(), self.get_holes(), self.get_height()]

        
    def set_preset_board(self):
        # Clear any existing blocks
        self.placed_blocks_grid = np.zeros((GRID_HEIGHT + HIDDEN_ROWS, GRID_WIDTH), dtype=int)
        # Add blocks to create a predefined pattern
        self.placed_blocks_grid[GRID_HEIGHT + HIDDEN_ROWS - 1, (0, 1, 3, 7)] = ord('O')
        self.placed_blocks_grid[GRID_HEIGHT + HIDDEN_ROWS - 2, (1, 2, 3, 7, 8)] = ord('I')
        self.placed_blocks_grid[GRID_HEIGHT + HIDDEN_ROWS - 3, (0, 1, 2, 3, 7, 8)] = ord('T')
        self.placed_blocks_grid[GRID_HEIGHT + HIDDEN_ROWS - 4, (0, 3, 7, 8, 9)] = ord('S')
        self.placed_blocks_grid[GRID_HEIGHT + HIDDEN_ROWS - 5, (6, 7, 8, 9)] = ord('J')
        
    
    def draw_grid(self, img):
        # Fill the background
        img[:] = (40, 40, 40)  # Dark gray background

        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                block_val = self.placed_blocks_grid[y + HIDDEN_ROWS, x]
                if block_val != 0:
                    color = COLORS[chr(block_val)]
                    top_left = (x * BLOCK_SIZE + GRID_PADDING, y * BLOCK_SIZE + GRID_PADDING)
                    bottom_right = ((x + 1) * BLOCK_SIZE - GRID_PADDING, (y + 1) * BLOCK_SIZE - GRID_PADDING)
                    cv2.rectangle(img, top_left, bottom_right, color, -1)
                    
                    # Add a border around each block
                    cv2.rectangle(img, top_left, bottom_right, (50, 50, 50), 2)  # Darker border for cleaner look

        # Draw the current piece
        self.piece.draw(self.grid)
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                block_val = self.grid[y + HIDDEN_ROWS, x]
                if block_val != 0:
                    color = COLORS[chr(block_val)]
                    top_left = (x * BLOCK_SIZE + GRID_PADDING, y * BLOCK_SIZE + GRID_PADDING)
                    bottom_right = ((x + 1) * BLOCK_SIZE - GRID_PADDING, (y + 1) * BLOCK_SIZE - GRID_PADDING)
                    cv2.rectangle(img, top_left, bottom_right, color, -1)
                    
                    # Add a border around each block
                    cv2.rectangle(img, top_left, bottom_right, (50, 50, 50), 2)  # Darker border

        # Reset the grid for the next frame
        self.grid[y + HIDDEN_ROWS, x] = 0  # This line might cause an error; consider removing or adjusting

    def new_piece(self):
        if not self.bag:
            self.bag = ['I', 'O', 'T', 'S', 'Z', 'J', 'L']
        piece = random.choice(self.bag)
        self.bag.remove(piece)
        return piece
    
    def initialize_game(self):
        self.piece = Block(self.new_piece())
        cv2.namedWindow('Tetris')
    
    def update_grid(self):
        # Clear the grid before drawing
        self.grid = np.zeros((GRID_HEIGHT + HIDDEN_ROWS, GRID_WIDTH), dtype=int)
        self.img = np.zeros((GRID_HEIGHT * BLOCK_SIZE, GRID_WIDTH * BLOCK_SIZE + SCORE_PANEL_WIDTH, 3), dtype=np.uint8)
        self.draw_grid(self.img)
        
        # Draw the score panel
        self.draw_score_panel(self.img)
    
    def draw_score_panel(self, img):
        # Define the region for the score panel, ensuring it's wide enough for the additional 'High Score'
        panel_start_x = GRID_WIDTH * BLOCK_SIZE
        panel_end_x = GRID_WIDTH * BLOCK_SIZE + SCORE_PANEL_WIDTH

        # Fill the score panel background with a lighter gray (or you could implement a gradient for a modern look)
        cv2.rectangle(img, (panel_start_x, 0), (panel_end_x, GRID_HEIGHT * BLOCK_SIZE), (80, 80, 80), -1)  # Lighter gray

        # Add a soft border to the score panel
        cv2.rectangle(img, (panel_start_x, 0), (panel_end_x, GRID_HEIGHT * BLOCK_SIZE), (200, 200, 200), 2)

        # Display High Score
        cv2.putText(img, 'High Score:', (panel_start_x + 20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(img, str(self.high_score), (panel_start_x + 20, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 182, 102), 3, cv2.LINE_AA)  # Light Orange for High Score
        
        # Display the current score below High Score
        cv2.putText(img, 'Score:', (panel_start_x + 20, 160), cv2.FONT_HERSHEY_DUPLEX, 1, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(img, str(self.score), (panel_start_x + 20, 210), cv2.FONT_HERSHEY_DUPLEX, 1.5, (102, 205, 170), 3, cv2.LINE_AA)  # Soft Turquoise for Score
        
        # Display Total Lines Cleared
        cv2.putText(img, 'Lines:', (panel_start_x + 20, 270), cv2.FONT_HERSHEY_DUPLEX, 1, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(img, str(self.total_lines_cleared), (panel_start_x + 20, 320), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 218, 185), 3, cv2.LINE_AA)  # Peach Puff for Lines
        
        # Display Tetris Count
        cv2.putText(img, 'Tetrises:', (panel_start_x + 20, 380), cv2.FONT_HERSHEY_DUPLEX, 1, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(img, str(self.tetris_count), (panel_start_x + 20, 430), cv2.FONT_HERSHEY_DUPLEX, 1.5, (168, 134, 255), 3, cv2.LINE_AA)  # Light Lavender for Tetrises
        
        # Display Game Overs (or 'Dones') with another soft tone
        cv2.putText(img, 'Dones:', (panel_start_x + 20, 490), cv2.FONT_HERSHEY_DUPLEX, 1, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(img, str(self.game_overs), (panel_start_x + 20, 540), cv2.FONT_HERSHEY_DUPLEX, 1.5, (173, 216, 230), 3, cv2.LINE_AA)  # Light Blue for Dones




    
    def handle_input_human(self, piece):
        key = cv2.waitKey(1) & 0xFF
        self.render()
        if key == ord('a'):
            piece.move('left', self.placed_blocks_grid)
        if key == ord('d'):
            piece.move('right', self.placed_blocks_grid)
        if key == ord(' '):
            piece.move('drop', self.placed_blocks_grid)
        if key == ord('w'):
            piece.move('rotate', self.placed_blocks_grid)
        if key == ord('s'):
            piece.move('down', self.placed_blocks_grid)
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit()
        
    
    def handle_input(self, piece, action, render=False):
        if isinstance(action, str):
            piece.move(action, self.placed_blocks_grid)
        elif isinstance(action, list):
            for act in action:
                piece.move(act, self.placed_blocks_grid)
                self.update_grid()
                self.render()
        
    
    def handle_falling(self, piece, frame_count=0):
        if frame_count % int(50 * self.speed_factor) == 0:  # Forced down move every 50 frames (50/60 seconds)
            fell = piece.move('down', self.placed_blocks_grid)
            if not fell:
                piece.draw(self.placed_blocks_grid)
                self.handle_full_rows()
                # Check if the shape is spawning on an occupied space
                if np.any(self.placed_blocks_grid[0:HIDDEN_ROWS, 3:7] != 0):
                    return False
                self.piece = Block(self.new_piece())
                self.bool_new_piece = True
                return self.piece
        return self.piece
        
    def handle_full_rows(self):
        full_rows = [] 
        self.lines_cleared = 0
        for y in range(GRID_HEIGHT + HIDDEN_ROWS):
            if np.all(self.placed_blocks_grid[y] != 0):
                full_rows.append(y)
        if full_rows:
            self.placed_blocks_grid = np.delete(self.placed_blocks_grid, full_rows, axis=0)
            for _ in range(len(full_rows)):
                self.placed_blocks_grid = np.insert(self.placed_blocks_grid, 0, 0, axis=0)
            
            # Update statistics based on number of lines cleared
            self.lines_cleared = len(full_rows)
            self.total_lines_cleared += self.lines_cleared   # Update total lines cleared
            
            if self.lines_cleared == 1:
                self.score += 100
            elif self.lines_cleared == 2:
                self.score += 300
            elif self.lines_cleared == 3:
                self.score += 500
            elif self.lines_cleared == 4:
                self.score += 800
                self.tetris_count += 1  # Increment Tetris count when 4 lines are cleared
            
            # Update high score if the current score is higher
            if self.score >= self.high_score:
                self.high_score = self.score

    
    def play_game(self):
        self.initialize_game()
        self.bool_new_piece = True  # Initialize the flag
        while True:
            start_time = time.time()
            self.update_grid()
            self.handle_input_human(self.piece)

            self.piece = self.handle_falling(self.piece, self.frame_count)

            if not self.piece:
                print('Game Over! Score:', self.score)
                break
            self.frame_count += 1
            
        cv2.destroyAllWindows()
        
    def render(self, mode='human'):
        cv2.imshow('Tetris', self.img)
        cv2.waitKey(1)
        
    # This is a naive implementation of tetrimino pathfinding
    # I intend to implement a more effective pathfinding algorithm in the future
    # Which will take into account 'tucks' and 'spins'
    def get_possible_placements(self):
        # Assume this runs when piece is at the top
        possible_placements = []
        initial_x = self.piece.x
        initial_y = self.piece.y
        
        rotation_count = 4
        if self.piece.shape_type in ['I', 'S', 'Z']:
            rotation_count = 2
        elif self.piece.shape_type == 'O':
            rotation_count = 1
        
        for rotation in range(rotation_count):
            self.piece.rotate(self.placed_blocks_grid)
            og_path = ['rotate']
            # Move the piece to the leftmost position
            while self.piece.move('left', self.placed_blocks_grid):
                leftmost_x = self.piece.x
            # Move the piece to the rightmost position
            while self.piece.move('right', self.placed_blocks_grid):
                rightmost_x = self.piece.x
            
            # Iterate through all possible x positions
            for x in range(leftmost_x, rightmost_x + 1):
                path = og_path.copy()
                temp_x = initial_x
                while temp_x < x:
                    temp_x += 1
                    path.append('right')
                while temp_x > x:
                    temp_x -= 1
                    path.append('left')
                # Move the piece down until it collides with another piece
                self.piece.move('drop', self.placed_blocks_grid)
                path.append('drop')
                possible_placements.append((self.piece.shape, self.piece.position, path))
        
        self.piece.x = initial_x
        self.piece.y = initial_y
        # Rotate the piece back to its original orientation
        for _ in range(4 - rotation_count):
            self.piece.rotate(self.placed_blocks_grid)
        return possible_placements
    
    
    def get_possible_actions(self):
        visited = set()
        queue = deque()
        initial_state = (self.piece.shape.copy(), self.piece.x, self.piece.y)
        queue.append((initial_state, []))
        visited.add(self.piece.hash())
        possible_placements = []
        temp_lines_cleared = self.lines_cleared
        temp_total_lines_cleared = self.total_lines_cleared
        temp_tetris_count = self.tetris_count
        temp_score = self.score

        while queue:
            (shape, x, y), path = queue.popleft()
            for action in ['left', 'right', 'rotate', 'down', 'drop']:
                new_piece = self.piece.copy()
                new_piece.shape = shape.copy()
                new_piece.x = x
                new_piece.y = y
                moved = new_piece.move(action, self.placed_blocks_grid)
                if not moved:
                    continue
                state_id = new_piece.hash()
                if state_id in visited:
                    continue
                visited.add(state_id)
                new_path = path + [action]
                if not new_piece.move('down', self.placed_blocks_grid):
                    # The piece cannot move down further, so it's a valid placement
                    # Save the current grid and piece
                    temp_grid = self.placed_blocks_grid.copy()
                    temp_piece = self.piece.copy()
                    
                    
                    self.piece = new_piece
                    self.piece.draw(self.placed_blocks_grid)
                    self.handle_full_rows()
                    next_state = self.get_state()
                    done = np.any(self.placed_blocks_grid[0:HIDDEN_ROWS, 3:7] != 0)
                    done_reward = -100 if done else 0
                    #reward = self.lines_cleared**2 * GRID_WIDTH + 1 + done_reward
                    reward = self.score - temp_score + done_reward + 5
                    height = self.get_height()
                    bumpiness = self.get_bumpiness()
                    holes = self.get_holes()
                    
                    possible_placements.append({
                        'next_state': next_state,
                        'next_board': self.placed_blocks_grid.copy(),
                        'action_sequence': new_path,
                        'reward': reward,
                        'done': done,
                        'lines_cleared': self.lines_cleared,
                        'score': reward,
                        'height': height,
                        'bumpiness': bumpiness,
                        'holes': holes
                    })
                    
                    # Restore the grid and piece
                    self.placed_blocks_grid = temp_grid
                    self.lines_cleared = temp_lines_cleared
                    self.tetris_count = temp_tetris_count
                    self.piece = temp_piece
                    self.score = temp_score
                    self.total_lines_cleared = temp_total_lines_cleared
                    
                    
                else:
                    new_piece.y -= 1
                    # Continue searching from the new state
                    queue.append(((new_piece.shape.copy(), new_piece.x, new_piece.y), new_path))
        return possible_placements


    def visualize_possible_placements(self, piece_type):
        # Set the current piece to the specified type
        self.piece = Block(piece_type)
        # Generate all possible placements
        placements = self.get_possible_actions()
        print(f"Total placements found: {len(placements)}")
        # Loop through each placement
        for idx, placement in enumerate(placements):
            shape = placement['next_state'][1]  # Assuming next_state contains the shape
            x, y = placement['next_state'][0]  # Adjust based on actual structure
            path = placement['action_sequence']
            # Reset the grid to the preset board
            temp_grid = self.placed_blocks_grid.copy()
            # Place the piece on the temp grid
            for dy in range(shape.shape[0]):
                for dx in range(shape.shape[1]):
                    if shape[dy, dx]:
                        grid_x = x + dx
                        grid_y = y + dy
                        if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT + HIDDEN_ROWS:
                            temp_grid[grid_y, grid_x] = ord(piece_type)
            # Draw the temp grid
            self.img = np.zeros((GRID_HEIGHT * BLOCK_SIZE, GRID_WIDTH * BLOCK_SIZE + SCORE_PANEL_WIDTH, 3), dtype=np.uint8)
            for grid_y in range(HIDDEN_ROWS, GRID_HEIGHT + HIDDEN_ROWS):
                for grid_x in range(GRID_WIDTH):
                    block_value = temp_grid[grid_y, grid_x]
                    if block_value != 0:
                        color = COLORS[chr(block_value)]
                        top_left = (grid_x * BLOCK_SIZE + GRID_PADDING, (grid_y - HIDDEN_ROWS) * BLOCK_SIZE + GRID_PADDING)
                        bottom_right = ((grid_x + 1) * BLOCK_SIZE - GRID_PADDING, (grid_y - HIDDEN_ROWS + 1) * BLOCK_SIZE - GRID_PADDING)
                        cv2.rectangle(self.img, top_left, bottom_right, color, -1)
                        cv2.rectangle(self.img, top_left, bottom_right, (50, 50, 50), 2)
            # Draw the score panel
            self.draw_score_panel(self.img)
            # Display the action sequence and placement number
            cv2.putText(self.img, f'Placement {idx + 1}/{len(placements)}', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(self.img, f'Actions: {" ".join(path)}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            print("Actions:", path)
            cv2.imshow('Possible Placements', self.img)
            # Wait for a key press or a delay before showing the next placement
            key = cv2.waitKey(1)  # Display each placement for 500 milliseconds
            if key == ord('q'):
                break  # Exit if 'q' is pressed

def main():
    tetris = TetrisEnv()
    tetris.play_game()

if __name__ == "__main__":
    main()
