import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np
from game import Board

board = Board()
N = 70 + 14*2 + 2  # Tiles and rack spaces
input_size = 97
hidden_size = 128  
output_size = len(board.get_all_possible_moves())

index_to_roll_mapping = {}

class CustomGameEnv(gym.Env):
    def __init__(self):
        super(CustomGameEnv, self).__init__()
        self.board = Board()
        self.action_space = gym.spaces.Discrete(output_size)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(input_size,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Reset the environment with the provided seed
        self.board = Board()
        state = np.array(self.board.encode_state())
        valid_moves = self.board.get_valid_moves()
        self.action_mask = self.create_action_mask(valid_moves)
        print("Reset: Valid moves:", valid_moves)
        print("Reset: Action mask:", self.action_mask)
        return state, {}

    def step(self, action):

        move = index_to_move(action, self.board)  # Convert action index back to move
        print("Step: Move:", move)

        if not self.action_mask[action]:
            raise ValueError("Invalid action taken:", action)
    
        self.board.apply_move(move)
        reward = 0
        done = False

        winner, reward = self.board.check_game_over()
        if winner:
            done = True
        
        state = np.array(self.board.encode_state())
        valid_moves = self.board.get_valid_moves()
        self.action_mask = self.create_action_mask(valid_moves)
        
        print("Step: Action taken:", action)
        print("Step: Valid moves:", valid_moves)
        print("Step: Action mask:", self.action_mask)
        
        return state, reward, done, {}

    def create_action_mask(self, valid_moves):
        mask = np.zeros(output_size, dtype=np.int8)
        for move in valid_moves:
            index = move_to_index(move, self.board)
            print("Create action mask: Move:", move, "Index:", index)
            mask[index] = 1
        return mask

def move_to_index(move, board):  
    if move == (0, 0, 0):
        return 0  # Pass move

    piece_id, destination, roll = move

    _, number = piece_id
    piece_offset = (number - 1)

    if destination == 'save':
        destination_offset = len(board.tiles)  # Separate index for 'save' move
    else:
        ring, pos = destination
        destination_offset = board.get_tile(ring, pos).index

    index = (
        piece_offset * (len(board.tiles) + 1) +  # +1 for the 'save' move index
        destination_offset
    )

    # Store the roll in the mapping
    index_to_roll_mapping[index] = roll
    return index

def index_to_move(index, board):
    if index == 0:
        return (0, 0, 0, board.current_player)  # Pass move

    num_tiles = len(board.tiles) + 1  # +1 for the 'save' move index

    # Calculate piece offset and destination offset
    piece_offset = index // num_tiles
    destination_offset = index % num_tiles

    piece_number = piece_offset + 1  # Since piece_offset = (number - 1)

    if destination_offset == len(board.tiles):
        destination = 'save'
    else:
        tile = board.tiles[destination_offset]
        ring = tile.ring
        pos = tile.pos
        destination = (ring, pos)

    roll = index_to_roll_mapping.get(index, None)

    current_player = board.current_player
    piece = (current_player, piece_number)

    return (piece, destination, roll, current_player)  # returning current_player so on game over, reward can be positive or negative


env = DummyVecEnv([lambda: Monitor(CustomGameEnv())])

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./ppo_custom_game_tensorboard/")
model.learn(total_timesteps=100000)

# Save the model
model.save("ppo_custom_game")

# Load the model and play
model = PPO.load("ppo_custom_game")
obs = env.reset()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        obs = env.reset()
