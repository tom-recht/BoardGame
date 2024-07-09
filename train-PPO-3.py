import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch as th
from game import Board


index_to_roll_mapping = {}

board = Board()
action_space_size = len(board.get_all_possible_moves())

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

    piece = next((p for p in board.pieces if (p.player, p.number) == piece_id), None)
    current_tile = piece.tile
    if current_tile and current_tile.type == 'save':
        is_on_goal_tile = 1 
    else:
        is_on_goal_tile = 0

    index = (
        piece_offset * (len(board.tiles) + 1) * 2 +  # +1 for the 'save' move index, *2 for the flag
        destination_offset * 2 + 
        is_on_goal_tile
    )

    index_to_roll_mapping[index] = roll
    return index

def index_to_move(index, board):
    if index == 0:
        return (0, 0, 0, board.current_player)  # Pass move

    num_tiles = len(board.tiles) + 1  # +1 for the 'save' move index
    is_on_goal_tile = index % 2
    index //= 2

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

    roll = index_to_roll_mapping.get(index * 2 + is_on_goal_tile, None)

    current_player = board.current_player
    piece = (current_player, piece_number)

    return (piece, destination, roll, current_player)

class BoardGameEnv(gym.Env):
    def __init__(self, board):
        super(BoardGameEnv, self).__init__()
        self.board = board

        # Define the action and observation space
        self.action_space = spaces.Discrete(action_space_size)  # Number of possible moves including pass move
        self.observation_space = spaces.Box(low=0, high=1, shape=(97,), dtype=np.float32)  # Adjusted size

    def reset(self):
        self.board = Board()  # Reinitialize the board
        state = self.board.encode_state()
        return np.array(state, dtype=np.float32)

    def step(self, action):
        move = index_to_move(action, self.board)
        next_state, reward, done = self.board.step(move)
        return np.array(next_state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):
        print(self.board)

    def close(self):
        pass

def get_action_mask(board):
    valid_moves = board.get_valid_moves()
    action_mask = np.zeros(action_space_size, dtype=bool)
    for move in valid_moves:
        index = move_to_index(move, board)
        action_mask[index] = True
    return action_mask

class MaskedPolicy(ActorCriticPolicy):
    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        latent_vf = self.mlp_extractor.forward_critic(features)
        
        distribution = self._get_action_dist_from_latent(latent_pi)
        if deterministic:
            actions = distribution.get_actions(deterministic=True)
        else:
            action_mask = get_action_mask(self.board)
            action_logits = distribution.distribution.logits
            action_logits[~action_mask] = -np.inf
            actions = th.argmax(action_logits, dim=1)

        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return actions, values, log_prob

    def _get_action_dist_from_latent(self, latent_pi):
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(mean_actions)
    
def make_env(board, rank, seed=0):
    def _init():
        env = BoardGameEnv(board)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

board = Board()
num_cpu = 4  # Number of processes to use
env = SubprocVecEnv([make_env(board, i) for i in range(num_cpu)])

model = PPO(
    MaskedPolicy,
    env,
    verbose=1,
    tensorboard_log="./ppo_board_game_tensorboard/"
)

model.learn(total_timesteps=100000)