import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import namedtuple, deque
import matplotlib.pyplot as plt
from game import Board



Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'mask', 'next_mask'))

model_path = "policy_net.pth"

epsilon_decay = 0.995
min_epsilon = 0.01

board = Board()
input_size = 97   # based on game.encode_state()
hidden_size = 128  
output_size = len(board.get_all_possible_moves())

index_to_roll_mapping = {}


class GameNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GameNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, mask):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x * mask  # Apply the mask to zero out invalid actions
        return x

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def get_action_mask(board):
    valid_moves = board.get_valid_moves()
    mask = torch.zeros(output_size)
    for move in valid_moves:
        mask[move_to_index(move, board)] = 1
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


def select_action(state, policy_net, epsilon, mask):
    if random.random() < epsilon:
        valid_indices = mask.nonzero(as_tuple=True)[0].tolist()
        return random.choice(valid_indices)
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return policy_net(state_tensor, mask_tensor).argmax().item()

def optimize_model(policy_net, optimizer, memory, batch_size):
    if len(memory) < batch_size:
        return

    transitions = random.sample(memory, batch_size)
    batch = Transition(*zip(*transitions))

    state_batch = torch.tensor(batch.state, dtype=torch.float32)
    action_batch = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32)
    mask_batch = torch.tensor(batch.mask, dtype=torch.float32)

    # Compute the state-action values predicted by the policy network
    state_action_values = policy_net(state_batch, mask_batch).gather(1, action_batch)

    # The expected state-action values are the final rewards of the episodes
    expected_state_action_values = reward_batch.unsqueeze(1)

    # Compute Huber loss between predicted and expected state-action values
    loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Remove old memory if it exceeds a limit
    if len(memory) > 10000:
        del memory[:batch_size]

def train():
    num_episodes = 1000
    epsilon = 0.1
    epsilon_decay = 0.995
    min_epsilon = 0.01
    batch_size = 64
    learning_rate = 0.001

    policy_net = GameNet(input_size, hidden_size, output_size)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = deque(maxlen=10000)  # Efficient memory management with a deque
    
    # Logging
    rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        board = Board()
        state = board.encode_state()
        mask = get_action_mask(board)
        episode_memory = []
        episode_reward = 0
        steps = 0

        for t in range(2000):  # Limit the length of each episode
            action = select_action(state, policy_net, epsilon, mask)
            next_state, reward, done = board.step(index_to_move(action, board))
            next_mask = get_action_mask(board)
            episode_memory.append((state, action, reward, next_state, mask, next_mask))
            state = next_state
            mask = next_mask
            episode_reward += reward
            steps += 1

            if t == 1999:
                print(len(board.white_saved), len(board.black_saved), board.game_stages, episode_reward)

            if done:
                print(f"Episode {episode} completed in {steps} steps with reward {episode_reward}")
                final_score = episode_reward
                for state, action, reward, next_state, mask, next_mask in episode_memory:
                    memory.append(Transition(state, action, final_score, next_state, mask, next_mask))
                break

        optimize_model(policy_net, optimizer, memory, batch_size)
        
        rewards.append(episode_reward)
        episode_lengths.append(steps)

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(f"Episode {episode}: Average Reward: {avg_reward}, Average Length: {avg_length}")

    # Save the trained model
    torch.save(policy_net.state_dict(), model_path)

    # Plotting results
    plt.plot(rewards)
    plt.title('Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    plt.plot(episode_lengths)
    plt.title('Episode Lengths over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Length')
    plt.show()


if __name__ == "__main__":
    train()