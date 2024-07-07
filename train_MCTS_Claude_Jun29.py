import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import os
from game import Board

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'mask', 'next_mask', 'player'))

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

def optimize_model(policy_net, target_net, optimizer, memory, batch_size, gamma=0.99):
    if len(memory) < batch_size:
        return

    transitions = random.sample(memory, batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.tensor([s for s in batch.next_state if s is not None], dtype=torch.float32)
    
    state_batch = torch.tensor(batch.state, dtype=torch.float32)
    action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32)
    mask_batch = torch.stack(batch.mask)
    next_mask_batch = torch.stack(batch.next_mask)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    state_action_values = policy_net(state_batch, mask_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(batch_size)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states, next_mask_batch[non_final_mask]).max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def save_checkpoint(model, optimizer, epoch, epsilon):
    version = 0
    base_path = f"policy_net_{epoch}"
    while os.path.exists(f"{base_path}_v{version}.pth"):
        version += 1
    path = f"{base_path}_v{version}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epsilon': epsilon
    }, path)
    print(f"Model saved at {path}\n")

def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    epsilon = checkpoint['epsilon']
    return model, optimizer, epoch, epsilon

def compute_discounted_rewards(rewards, gamma=0.99):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards

def load_model_for_inference(model_path, input_size, hidden_size, output_size):
    model = GameNet(input_size, hidden_size, output_size)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode
    return model

def select_action_for_inference(state, policy_net):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action = policy_net(state_tensor).argmax().item()
    return action

def get_model_move(board, model_path):
    # Define the input, hidden, and output sizes based on your model architecture
    input_size = 97   # based on game.encode_state()
    hidden_size = 128  
    output_size = len(board.get_all_possible_moves())

    # Load the trained model
    policy_net = load_model_for_inference(model_path, input_size, hidden_size, output_size)

    # Get the current state and mask
    state = board.encode_state()
    mask = get_action_mask(board)

    # Get the action from the model
    action = select_action_for_inference(state, policy_net, mask)
    
    return action

def train(model_path=None):
    num_episodes = 1000
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01
    batch_size = 128
    learning_rate = 0.0005
    step_penalty = 0.08

    policy_net = GameNet(input_size, hidden_size, output_size)
    target_net = GameNet(input_size, hidden_size, output_size)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

    start_epoch = 0
    if model_path and os.path.exists(model_path):
        policy_net, optimizer, start_epoch, epsilon = load_checkpoint(model_path, policy_net, optimizer)
        print(f"Resuming training from epoch {start_epoch}")

    target_net.load_state_dict(policy_net.state_dict())
    memory = deque(maxlen=100000)

    rewards = []
    episode_lengths = []

    for episode in range(start_epoch, num_episodes):
        board = Board()
        state = board.encode_state()
        mask = get_action_mask(board)
        episode_reward = {'white': 0, 'black': 0}
        steps = 0
        rewards_for_episode = []

        for t in range(2000):  # Limit the length of each episode
            current_player = board.current_player
            action = select_action(state, policy_net, epsilon, mask)
            next_state, reward, done = board.step(index_to_move(action, board))
            next_mask = get_action_mask(board)

            reward -= step_penalty * t

            # Store experience with intermediate reward
            memory.append(Transition(state, action, next_state, reward, mask, next_mask, current_player))
            rewards_for_episode.append(reward)

            state = next_state
            mask = next_mask
            episode_reward[current_player] += reward
            steps += 1

            # Perform optimization step
            if len(memory) > batch_size:
                optimize_model(policy_net, target_net, optimizer, memory, batch_size)

            if done:
                print(f"Episode {episode} completed in {steps} steps")
                break

        # Compute discounted rewards for the episode
        discounted_rewards = compute_discounted_rewards(rewards_for_episode)

        # Update rewards and episode lengths
        total_reward = sum(discounted_rewards)
        rewards.append(total_reward)
        episode_lengths.append(steps)

        # After 2000 moves or game completion
        print(f"Episode {episode} ended after {steps} moves. White reward: {episode_reward['white']}, Black reward: {episode_reward['black']}")
        print(f"White saved: {len(board.white_saved)}, Black saved: {len(board.black_saved)}")

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Update target network
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

            # Save the trained model and optimizer state
            save_checkpoint(policy_net, optimizer, episode, epsilon)

    # Plotting results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(1, 2, 2)
    plt.plot(episode_lengths)
    plt.title('Episode Lengths over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Length')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model_path = "policy_net_210_v0.pth"  # Modify this path if you have a different checkpoint to resume from
    train(model_path)





