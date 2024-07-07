import json
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
from collections import deque
from game import Board
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

index_to_roll_mapping = {}  # Dictionary to map action indices to rolls

# Load JSON data
with open('tile_neighbors.json', 'r') as f:
    tiles_data = json.load(f)

# Define racks
saved_racks = {
    "white_saved": {"type": "save", "pieces": []},
    "black_saved": {"type": "save", "pieces": []}
}

unentered_racks = {
    "white_unentered": [{"type": "unentered", "pieces": []} for _ in range(14)],
    "black_unentered": [{"type": "unentered", "pieces": []} for _ in range(14)]
}

# Encode pieces as one-hot vectors (simplified example)
colors = {"white": 0, "black": 1}
max_number = 14

def encode_piece(player, number):
    feature_vector = [0] * (2 * max_number)
    feature_vector[colors[player] * max_number + (number - 1)] = 1
    return feature_vector

# Create node features and edge list
node_features = []
node_index_map = {}
current_index = 0

# Process tiles
for tile_id, tile_data in tiles_data.items():
    node_features.append([0] * (2 * max_number))  # Initialize with empty features
    node_index_map[tile_id] = current_index
    current_index += 1

# Process saved racks
for rack_id, rack_data in saved_racks.items():
    node_features.append([0] * (2 * max_number))  # Initialize with empty features
    node_index_map[rack_id] = current_index
    current_index += 1

# Process unentered racks
for rack_id, rack_tiles in unentered_racks.items():
    for i, _ in enumerate(rack_tiles):
        node_features.append([0] * (2 * max_number))  # Initialize with empty features
        node_index_map[f"{rack_id}_{i}"] = current_index
        current_index += 1

node_features = torch.tensor(node_features, dtype=torch.float)

# Create edge list
edge_index = []

# Add edges for tiles
for tile_id, tile_data in tiles_data.items():
    tile_idx = node_index_map[tile_id]
    for neighbor in tile_data["neighbors"]:
        neighbor_id = f"ring{neighbor['ring']}_sector{neighbor['sector']}"
        neighbor_idx = node_index_map[neighbor_id]
        edge_index.append([tile_idx, neighbor_idx])
        edge_index.append([neighbor_idx, tile_idx])

# Add edges for saved racks
for tile_id, tile_data in tiles_data.items():
    if tile_data["type"] == "save":
        for rack_id in saved_racks.keys():
            save_tile_idx = node_index_map[tile_id]
            rack_idx = node_index_map[rack_id]
            edge_index.append([save_tile_idx, rack_idx])

# Add edges for unentered racks
for rack_id, rack_tiles in unentered_racks.items():
    for i in range(len(rack_tiles)):
        current_tile_idx = node_index_map[f"{rack_id}_{i}"]
        if i == 0:
            home_tile_id = "ring0_sector0"
            home_tile_idx = node_index_map[home_tile_id]
            edge_index.append([current_tile_idx, home_tile_idx])
        else:
            previous_tile_idx = node_index_map[f"{rack_id}_{i - 1}"]
            edge_index.append([current_tile_idx, previous_tile_idx])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Create graph data
data = Data(x=node_features, edge_index=edge_index)

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, global_feature_size):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 128)
        self.conv4 = GCNConv(128, 64)
        self.fc1 = torch.nn.Linear(64 + global_feature_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
    
    def forward(self, x, edge_index, global_features, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)  # Pool over the batch dimension
        


        # Ensure global_features has the correct batch dimension
        if len(global_features.shape) == 1:
            global_features = global_features.unsqueeze(0)

        # Expand global features to match the batch size of x
        if global_features.size(0) != x.size(0):
            global_features = global_features.expand(x.size(0), -1)
        
        # Concatenate global features
        x = torch.cat([x, global_features], dim=-1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ActorCriticGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_actions, global_feature_size):
        super(ActorCriticGCN, self).__init__()
        self.gcn = GCN(num_node_features, global_feature_size)
        self.value_head = torch.nn.Linear(128, 1)  # Output a single value
        self.policy_head = torch.nn.Linear(128, num_actions)  # Output action logits
    
    def forward(self, x, edge_index, global_features, batch=None):
        features = self.gcn(x, edge_index, global_features, batch=batch)
        value = self.value_head(features)
        action_logits = self.policy_head(features)
        return action_logits, value


def encode_die(die):
    one_hot_roll = [0] * 6
    one_hot_roll[die.number - 1] = 1
    return one_hot_roll + [1 if die.used else 0]

def encode_state_GCN(board):
    global node_index_map, edge_index

    # Define the number of features per node
    num_features_per_node = 28  # Pieces only
    node_features = torch.zeros((len(node_index_map), num_features_per_node), dtype=torch.float)

    for piece in board.pieces:
        if piece.tile:
            tile_index = piece.tile.index
            feature_vector = encode_piece(piece.player, piece.number)
            node_features[tile_index][:28] += torch.tensor(feature_vector, dtype=torch.float)
        elif piece.rack == board.white_unentered:
            rack_index = node_index_map[f"white_unentered_{board.white_unentered.index(piece)}"]
            feature_vector = encode_piece(piece.player, piece.number)
            node_features[rack_index][:28] += torch.tensor(feature_vector, dtype=torch.float)
        elif piece.rack == board.black_unentered:
            rack_index = node_index_map[f"black_unentered_{board.black_unentered.index(piece)}"]
            feature_vector = encode_piece(piece.player, piece.number)
            node_features[rack_index][:28] += torch.tensor(feature_vector, dtype=torch.float)

    # Encode die rolls (one-hot encoding and used status)
    die_features = []
    for die in board.dice:
        die_features.extend(encode_die(die))
    die_features = torch.tensor(die_features, dtype=torch.float)

    # Encode the current turn (binary: 0 for white, 1 for black)
    current_turn = 0 if board.current_player == 'white' else 1
    current_turn = torch.tensor([current_turn], dtype=torch.float)

    # Encode the game stage for each player (one-hot vectors)
    stages = {'opening': 0, 'midgame': 1, 'endgame': 2}
    white_stage = stages[board.game_stages['white']]
    black_stage = stages[board.game_stages['black']]
    stage_vector = [0] * 6
    stage_vector[white_stage] = 1
    stage_vector[3 + black_stage] = 1
    stage_vector = torch.tensor(stage_vector, dtype=torch.float)

    # Create graph data
    data = Data(x=node_features, edge_index=edge_index, die_features=die_features, current_turn=current_turn, stage_vector=stage_vector)

    return data

def interpret_node_features(node_features, node_index_map, die_features, current_turn, stage_vector):
    # Define feature sizes
    piece_features_size = 28

    # Reverse map to get node names from indices
    reverse_node_index_map = {v: k for k, v in node_index_map.items()}

    # Initialize results dictionary
    results = {}

    for i, features in enumerate(node_features):
        node_name = reverse_node_index_map[i]
        pieces = features[:piece_features_size].numpy()

        # Interpret pieces
        white_pieces = pieces[:14]
        black_pieces = pieces[14:]
        white_piece_numbers = [i + 1 for i, bit in enumerate(white_pieces) if bit == 1]
        black_piece_numbers = [i + 1 for i, bit in enumerate(black_pieces) if bit == 1]

        # Describe node by tile ring and position, or rack type
        if "ring" in node_name:
            description = node_name.replace('_', ' ').capitalize()
        else:
            description = node_name.replace('_', ' ').capitalize()

        results[f'Node {i} ({description})'] = {
            'white_pieces': white_piece_numbers,
            'black_pieces': black_piece_numbers
        }

    # Interpret die features
    def decode_single_die(encoded_die):
        roll = encoded_die[:6].tolist().index(1) + 1
        used = bool(encoded_die[6])
        return roll, used

    encoded_die_1 = die_features[:7]
    encoded_die_2 = die_features[7:14]

    die_roll_1, die_1_used = decode_single_die(encoded_die_1)
    die_roll_2, die_2_used = decode_single_die(encoded_die_2)

    # Interpret current turn
    current_player = 'white' if current_turn.item() == 0 else 'black'

    # Interpret game stages
    stages = {'opening': 0, 'midgame': 1, 'endgame': 2}
    white_stage = stage_vector[:3].tolist().index(1) if 1 in stage_vector[:3] else None
    black_stage = stage_vector[3:].tolist().index(1) if 1 in stage_vector[3:] else None
    stage_map = {0: 'opening', 1: 'midgame', 2: 'endgame'}
    white_stage = stage_map.get(white_stage, 'unknown')
    black_stage = stage_map.get(black_stage, 'unknown')

    # Add global features to the results
    results['Global'] = {
        'die_roll_1': die_roll_1,
        'die_1_used': die_1_used,
        'die_roll_2': die_roll_2,
        'die_2_used': die_2_used,
        'current_player': current_player,
        'white_stage': white_stage,
        'black_stage': black_stage
    }
    for node, data in results.items():
        print(f"{node}: {data}")

    return results

def get_action_mask(board, num_actions):
    valid_moves = board.get_valid_moves(mask_offgoals=True)
    mask = torch.zeros(num_actions, dtype=torch.bool)
    
    # Debugging log
    if len(valid_moves) == 0:
        logger.error("No valid moves available from the board state.")

    for move in valid_moves:
        action_index = move_to_action_index(move, board)  # Define this function to map move to action index
        mask[action_index] = 1

    return mask


def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))


def save_checkpoint(model, optimizer, episode):
    filepath = f'GCN_checkpoint_{episode}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': episode
    }, filepath)


def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    episode = checkpoint['episode']
    return model, optimizer, episode



def collect_trajectories(model, num_actions, epsilon, max_steps=1000):
    states, actions, rewards, dones = [], [], [], []
    board = Board()  # Initialize a new board instance to reset the state
    done = False
    step_count = 0

    while not done and step_count < max_steps:
        state = encode_state_GCN(board)  # Encode the state
        action_mask = get_action_mask(board, num_actions)  # Get action mask
        global_features = torch.cat([state.die_features, state.current_turn, state.stage_vector]).unsqueeze(0)
        policy_logits, value = model(state.x, state.edge_index, global_features)
        action = select_action(policy_logits, action_mask, epsilon)  # Pass epsilon to select_action
        move = action_index_to_move(action, board)  # Convert action index to move
        next_state, reward, done = board.step(move)  # Apply the move

        # Apply step penalty
        reward -= 500

        # Detailed logging
        #logger.info(f"Step {step_count}: Action taken: {action}, Move: {move}, Reward: {reward}, Done: {done}")

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        step_count += 1

    num_steps = len(rewards)
    logger.info(f'Episode {episode}: Total Reward: {sum(rewards)}, Number of Steps: {num_steps}')
    logger.info(f"White saved: {len(board.white_saved)}, Black saved: {len(board.black_saved)}")
    logger.info(f"White saveable: {len([p for p in board.pieces if p.player == 'white' and p.tile and p.can_be_saved()])}, Black saveable: {len([p for p in board.pieces if p.player == 'black' and p.tile and p.can_be_saved()])}")
    logger.info(f"Total offgoals: {board.offgoals['white'] + board.offgoals['black']}")

    return states, actions, rewards, dones

def select_action(policy_logits, action_mask, epsilon=0.1):
    valid_actions = torch.nonzero(action_mask).squeeze().tolist()

    # Ensure valid_actions is always a list
    if isinstance(valid_actions, int):
        valid_actions = [valid_actions]

    # Debugging log
    if len(valid_actions) == 0:
        logger.error("No valid actions available. Action mask may be incorrect.")

    # Fallback to a default action if no valid actions (should not happen)
    if len(valid_actions) == 0:
        valid_actions = [0]  # Assuming 0 is the index for passing the turn or a default action

    if np.random.rand() < epsilon:
        action = np.random.choice(valid_actions)
    else:
        masked_logits = policy_logits.masked_fill(~action_mask, -1e9)  # Mask invalid actions
        action_probs = F.softmax(masked_logits, dim=-1)
        action = torch.multinomial(action_probs, 1).item()

    return action



def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values.append(0)  # Append 0 to the end to handle the last value correctly
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns

def mini_batch_generator(states, actions, rewards, dones, advantages, returns, mini_batch_size):
    data_size = len(states)
    indices = np.arange(data_size)
    np.random.shuffle(indices)

    for start_idx in range(0, data_size, mini_batch_size):
        end_idx = min(start_idx + mini_batch_size, data_size)
        batch_indices = indices[start_idx:end_idx]

        state_batch = [states[i] for i in batch_indices]
        action_batch = torch.tensor([actions[i] for i in batch_indices], dtype=torch.long)
        reward_batch = torch.tensor([rewards[i] for i in batch_indices], dtype=torch.float)
        done_batch = torch.tensor([dones[i] for i in batch_indices], dtype=torch.float)
        advantage_batch = torch.tensor([advantages[i] for i in batch_indices], dtype=torch.float)
        return_batch = torch.tensor([returns[i] for i in batch_indices], dtype=torch.float)

        # Create a batch of graph data
        state_batch_x = torch.cat([state.x for state in state_batch], dim=0)
        state_batch_edge_index = torch.cat([state.edge_index + i * state.x.size(0) for i, state in enumerate(state_batch)], dim=1)
        state_batch_die_features = torch.cat([state.die_features.unsqueeze(0) for state in state_batch], dim=0)
        state_batch_current_turn = torch.cat([state.current_turn.unsqueeze(0) for state in state_batch], dim=0)
        state_batch_stage_vector = torch.cat([state.stage_vector.unsqueeze(0) for state in state_batch], dim=0)

        # Create batch tensor
        batch = torch.cat([torch.full((state.x.size(0),), i, dtype=torch.long) for i, state in enumerate(state_batch)])

        state_batch_data = Data(x=state_batch_x, edge_index=state_batch_edge_index, 
                                die_features=state_batch_die_features, current_turn=state_batch_current_turn, 
                                stage_vector=state_batch_stage_vector)

        yield state_batch_data, action_batch, reward_batch, done_batch, advantage_batch, return_batch, batch


def ppo_update(states, actions, rewards, dones, advantages, returns, model, optimizer, ppo_epochs=4, mini_batch_size=32, clip_epsilon=0.2):
    for _ in range(ppo_epochs):
        for batch in mini_batch_generator(states, actions, rewards, dones, advantages, returns, mini_batch_size):
            state_batch, action_batch, reward_batch, done_batch, advantage_batch, return_batch, batch_tensor = batch

            actual_batch_size = state_batch.die_features.size(0)  # Determine the actual batch size dynamically

            die_features_batch = state_batch.die_features.view(actual_batch_size, -1)  # Use actual_batch_size instead of mini_batch_size
            current_turn_batch = state_batch.current_turn.view(actual_batch_size, -1)
            stage_vector_batch = state_batch.stage_vector.view(actual_batch_size, -1)

            global_features_batch = torch.cat([die_features_batch, current_turn_batch, stage_vector_batch], dim=-1)


            policy_logits, values = model(state_batch.x, state_batch.edge_index, global_features_batch, batch=batch_tensor)

            # Compute policy loss
            action_log_probs = F.log_softmax(policy_logits, dim=-1)
            old_action_log_probs = action_log_probs.gather(1, action_batch.unsqueeze(-1))
            new_action_log_probs = action_log_probs.gather(1, action_batch.unsqueeze(-1))
            ratio = torch.exp(new_action_log_probs - old_action_log_probs)
            surr1 = ratio * advantage_batch.unsqueeze(-1)
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage_batch.unsqueeze(-1)
            policy_loss = -torch.min(surr1, surr2).mean()

            # Compute value loss
            value_loss = F.mse_loss(values, return_batch.unsqueeze(-1))

            # Compute total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * action_log_probs.mean()

            # Update the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def move_to_action_index(move, board):
    if move == (0, 0, 0):
        return 0  # Pass move

    piece_id, destination, roll = move
    player, number = piece_id
    piece_offset = (number - 1) + (0 if player == 'white' else 14)

    if destination == 'save':
        destination_offset = len(board.tiles)  # Separate index for 'save' move
    else:
        ring, pos = destination
        destination_offset = board.get_tile(ring, pos).index

    action_index = piece_offset * (len(board.tiles) + 1) + destination_offset + 1
    index_to_roll_mapping[action_index] = roll
    return action_index

def action_index_to_move(action_index, board):
    if action_index == 0:
        return (0, 0, 0, board.current_player)  # Pass move

    roll = index_to_roll_mapping[action_index]
    num_destinations = len(board.tiles) + 1  # Include 'save' destination

    # Extract destination index
    destination_index = (action_index - 1) % num_destinations

    if destination_index == len(board.tiles):
        destination = 'save'
    else:
        destination_tile = board.tiles[destination_index]
        destination = (destination_tile.ring, destination_tile.pos)

    # Extract piece index
    piece_index = (action_index - 1) // num_destinations
    player = 'white' if piece_index < 14 else 'black'
    piece_number = piece_index % 14 + 1

    piece = (player, piece_number)

    return piece, destination, roll, player


if __name__ == '__main__':
    board = Board()

    num_node_features = 28  # pieces only
    num_actions = 2002
    global_feature_size = 21  # 14 for die rolls and used status, 1 for current turn, 6 for game stages

    model = ActorCriticGCN(num_node_features, num_actions, global_feature_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Epsilon parameters
    initial_epsilon = 1.0
    min_epsilon = 0.1
    decay_rate = 0.995
    epsilon = initial_epsilon

    # Try to find the latest checkpoint
    import glob
    checkpoint_files = glob.glob('GCN_checkpoint_*.pt')
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
        model, optimizer, start_episode = load_checkpoint(latest_checkpoint, model, optimizer)
        logger.info(f"Loaded checkpoint from episode {start_episode}")
    else:
        start_episode = 0
        logger.info("No checkpoint found, starting training from scratch")

    num_episodes = 1000
    for episode in range(start_episode, num_episodes):
        states, actions, rewards, dones = collect_trajectories(model, num_actions, epsilon)
        values = [model(state.x, state.edge_index, torch.cat([state.die_features, state.current_turn, state.stage_vector]).unsqueeze(0))[1].item() for state in states]
        advantages, returns = compute_advantages(rewards, values)
        ppo_update(states, actions, rewards, dones, advantages, returns, model, optimizer)

        if episode > 0 and episode % 50 == 0:  # Save checkpoint every 50 episodes
            save_checkpoint(model, optimizer, episode)
            logger.info(f"Saved checkpoint at episode {episode}")

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * decay_rate)
        logger.info(f"Episode {episode+1}: Epsilon: {epsilon}")
