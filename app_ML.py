# python -m http.server 8000
# python app.py
# http://localhost:8000
# 
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from game import Board
from GCN import ActorCriticGCN, encode_state_GCN, action_index_to_move, get_action_mask, select_action, interpret_node_features

app = Flask(__name__)
CORS(app) 

# Initialize board
board = Board()

# Define the model path and load the trained model
model_path = 'GCN_checkpoint_990.pt'
num_node_features = 28
num_actions = 2002
global_feature_size = 21 

policy_net = ActorCriticGCN(num_node_features, num_actions, global_feature_size)
policy_net.load_state_dict(torch.load(model_path))
policy_net.eval()

@app.route('/select_moves', methods=['POST'])
def select_moves():
    try:
        state = request.json
        board.update_state(state)
        moves = board.get_valid_moves(mask_offgoals=True)
        if moves:
            # Get the current state and mask
            encoded_state = encode_state_GCN(board)
            action_mask = get_action_mask(board, num_actions)
            global_features = torch.cat([encoded_state.die_features, encoded_state.current_turn, encoded_state.stage_vector]).unsqueeze(0)
            policy_logits, _ = policy_net(encoded_state.x, encoded_state.edge_index, global_features)
            action = select_action(policy_logits, action_mask)
            chosen_move = action_index_to_move(action, board)
            print(f"Action {action}. Chosen move: {chosen_move}")
            return jsonify({"message": "Game state updated successfully", "move": chosen_move}), 200
        else:
            return jsonify({"message": "No valid moves available"}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"message": "An error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
