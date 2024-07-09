# python -m http.server 8000
# python app.py
# http://localhost:8000

import random
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from game import Board
from model import GameNet, get_action_mask, index_to_move, load_model_for_inference, select_action

app = Flask(__name__)
CORS(app)

# Initialize board
board = Board()

# Define the model path and load the trained model
#model_path = "policy_net_2220_v0_offgoalflag.pth"
model_path = "policy_net_9990_v0_offgoalflag_doubleactionspace_withinreachreward.pth"
input_size = 97   # based on game.encode_state()
hidden_size = 128  
output_size = 4004
policy_net = load_model_for_inference(model_path, input_size, hidden_size, output_size)

@app.route('/select_moves', methods=['POST'])
def select_moves():
    try:
        state = request.json
        board.update_state(state)
        moves = board.get_valid_moves()
        print(f"Moves: {moves}")
        if moves:
            # Get the current state and mask
            current_state = board.encode_state()
            mask = get_action_mask(board)
            
            # Get the action from the model
            action = select_action(current_state, policy_net, 0, mask)  # 0 epsilon to ensure exploitation
            chosen_move = index_to_move(action, board)
            print(f"Chosen move: {chosen_move}")
            return jsonify({"message": "Game state updated successfully", "move": chosen_move}), 200
        else:
            return jsonify({"message": "No valid moves available"}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"message": "An error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
