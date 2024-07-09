# python -m http.server 8000
# python app.py
# http://localhost:8000
# 
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from game import Board
from agent import Agent

app = Flask(__name__)
CORS(app) 

# Initialize board
board = Board()
agent = Agent()

@app.route('/select_moves', methods=['POST'])
def select_moves():
    try:
        state = request.json
        board.update_state(state)
        moves = board.get_valid_moves(mask_offgoals=True)
        if moves:
            # Get the current state and mask
            chosen_moves = agent.select_move_pair(moves, board, board.current_player)
            print(f"Chosen moves: {chosen_moves}")
            return jsonify({"message": "Game state updated successfully", "move": chosen_moves}), 200
        else:
            return jsonify({"message": "No valid moves available"}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"message": "An error occurred"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
