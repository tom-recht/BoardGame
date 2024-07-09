# python -m http.server 8000
# python app.py
# http://localhost:8000


import random
from flask import Flask, request, jsonify
from flask_cors import CORS
from game import Board
from agent import Agent

app = Flask(__name__)
CORS(app)

board = Board()
agent = Agent()

@app.route('/select_moves', methods=['POST'])
def select_moves():
    try:
        state = request.json
        board.update_state(state)
        moves = board.get_valid_moves()
        print(f"Moves: {moves}")
        if moves:
            chosen_move = agent.select_move_pair(moves, board, board.current_player)
            print(f"Chosen move: {chosen_move}")
            return jsonify({"message": "Game state updated successfully", "move": chosen_move}), 200
        else:
            return jsonify({"message": "No valid moves available"}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"message": "An error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
