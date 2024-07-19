# python -m http.server 8000
# python app.py
# http://localhost:8000

from flask import Flask, request, jsonify, Response
from flask_cors import CORS, cross_origin
import os
from game import Board
from agent import Agent
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)


app = Flask(__name__, static_folder='', static_url_path='')
CORS(app)
#CORS(app, resources={r"/*": {"origins": "https://tom-recht.github.io"}})

# Initialize board
board = Board()
agent = Agent()

@app.route('/select_moves', methods=['POST'])
def select_moves():
    try:
        state = request.json
        logger.debug(f"Received state: {state}")
        board.update_state(state)
        moves = board.get_valid_moves(mask_offgoals=True)
        logger.debug(f"Valid moves: {moves}")
        if moves:
            chosen_moves = agent.select_move_pair(moves, board, board.current_player)
            logger.debug(f"Chosen moves: {chosen_moves}")
            return jsonify({"message": "Game state updated successfully", "move": chosen_moves}), 200
        else:
            return jsonify({"message": "No valid moves available"}), 200
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"message": "An error occurred"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
