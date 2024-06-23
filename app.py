# python -m http.server 8000
# python app.py
# http://localhost:8000


import random
from flask import Flask, request, jsonify
from flask_cors import CORS
from game import Board

app = Flask(__name__)
CORS(app)

board = Board()

@app.route('/select_moves', methods=['POST'])
def select_moves():
    state = request.json
    board.update_state(state)
    return jsonify({"message": "Game state updated successfully"}), 200

if __name__ == '__main__':
    app.run()
