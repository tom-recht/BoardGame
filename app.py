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
    try:
        state = request.json
        board.update_state(state)
        print(board)
        return jsonify({"message": "Game state updated successfully"}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"message": "An error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
