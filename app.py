import random
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class GameAgent:
    def __init__(self, state):
        self.state = state

    def select_moves(self):
        # Dummy logic to select multiple moves
        current_player = self.state['current_player']
        dice = self.state['dice']
        pieces = self.state['pieces']

        available_moves = []

        # Consider moves with each die independently
        for piece in pieces:
            if piece['player'] == current_player:
                for move in piece.get('available_moves', []):
                    available_moves.append({'piece_id': piece['id'], 'target': move, 'used_dice': [dice[0]]})
                    if len(dice) > 1:
                        available_moves.append({'piece_id': piece['id'], 'target': move, 'used_dice': [dice[1]]})

        # Consider moves combining both dice
        for piece in pieces:
            if piece['player'] == current_player:
                for move in piece.get('available_moves', []):
                    if len(dice) > 1:
                        combined_move = {'piece_id': piece['id'], 'target': move, 'used_dice': dice}
                        available_moves.append(combined_move)

        if available_moves:
            # Choose one move to simulate using one or both dice
            selected_move = random.choice(available_moves)
            return [selected_move]

        # If no moves available, pass
        return None

    def update_state(self, new_state):
        self.state = new_state

initial_state = {
    'current_player': 'white',
    'pieces': [
        {'id': 1, 'player': 'white', 'available_moves': [{'ring': 1, 'sector': 2}]},
        {'id': 2, 'player': 'black', 'available_moves': []}
    ]
}
game_agent = GameAgent(initial_state)

@app.route('/select_moves', methods=['POST'])
def select_moves():
    state = request.json
    game_agent.update_state(state)
    moves = game_agent.select_moves()
    if moves:
        print('Moves selected by agent:', moves)  # Debug log
        return jsonify(moves)
    else:
        print('No moves found')  # Debug log
        return jsonify([]), 204  # No Content if no moves found

if __name__ == '__main__':
    app.run(debug=True)
