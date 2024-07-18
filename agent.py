import random
import copy
import json

GAME_OVER_SCORE = 10000
LOG_TO_FILE = True

INITIAL_WEIGHTS = {
    'saved_bonuses': {0:0, 1:12, 2:14, 3:16, 4:18, 5:20, 6:22},
    'goal_bonuses': {0:0, 1:12, 2:14, 3:16, 4:18, 5:20, 6:22},
    'game_stage_bonuses': {'midgame': 50, 'endgame': 100},
    'saved_piece': 20,
    'goal_piece': 10,
    'near_goal_piece': 4,
    'unentered_piece': -1,
    'loose_piece': -1,
    'distance_penalty': -.5
}
class Agent():
    def __init__(self, board = None, weights = INITIAL_WEIGHTS, log_file='game_log.json'):
        self.board = board
        self.weights = weights
        self.log = []
        self.log_file = log_file
        with open(self.log_file, 'w') as file:
            file.write(json.dumps(self.log, indent=4))
        print(f"Log file {self.log_file} created.")

    def evaluate_player(self, board, player):
        # number of saved pieces
        save_rack = board.get_save_rack(player)
        saved_pieces = len(save_rack)
        saved_bonus = sum(self.weights['saved_bonuses'].get(piece.number, 0) for piece in save_rack)

        # number of pieces on goals
        goal_pieces = [piece for piece in board.pieces if piece.player == player and piece.can_be_saved()]
        goal_bonus = sum(self.weights['goal_bonuses'].get(piece.number, 0) for piece in goal_pieces if piece.number <= 6)

        # number of pieces within reach of a goal
        board_pieces = [piece for piece in board.pieces if piece.player == player and piece.tile]
        pieces_near_goal = len([piece for piece in board_pieces if board.shortest_route_to_goal(piece) <= 6])

        # numbered piece not on goal
        numbered_off_goal = [piece for piece in board.pieces if piece.number <= 6 and not piece.can_be_saved()]
        off_goal_penalty = -1 * sum(self.weights['goal_bonuses'].get(piece.number, 0) for piece in numbered_off_goal)

        # total distance froms goals of other pieces
        pieces_not_near_goal = [piece for piece in board.pieces if board.shortest_route_to_goal(piece) > 6]
        total_distance = min(sum(board.shortest_route_to_goal(piece) for piece in pieces_not_near_goal), 1000)  # to avoid infinity
        total_distance += sum(self.weights['goal_bonuses'].get(piece.number, 0) for piece in pieces_not_near_goal if piece.number <= 6) / 10

        # number of loose pieces
        loose_pieces = len([piece for piece in board_pieces if piece.tile.type == 'field' and len(piece.tile.pieces) == 1])

        # number of pieces not entered
        unentered_rack = board.get_unentered_rack(player)
        unentered_pieces = len(unentered_rack)

        # game stage bonus
        game_stage = board.game_stages[player]
        game_stage_bonus = self.weights['game_stage_bonuses'].get(game_stage, 0)

        total_score = (saved_pieces * self.weights['saved_piece'] + saved_bonus +
                    len(goal_pieces) * self.weights['goal_piece'] + goal_bonus +
                    pieces_near_goal * self.weights['near_goal_piece'] +
                    loose_pieces * self.weights['loose_piece'] +
                    total_distance * self.weights['distance_penalty'] +
                    unentered_pieces * self.weights['unentered_piece'] +
                    off_goal_penalty +
                    game_stage_bonus)
        
        score_components = {
            'saved_pieces': saved_pieces * self.weights['saved_piece'],
            'saved_bonus': saved_bonus,
            'goal_pieces': len(goal_pieces) * self.weights['goal_piece'],
            'goal_bonus': goal_bonus,
            'pieces_near_goal': pieces_near_goal * self.weights['near_goal_piece'],
            'loose_pieces': loose_pieces * self.weights['loose_piece'],
            'total_distance': total_distance * self.weights['distance_penalty'],
            'unentered_pieces': unentered_pieces * self.weights['unentered_piece'],
            'off_goal_penalty': off_goal_penalty,
            'game_stage_bonus': game_stage_bonus
        }

        return total_score, score_components

    def evaluate(self, board, player):
        winner, score = board.check_game_over()
        if winner:
            factor = 1 if winner == player else -1
            return factor * score * GAME_OVER_SCORE, {}

        player_eval, player_components = self.evaluate_player(board, player)
        opponent = 'white' if player == 'black' else 'black'
        opponent_eval, opponent_components = self.evaluate_player(board, opponent)

        total_score = player_eval - opponent_eval
        score_components = {
            'player': player_components,
            'opponent': opponent_components
        }

        return total_score, score_components

    def select_move_pair(self, moves, board, player):
        move_scores = dict()

        # Ensure moves is a set and does not contain integers
        if not isinstance(moves, (list, set)) or not all(isinstance(m, tuple) for m in moves):
            raise ValueError('Invalid moves format: expected a list or set of tuples.')

        # Evaluate the pass move
        move_scores[((0, 0, 0), (0, 0, 0))] = self.evaluate(board, player)

        # Create a set of moves without the pass move
        moves = set(moves)
        moves.discard((0, 0, 0))

        for move in moves:
            if not isinstance(move, tuple) or len(move) != 3:
                raise ValueError('Invalid move format: each move should be a tuple of length 3.')

            board.apply_move(move)
            move_scores[(move, (0, 0, 0))] = self.evaluate(board, player)  # make one move then pass

            next_moves = set(board.get_valid_moves(mask_offgoals=True))

            if not next_moves:
                continue
            next_moves.discard((0, 0, 0))

            for next_move in next_moves:
                if not isinstance(next_move, tuple) or len(next_move) != 3:
                    raise ValueError('Invalid next move format: each move should be a tuple of length 3.')

                board.apply_move(next_move, switch_turn = False)
                move_scores[(move, next_move)] = self.evaluate(board, player)
                board.undo_last_move()

            board.undo_last_move()

        best_move_pair = max(move_scores, key=lambda k: move_scores[k][0])
        best_move_score, best_move_components = move_scores[best_move_pair]

        self.log.append({
            'move': best_move_pair,
            'score': best_move_score,
            'components': best_move_components
        })

        if LOG_TO_FILE:
            with open(self.log_file, 'w') as file:
                file.write(json.dumps(self.log, indent=4))
            print(f"Log updated with move: {best_move_pair}")

        return best_move_pair



    def save_log_to_file(self):
        return json.dumps(self.log, indent=4)


# when agent brings out a captured piece it sometimes thinks it has no further moves?