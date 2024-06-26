import random
import copy
from collections import deque

GAME_OVER_SCORE = 10000

INITIAL_WEIGHTS = {
    'saved_bonuses': {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6},
    'goal_bonuses': {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6},
    'game_stage_bonuses': {'midgame': 50, 'endgame': 100},
    'saved_piece': 20,
    'goal_piece': 10,
    'near_goal_piece': 4,
    'loose_piece': -1
}
class Agent():
    def __init__(self, board = None, weights = INITIAL_WEIGHTS):
        self.board = board
        self.weights = weights

    def random_move(self, valid_moves):

         # Remove pass move if there are other options
        non_pass_moves = [move for move in valid_moves if move != (0, 0, 0)]
        
        if not non_pass_moves:
            return (0, 0, 0)

        save_moves = [move for move in valid_moves if move[1] == 'save']
        if save_moves:
            chosen_move = random.choice(save_moves)
        else:
            # Check for moves that place a piece on a save goal where it can be saved
            prioritized_moves = []
            for move in valid_moves:
                piece_id, destination, roll = move
                if isinstance(destination, tuple):
                    ring, pos = destination
                    tile = self.board.get_tile(ring, pos)
                    if tile and tile.type == 'save':
                        piece = next((p for p in self.board.pieces if (p.player, p.number) == piece_id), None)
                        if piece and (piece.number > 6 or piece.number == tile.number):
                            prioritized_moves.append(move)

            # Filter out moves that involve moving pieces already on save goals where they can be saved
            non_pass_moves = [
                move for move in non_pass_moves
                if move[1] == 'save' or 
                not (isinstance(move[1], tuple) and 
                    self.board.get_tile(*move[1]) and 
                    self.board.get_tile(*move[1]).type == 'save' and 
                    any(p.number > 6 or (p.number == self.board.get_tile(*move[1]).number) 
                        for p in self.board.get_tile(*move[1]).pieces))
            ]

            if prioritized_moves:
                chosen_move = random.choice(prioritized_moves)
            else:
                chosen_move = random.choice(non_pass_moves)

        return chosen_move
    
    def evaluate_player(self, board, player):
        # number of saved pieces
        save_rack = board.get_save_rack(player)
        saved_pieces = len(save_rack)
        saved_bonus = sum(self.weights['saved_bonuses'].get(piece.number, 0) for piece in save_rack)
        print(f"Player {player} - Saved pieces: {saved_pieces}, Saved bonus: {saved_bonus}")

        # number of pieces on goals
        goal_pieces = [piece for piece in board.pieces if piece.player == player and piece.can_be_saved()]
        goal_bonus = sum(self.weights['goal_bonuses'].get(piece.number, 0) for piece in goal_pieces if piece.number <= 6)
        print(f"Player {player} - Goal pieces: {len(goal_pieces)}, Goal bonus: {goal_bonus}")

        # number of pieces within reach of a goal
        board_pieces = [piece for piece in board.pieces if piece.player == player and piece.tile]
        pieces_near_goal = len([piece for piece in board_pieces if board.shortest_route_to_goal(piece) <= 6])
        print(f"Player {player} - Pieces near goal: {pieces_near_goal}")

        # number of loose pieces
        loose_pieces = len([piece for piece in board_pieces if piece.tile.type == 'field' and len(piece.tile.pieces) == 1])
        print(f"Player {player} - Loose pieces: {loose_pieces}")

        # game stage bonus
        game_stage = board.game_stages[player]
        game_stage_bonus = self.weights['game_stage_bonuses'].get(game_stage, 0)
        print(f"Player {player} - Game stage: {game_stage}, Game stage bonus: {game_stage_bonus}")

        total_score = (saved_pieces * self.weights['saved_piece'] + saved_bonus +
                    len(goal_pieces) * self.weights['goal_piece'] + goal_bonus +
                    pieces_near_goal * self.weights['near_goal_piece'] +
                    loose_pieces * self.weights['loose_piece'] +
                    game_stage_bonus)
        print(f"Player {player} - Total score: {total_score}")

        return total_score


    def evaluate(self, board, player):
        print('Evaluating board')
        winner, score = board.check_game_over()
        if winner:
            factor = 1 if winner == player else -1
            return factor * score * GAME_OVER_SCORE
        print('Not game over')
        player_eval = self.evaluate_player(board, player)
        opponent = 'white' if player == 'black' else 'black'
        opponent_eval = self.evaluate_player(board, opponent)
        print('Player eval:', player_eval)
        return player_eval - opponent_eval

    def select_move_pair(self, moves, board, player):
        move_scores = dict()

        # Ensure moves is a set and does not contain integers
        if not isinstance(moves, (list, set)) or not all(isinstance(m, tuple) for m in moves):
            raise ValueError('Invalid moves format: expected a list or set of tuples.')

        print("Initial moves:", moves)

        # Evaluate the pass move
        move_scores[((0, 0, 0), (0, 0, 0))] = self.evaluate(board, player)

        # Create a set of moves without the pass move
        moves = set(moves)
        moves.discard((0, 0, 0))

        for move in moves:
            if not isinstance(move, tuple) or len(move) != 3:
                raise ValueError('Invalid move format: each move should be a tuple of length 3.')

            simulated_board = copy.deepcopy(board)
            simulated_board.apply_move(move)
            move_scores[(move, (0, 0, 0))] = self.evaluate(simulated_board, player)  # make one move then pass
            
            next_moves = set(simulated_board.get_valid_moves())
            
            print("Next moves for move", move, ":", next_moves)
            
            if not next_moves:
                continue
            next_moves.discard((0, 0, 0))

            for next_move in next_moves:
                if not isinstance(next_move, tuple) or len(next_move) != 3:
                    raise ValueError('Invalid next move format: each move should be a tuple of length 3.')

                simulated_board2 = copy.deepcopy(simulated_board)
                simulated_board2.apply_move(next_move)
                move_scores[(move, next_move)] = self.evaluate(simulated_board2, player)

        best_move_pair = max(move_scores, key=move_scores.get)
        return best_move_pair

