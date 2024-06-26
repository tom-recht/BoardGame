from game import Board
import time
import random
import copy
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def play(self):
    while True:
        valid_moves = self.get_valid_moves()

        winner, score = self.check_game_over()
        if winner:
            break

        # Check for save moves and prioritize them
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
                    tile = self.get_tile(ring, pos)
                    if tile and tile.type == 'save':
                        piece = next((p for p in self.pieces if (p.player, p.number) == piece_id), None)
                        if piece and (piece.number > 6 or piece.number == tile.number):
                            prioritized_moves.append(move)

            # Filter out moves that involve moving pieces already on save goals where they can be saved
            valid_moves = [
                move for move in valid_moves
                if move[1] == 'save' or 
                not (isinstance(move[1], tuple) and 
                     self.get_tile(*move[1]) and 
                     self.get_tile(*move[1]).type == 'save' and 
                     any(p.number > 6 or (p.number == self.get_tile(*move[1]).number) 
                         for p in self.get_tile(*move[1]).pieces))
            ]

            if prioritized_moves:
                chosen_move = random.choice(prioritized_moves)
            else:
                chosen_move = random.choice(valid_moves)

        # Apply the move
        self.apply_move(chosen_move)


board = Board()

tic = time.time()
for i in range(100):
    b = copy.deepcopy(board)
    play(b)
toc = time.time()
print(toc - tic)
# 3-4 seconds


