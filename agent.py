import random
from collections import deque
from game import Board

class Agent():
    def __init__(self, board):
        self.board = board

    def choose_move(self, valid_moves):

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

    def shortest_distance(self, start_tile, end_tile):
        if start_tile == end_tile:
            return 0
        
        queue = deque([(start_tile, 0)])  # (current_tile, distance)
        visited = set()
        visited.add(start_tile)

        while queue:
            current_tile, distance = queue.popleft()

            for neighbor in current_tile.neighbors:
                if neighbor == end_tile:
                    return distance + 1
                if neighbor not in visited and not neighbor.is_blocked():
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        
        return float('inf')  # Return a large number if there is no path