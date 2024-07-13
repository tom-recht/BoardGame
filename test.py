import copy

class Tile:
    def __init__(self, id):
        self.id = id

class Piece:
    def __init__(self, tile):
        self.tile = tile

class Board:
    def __init__(self):
        self.firstMove = None
        self.tiles = [Tile(i) for i in range(5)]
        self.pieces = [Piece(self.tiles[i]) for i in range(5)]

    def apply_move(self, piece):
        self.firstMove = {'piece': piece, 'origin_tile': piece.tile}

# Original board setup
board = Board()
board.apply_move(board.pieces[0])

# Deep copy of the board
simulated_board = copy.deepcopy(board)

# Checking the firstMove
print(simulated_board.firstMove['piece'] is simulated_board.pieces[0])  # Should be True
print(simulated_board.firstMove['origin_tile'] is simulated_board.tiles[0])  # Should be True
