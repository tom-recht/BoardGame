import random
import itertools
from collections import deque
import json

class Die:
    def __init__(self, board):
        self.board = board
        self.number = None
        self.used = False

class Piece:
    def __init__(self, player, number):
        self.player = player
        self.number = number
        self.tile = None

    def __repr__(self):
        return f'[{self.player}, {self.number}, {self.tile}]'

class Tile:
    def __init__(self, tile_type, ring, pos, board, number=None):
        self.type = tile_type
        self.ring = ring
        self.pos = pos
        self.pieces = []
        self.neighbors = []
        self.board = board
        self.number = number  # for goal tiles

    def __repr__(self):
        return f"Tile(type={self.type}, ring={self.ring}, pos={self.pos}, number={self.number})"

class Board:
    def __init__(self):
        self.players = ['white', 'black']
        self.dice = [Die(self), Die(self)] 
        self.pieces = []
        self.tiles = []
        self.tile_map = {}
        self.load_from_json('tile_neighbors.json')

        self.white_unentered = []
        self.black_unentered = []
        self.white_saved = []
        self.black_saved = []

        self.firstMove = None

    def __repr__(self):

        board_repr = "White unentered: " + str(self.white_unentered) + "\n"
        board_repr += "White saved: " + str(self.white_saved) + "\n"
        board_repr += "Black unentered: " + str(self.black_unentered) + "\n"
        board_repr += "Black saved: " + str(self.black_saved) + "\n"
        board_repr += "Pieces on board:\n"
        for piece in self.pieces:
            board_repr += f"  {piece}\n"
        return board_repr

    def add_tile(self, tile):
        self.tiles.append(tile)
        key = (tile.ring, tile.pos)
        self.tile_map[key] = tile

    def get_tile(self, ring, pos):
        return self.tile_map.get((ring, pos))

    def load_from_json(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)

        for key, value in data.items():
            ring, sector = map(int, key.replace('ring', '').replace('sector', '').split('_'))
            tile_type = value['type']
            number = value.get('number')  # Retrieve number if it's a save tile
            tile = Tile(tile_type, ring, sector, self, number)
            self.add_tile(tile)

        for key, value in data.items():
            ring, sector = map(int, key.replace('ring', '').replace('sector', '').split('_'))
            tile = self.get_tile(ring, sector)
            if tile:
                for neighbor in value['neighbors']:
                    neighbor_tile = self.get_tile(neighbor['ring'], neighbor['sector'])
                    if neighbor_tile:
                        tile.neighbors.append(neighbor_tile)

    def update_state(self, game_state_details):
        # Set the current turn
        self.turn = game_state_details['currentTurn']
        
        # Set dice values and used status
        for die, die_details in zip(self.dice, game_state_details['dice']):
            die.number = die_details['value']
            die.used = die_details['used']

        # Function to place pieces in their respective racks
        def place_pieces_in_rack(rack, pieces_details, player):
            rack.clear()
            for piece_details in pieces_details:
                piece = Piece(player, piece_details['number'])
                rack.append(piece)
        
        # Place pieces in the unentered and saved racks
        place_pieces_in_rack(self.white_unentered, game_state_details['racks']['whiteUnentered'], 'white')
        place_pieces_in_rack(self.white_saved, game_state_details['racks']['whiteSaved'], 'white')
        place_pieces_in_rack(self.black_unentered, game_state_details['racks']['blackUnentered'], 'black')
        place_pieces_in_rack(self.black_saved, game_state_details['racks']['blackSaved'], 'black')
        
        # Clear the board pieces
        self.pieces.clear()
        
        # Place pieces on the board
        for piece_details in game_state_details['boardPieces']:
            player = piece_details['color']
            number = piece_details['number']
            ring = piece_details['tile']['ring']
            sector = piece_details['tile']['sector']
            tile = self.get_tile(ring, sector)
            piece = Piece(player, number)
            piece.tile = tile
            tile.pieces.append(piece)
            self.pieces.append(piece)
            
            if 'reachableBySum' in piece_details:
                piece.reachable_by_sum = {
                    'reachableBySum': [self.get_tile(t['ring'], t['sector']) for t in piece_details['reachableBySum']]
                }


board = Board()


