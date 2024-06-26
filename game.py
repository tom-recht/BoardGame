import random
from collections import deque
import json

class Die:
    def __init__(self, board):
        self.board = board
        self.roll()

    def roll(self):
        self.number = random.randint(1, 6)  
        self.used = False

class Piece:
    def __init__(self, player, number, board):
        self.player = player
        self.number = number
        self.board = board
        self.tile = None
        self.rack = None
        self.reachable_tiles = None
        self.reachable_by_sum = None
        self.index = None

    def __repr__(self):
        return f'{self.player}({self.number})'
    
    def can_be_saved(self):
        if self.rack and self.rack == self.board.white_saved or self.rack == self.board.black_saved:
            return True  # already saved
        
        tile = self.tile
        if tile and tile.type == 'save':
            if self.number > 6 or (self.number == tile.number):
                return True
        return False

class Tile:
    def __init__(self, tile_type, ring, pos, board, number=None):
        self.type = tile_type
        self.ring = ring
        self.pos = pos
        self.pieces = []
        self.neighbors = []
        self.board = board
        self.number = number  # for goal tiles
        self.index = None

    def __repr__(self):
        return f"{self.type}({self.ring}, {self.pos})"
        return f"Tile(type={self.type}, ring={self.ring}, pos={self.pos}, number={self.number})"
    
    def is_blocked(self):
        return self.type == 'field' and len(self.pieces) > 1 and self.pieces[0].player != self.board.current_player

class Board:
    def __init__(self):
        self.players = ['white', 'black']
        self.dice = [Die(self), Die(self)] 
        self.pieces = []
        self.tiles = []
        self.tile_map = {}
        self.load_from_json('tile_neighbors.json')
        self.home_tile = self.get_tile(0, 0)
        self.current_player = 'white'
        self.white_unentered = []
        self.black_unentered = []
        self.white_saved = []
        self.black_saved = []
        self.assign_tile_indices()
        self.game_stages = {'white': 'opening', 'black': 'opening'}
        self.initialize_pieces()
        self.firstMove = None

    def __repr__(self):

        board_repr = "White unentered: " + str(self.white_unentered) + "\n"
        board_repr += "White saved: " + str(self.white_saved) + "\n"
        board_repr += "Black unentered: " + str(self.black_unentered) + "\n"
        board_repr += "Black saved: " + str(self.black_saved) + "\n"
        board_repr += "Pieces on board:\n"
        for piece in self.pieces:
            if piece.tile:
                board_repr += f"  {piece} on {piece.tile}\n"
        return board_repr

    def clear(self):
        self.white_unentered.clear()
        self.black_unentered.clear()
        self.white_saved.clear()
        self.black_saved.clear()
        self.pieces.clear()
        for tile in self.tiles:
            tile.pieces.clear()

    def add_tile(self, tile):
        self.tiles.append(tile)
        key = (tile.ring, tile.pos)
        self.tile_map[key] = tile

    def get_tile(self, ring, pos):
        return self.tile_map.get((ring, pos))

    def initialize_pieces(self):
        for player in self.players:
            pieces = [Piece(player, i + 1, self) for i in range(14)]
            random.shuffle(pieces)  # Shuffle the pieces randomly

            if player == 'white':
                self.white_unentered.extend(pieces)
                for piece in pieces:
                    piece.rack = self.white_unentered
            else:
                self.black_unentered.extend(pieces)
                for piece in pieces:
                    piece.rack = self.black_unentered

            self.pieces.extend(pieces)

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
        self.current_player = game_state_details['currentTurn']

        # Clear the board pieces
        self.clear()
        
        # Set dice values and used status
        for die, die_details in zip(self.dice, game_state_details['dice']):
            die.number = die_details['value']
            die.used = die_details['used']

        # Function to place pieces in their respective racks
        def place_pieces_in_rack(rack, pieces_details, player):
            rack.clear()
            for piece_details in pieces_details:
                piece = Piece(player, piece_details['number'], self)
                self.pieces.append(piece)
                rack.append(piece)
                piece.rack = rack
        
        # Place pieces in the unentered and saved racks
        place_pieces_in_rack(self.white_unentered, game_state_details['racks']['whiteUnentered'], 'white')
        place_pieces_in_rack(self.white_saved, game_state_details['racks']['whiteSaved'], 'white')
        place_pieces_in_rack(self.black_unentered, game_state_details['racks']['blackUnentered'], 'black')
        place_pieces_in_rack(self.black_saved, game_state_details['racks']['blackSaved'], 'black')

        
        # Place pieces on the board
        for piece_details in game_state_details['boardPieces']:
            player = piece_details['color']
            number = piece_details['number']
            ring = piece_details['tile']['ring']
            sector = piece_details['tile']['sector']
            tile = self.get_tile(ring, sector)
            piece = Piece(player, number, self)
            piece.tile = tile
            tile.pieces.append(piece)
            self.pieces.append(piece)

            print('Placed piece:', piece, 'on tile:', tile)
            
            if 'reachableBySum' in piece_details:
                piece.reachable_by_sum = [self.get_tile(t['ring'], t['sector']) for t in piece_details['reachableBySum']]

        self.assign_piece_indices()
        self.game_stages[self.current_player] = self.get_game_stage(self.current_player)


    def assign_tile_indices(self):
        for i in range(len(self.tiles)):
            self.tiles[i].index = i

    def assign_piece_indices(self):
        # Sort the pieces list by color (white then black) and then by their number
        self.pieces.sort(key=lambda piece: (piece.player != 'white', piece.number))
        # Assign the indices
        for i in range(len(self.pieces)):
            self.pieces[i].index = i+1

    def get_game_stage(self, player):
        unentered_rack = self.white_unentered if player == 'white' else self.black_unentered
        if len(unentered_rack) > 0:
            return 'opening'
        
        player_pieces = [p for p in self.pieces if p.player == player]
        if all(p.can_be_saved() for p in player_pieces):
            return 'endgame'
        return 'midgame'
    
    def switch_turn(self):
        self.firstMove = None  
        for die in self.dice:
            die.roll()
        self.current_player = 'white' if self.current_player == 'black' else 'black'

    def check_game_over(self):
        TOTAL_PIECES = len(self.pieces) // 2 
        white_saved_count = len(self.white_saved)
        black_saved_count = len(self.black_saved)
        
        if white_saved_count == TOTAL_PIECES:
            black_unsaved_count = TOTAL_PIECES - black_saved_count
            return 'white', black_unsaved_count
        
        if black_saved_count == TOTAL_PIECES:
            white_unsaved_count = TOTAL_PIECES - white_saved_count
            return 'black', white_unsaved_count
        
        return None, None  # No winner yet

    def get_unentered_piece(self):
        unentered_rack = self.white_unentered if self.current_player == 'white' else self.black_unentered
        if len(unentered_rack) > 0:
            return unentered_rack[0]
        return None

    def must_move_unentered(self):
        unentered_rack = self.white_unentered if self.current_player == 'white' else self.black_unentered
        if len(unentered_rack) == 0:
            return False
        if self.home_tile.pieces and any(piece.player == self.current_player for piece in self.home_tile.pieces):
            return False
        if self.firstMove:
            return False
        return True

    def get_saving_die(self, piece):
        current_tile = piece.tile
        if current_tile and current_tile.type == 'save' and (piece.number > 6 or piece.number == current_tile.number):
            if self.game_stages[piece.player] == 'endgame':
                if piece.number > 6:
                    highest_occupied_goal_number = max((tile.number for tile in self.tiles if tile.type == 'save' and len(tile.pieces) > 0 and any(p.player == piece.player for p in tile.pieces)), default=0)
                    valid_dice = [die for die in self.dice if (not die.used) and die.number == current_tile.number or (die.number > current_tile.number and current_tile.number >= highest_occupied_goal_number)]
                else:
                    valid_dice = [die for die in self.dice if (not die.used) and die.number == current_tile.number]
            else:
                valid_dice = [die for die in self.dice if (not die.used) and die.number == current_tile.number]

            if valid_dice:
                matching_die = next((die for die in valid_dice if die.number == current_tile.number), None)
                if matching_die:
                    die = matching_die
                else:
                    die = max(valid_dice, key=lambda die: die.number)
                return die.number
            else:
                return False  # The piece cannot be saved with the current dice rolls
            
    def get_reachable_tiles(self, start_tile, steps):
        queue = deque([(start_tile, 0)])  # Start with the current tile and 0 steps taken
        visited = set([start_tile])
        reachable_tiles = []

        while queue:
            current_tile, current_steps = queue.popleft()
            if current_steps < steps:     
                for neighbor in current_tile.neighbors:
                    if (neighbor not in visited and neighbor.type not in ['nogo', 'home'] and not neighbor.is_blocked()):  
                        queue.append((neighbor, current_steps + 1))
                        visited.add(neighbor)
                        if current_steps + 1 == steps:
                            reachable_tiles.append(neighbor)
            elif current_steps == steps:
                reachable_tiles.append(current_tile)

        return list(set(reachable_tiles))

    def get_reachable_tiles_by_dice(self, piece):   
        reachable_tiles = {self.dice[0].number: [], self.dice[1].number: []}
        
        if piece.rack and piece.rack in [self.white_unentered, self.black_unentered]:   # if an unentered piece, start from the home tile
            start_tile = self.home_tile
        else:
            start_tile = piece.tile

        if not self.dice[0].used:
            reachable_tiles[self.dice[0].number] = self.get_reachable_tiles(start_tile, self.dice[0].number)

            # uses 2 alternative ways of finding the sum-reachable tiles for a moved piece
            reachable_by_sum = None
            if piece.reachable_by_sum:  # this comes from update_state()
                reachable_by_sum = piece.reachable_by_sum
            elif self.firstMove and self.firstMove['piece'] == piece: 
                origin_tile = self.firstMove['origin_tile'] or self.home_tile
                reachable_by_sum = self.get_reachable_tiles(origin_tile, self.dice[0].number + self.dice[1].number)
            if reachable_by_sum:
                reachable_tiles[self.dice[0].number] = [tile for tile in reachable_tiles[self.dice[0].number] if tile in reachable_by_sum]

        if not self.dice[1].used:
            reachable_tiles[self.dice[1].number] = self.get_reachable_tiles(start_tile, self.dice[1].number)

            reachable_by_sum = None
            if piece.reachable_by_sum:
                reachable_by_sum = piece.reachable_by_sum
            elif self.firstMove and self.firstMove['piece'] == piece:
                origin_tile = self.firstMove['origin_tile'] or self.home_tile
                reachable_by_sum = self.get_reachable_tiles(origin_tile, self.dice[0].number + self.dice[1].number)      
            if reachable_by_sum:
                reachable_tiles[self.dice[1].number] = [tile for tile in reachable_tiles[self.dice[1].number] if tile in reachable_by_sum]
       
             # removed total rolls to avoid en-route capture complications   
 #       if not self.dice[0].used and not self.dice[1].used:
  #          reachable_tiles['total'] = self.get_reachable_tiles(start_tile, self.dice[0].number+self.dice[1].number)

        if piece.tile and piece.tile.type == 'save' and self.game_stages[piece.player] != 'opening':
            save_roll = self.get_saving_die(piece)
            if save_roll:             
                reachable_tiles[save_roll].append('save')  # this needs changing?
                print('Save roll', reachable_tiles[save_roll])

        piece.reachable_tiles = reachable_tiles
        print(piece, reachable_tiles)

    def get_valid_moves(self):

        # if must move captured piece(s), do so
        captured_pieces = [piece for piece in self.home_tile.pieces if piece.player == self.current_player]
        if captured_pieces:
            print('Captured pieces:', captured_pieces)
            for piece in captured_pieces:
                self.get_reachable_tiles_by_dice(piece)
            self.destinations_by_piece = {piece: piece.reachable_tiles for piece in captured_pieces}

        # if must move unentered piece, do so
        elif self.must_move_unentered():
            print('Must move unentered')
            piece = self.get_unentered_piece()
            print('Unentered piece:', piece)
            self.get_reachable_tiles_by_dice(piece)
            self.destinations_by_piece = {piece: piece.reachable_tiles}
            
        else:
            player_pieces = [p for p in self.pieces if p.player == self.current_player and p.tile and p.tile.type in ['field', 'save']]

            # check if there's an unentered piece which can enter, and if so add it to the list of pieces
            unentered_piece = self.get_unentered_piece()
            if unentered_piece:
                player_pieces.append(unentered_piece)

            for piece in player_pieces:
                self.get_reachable_tiles_by_dice(piece)
        
            self.destinations_by_piece = {piece: piece.reachable_tiles for piece in player_pieces}

        # transform the dictionary so that items are tuples of (piece, tile, roll)
        tuples_list = []
        for piece, moves in self.destinations_by_piece.items():
            for roll, destinations in moves.items():
                if destinations:  # Ignore empty destinations
                    for destination in destinations:
                        if destination == 'save':
                            tuples_list.append(((piece.player, piece.number), destination, roll))
                        else:
                            tuples_list.append(((piece.player, piece.number), (destination.ring, destination.pos), roll))

        tuples_list.append((0, 0, 0))  # add a pass move

        # add tuples of form (0, tile_index, 0) for saving opponent's block -- this doesn't seem to work
      #  for tile in self.tiles:
       #     if tile.is_blocked():
        #        tuples_list.append((0, (tile.ring, tile.pos), 0))
  
        return tuples_list
    
    def apply_move(self, move):
        piece_id, destination, roll = move

        # Handle the pass move (0, 0, 0)
        if move == (0, 0, 0):
            self.firstMove = None  # Reset first move for the next turn
            self.current_player = 'white' if self.current_player == 'black' else 'black'
            return

        # Find the piece object
        piece = next((p for p in self.pieces if (p.player, p.number) == piece_id), None)
        if not piece:
            print(f"No piece found for {piece_id}")
            return

        # Handle saving a piece
        if destination == 'save':
            saved_rack = self.white_saved if piece.player == 'white' else self.black_saved
            saved_rack.append(piece)
            if piece.tile:
                piece.tile.pieces.remove(piece)
            piece.tile = None
            piece.rack = saved_rack

        else:
            # Handle moving to a new tile
            ring, pos = destination
            new_tile = self.get_tile(ring, pos)

            # Remove the piece from its current location (rack or tile)
            if piece.rack:
                piece.rack.remove(piece)
                piece.rack = None
            if piece.tile:
                piece.tile.pieces.remove(piece)
            
            # Set the first move if not set already
            if not self.firstMove:
                self.firstMove = {'piece': piece, 'origin_tile': piece.tile}

            # Check if we are capturing an opponent piece (only on field tiles)
            if new_tile.type == 'field' and new_tile.pieces and new_tile.pieces[0].player != piece.player:
                captured_piece = new_tile.pieces.pop()
                captured_piece.tile = self.home_tile
                self.home_tile.pieces.append(captured_piece)

            # Move the piece to the new tile
            new_tile.pieces.append(piece)
            piece.tile = new_tile

        # Mark the die as used
        if roll == self.dice[0].number and not self.dice[0].used:
            self.dice[0].used = True
        elif roll == self.dice[1].number and not self.dice[1].used:
            self.dice[1].used = True


        self.game_stages[self.current_player] = self.get_game_stage(self.current_player)

        # Switch to the next player if both dice are used
        if all(die.used for die in self.dice):
            self.switch_turn()

    def get_save_rack(self, player):
        return self.white_saved if player == 'white' else self.black_saved
    
    def get_unentered_rack(self, player):
        return self.white_unentered if player == 'white' else self.black_unentered

    def shortest_route_to_goal(self, piece):
        start_tile = piece.tile if piece.tile else self.home_tile  # Use home tile if the piece has no tile

        if piece.can_be_saved():
            return 0

        queue = deque([(start_tile, 0)])  # (current tile, distance)
        visited = set([start_tile])

        while queue:
            current_tile, distance = queue.popleft()

            for neighbor in current_tile.neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    if neighbor.type == 'save' and (piece.number > 6 or piece.number == neighbor.number):
                        return distance + 1  # Found a goal tile from which the piece can be saved
                    if neighbor.type not in ['nogo', 'home'] and not neighbor.is_blocked():
                        queue.append((neighbor, distance + 1))

        return float('inf')  # No path found to a goal tile

        
def text_interface(board):
    while True:
        # Display the current state of the board
        print("\nCurrent Board State:")
        print(board)

        # Get valid moves
        valid_moves = board.get_valid_moves()

        # List valid moves
        print("\nValid moves:")
        for i, move in enumerate(valid_moves):
            piece_id, destination, roll = move
            piece_desc = f"{piece_id}" if piece_id != 0 else "Pass"
            dest_desc = f"{destination}" if destination != "save" else "Save"
            print(f"{i}: Move {piece_desc} to {dest_desc} with roll {roll}")

        # Prompt the user for a choice
        choice = input("Enter the number of the move you want to make (or 'q' to quit): ")

        if choice.lower() == 'q':
            print("Exiting...")
            break

        try:
            choice = int(choice)
            if 0 <= choice < len(valid_moves):
                chosen_move = valid_moves[choice]
                board.apply_move(chosen_move)
                print("Move applied!")
            else:
                print("Invalid choice. Please select a valid move number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def random_play(self):
    while True:
        print('Dice:', [die.number for die in self.dice])
        print('Current player:', board.current_player)
        # Get valid moves
        valid_moves = self.get_valid_moves()

        # Check for the end of the game
        winner, score = self.check_game_over()
        if winner:
            print(f"Game over! {winner} wins with a score of {score}.")
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
        print(f"Applied move: {chosen_move}")

        # Display the current state of the board
        print("\nCurrent Board State:")
        print(self)

if __name__ == '__main__':
    # Initialize the board and load the game state
    board = Board()

    """ filename = 'game_state (4).json'
    with open(filename, 'r') as f:
        data = json.load(f)
    board.update_state(data) """

    # Run the text interface
    random_play(board)


# add logic for saving opponent's block
