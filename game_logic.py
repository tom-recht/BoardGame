# rings:
# 0: home, 1 tile
# 1-7: board
# 8: white unentered, 14 tiles
# 9: black unentered, 14 tiles
# 10: saved, 1 tile

# in this code "unnumbered" pieces have numbers from 7-14 instead of 0, to make all pieces distinct so the action space can be (piece, destination)
# make sure there's no bug as there was in game.js where if >1 captured pieces, one of them can move on combined roll

import random
import itertools
from collections import deque

TOTAL_PIECES = 14

class Piece:
    def __init__(self, player, number):
        self.player = player
        self.number = number
        self.tile = None
        self.index = None

    def __repr__(self):
        return f'[{self.player}, {self.number}, {self.tile}, {self.index}]'

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

    def add_piece(self, piece):
        self.pieces.append(piece)
        piece.tile = self

    def remove_piece(self, piece):
        if piece in self.pieces:
            self.pieces.remove(piece)
            piece.tile = None

    def __repr__(self):
        return f'[{self.ring},{self.pos}]'
    

class Board:
    def __init__(self):
        self.tiles = []
        self.create_tiles()
        self.tile_mapping = {(tile.ring, tile.pos): tile for tile in self.tiles}

        self.pieces = []
        self.initialize_pieces()
        self.moved_piece = None           # Piece that has been moved during the current turn

        self.game_stages = {'white': 'opening', 'black': 'opening'}
        self.current_player = 'white' 
        self.players = ['white', 'black']
        self.dice = [Die(self), Die(self)] 

        self.assign_neighbors()
        self.delete_nogo_tiles()   # don't need these anymore after neighbor assignment
        self.assign_tile_indices()
        self.assign_piece_indices()
        self.possible_moves = self.get_all_possible_moves()
        self.must_move_unentered = self.get_unentered_piece()
        self.game_over = False

    def get_tile(self, ring, pos):
        # Access the tile using the ring and position
        return self.tile_mapping.get((ring, pos))
        
    def tile_to_index(self, ring, pos):
        return self.tile_mapping.get((ring, pos)).index

    def index_to_tile(self, index):
        tile = self.tiles[index]
        return tile.ring, tile.pos
    
    def index_to_piece(self, index):
        return self.pieces[index]
    
    def assign_neighbors(self):
        # Define hardcoded neighbors for specific rings
        hardcoded_neighbors = {
            5: {
                12: [13],
                13: [12, 14],
                14: [13, 2],
                2: [14],
                4: [21],
                21: [4, 22],
                22: [6, 21],
                6: [22],
                8: [29],
                29: [8, 30],
                30: [10, 29],
                10: [30],
            },
            7: {
                6: [13],
                13: [6, 14],
                14: [13, 15],
                15: [14, 8],
                8: [15],
                10: [25],
                25: [10, 26],
                26: [25, 27],
                27: [26, 12],
                12: [27],
                2: [37],
                37: [2, 38],
                38: [37, 39],
                39: [38, 4],
                4: [39],
            }
        }

        for tile in self.tiles:
            ring_number, position = tile.ring, tile.pos
            if ring_number > 7:
                continue

            # Special handling for the home tile
            if tile.type == 'home':
                # Assign all field tiles in ring 1 as neighbors
                for potential_neighbor in self.tiles:
                    if potential_neighbor.ring == 1 and potential_neighbor.type == 'field':
                        tile.neighbors.append(potential_neighbor)
                continue  # Skip further neighbor assignment for the home tile

            # Standard neighbor assignment for non-hardcoded tiles
            if position not in hardcoded_neighbors.get(ring_number, {}):
                # Add neighbors from the same ring
                for offset in [-1, 1]:
                    neighbor_pos = (position + offset - 1) % 12 + 1
                    neighbor_tile = self.get_tile(ring_number, neighbor_pos)
                    if neighbor_tile and neighbor_tile.type not in ['nogo', 'home']:
                        tile.neighbors.append(neighbor_tile)

            # Add neighbors from the inner and outer rings
            for offset in [-1, 1]:
                neighbor_ring = ring_number + offset
                # If the tile is in ring 7, don't look at neighbors in ring 8
                if ring_number == 7 and neighbor_ring == 8:
                    continue
                neighbor_tile = self.get_tile(neighbor_ring, position)
                if neighbor_tile and neighbor_tile.type not in ['nogo', 'home']:
                    tile.neighbors.append(neighbor_tile)

            # Apply hardcoded neighbors for ring 5 and 7
            if ring_number in [5, 7] and position in hardcoded_neighbors[ring_number]:
                for hardcoded_neighbor_pos in hardcoded_neighbors[ring_number][position]:
                    hardcoded_neighbor_tile = self.get_tile(ring_number, hardcoded_neighbor_pos)
                    if hardcoded_neighbor_tile:
                        tile.neighbors.append(hardcoded_neighbor_tile)

    def delete_nogo_tiles(self):
        self.tiles = [tile for tile in self.tiles if tile.type != 'nogo']

    def assign_tile_indices(self):
        for i in range(len(self.tiles)):
            self.tiles[i].index = i

    def assign_piece_indices(self):
        # Sort the pieces list by color (white then black) and then by their number
        self.pieces.sort(key=lambda piece: (piece.player != 'white', piece.number))
        # Assign the indices
        for i in range(len(self.pieces)):
            self.pieces[i].index = i+1

    def get_reachable_tiles(self, start_tile, steps, index = False):
        queue = deque([(start_tile, 0)])  # Start with the current tile and 0 steps taken
        visited = set([start_tile])
        reachable_tiles = []

        while queue:
            current_tile, current_steps = queue.popleft()
            if current_steps < steps:     
                for neighbor in current_tile.neighbors:
                    if (neighbor not in visited and neighbor.type not in ['nogo', 'home'] and not self.is_blocked(neighbor)):  
                        queue.append((neighbor, current_steps + 1))
                        visited.add(neighbor)
                        if current_steps + 1 == steps:
                            if index:
                                reachable_tiles.append(neighbor.index)
                            else:
                                reachable_tiles.append(neighbor)
            elif current_steps == steps:
                if index:
                    reachable_tiles.append(current_tile.index)
                else:
                    reachable_tiles.append(current_tile)

        return list(set(reachable_tiles))
    
    def is_blocked(self, tile):
        return tile.type == 'field' and len(tile.pieces) > 1 and tile.pieces[0].player != self.current_player

    def get_unentered_piece(self):
        unentered_ring = 8 if self.current_player == 'white' else 9
        unentered_tiles = [self.get_tile(unentered_ring, pos) for pos in range(0,14)]
        unentered_tiles_with_a_piece = [tile for tile in unentered_tiles if len(tile.pieces) > 0]
        if not unentered_tiles_with_a_piece:
            return None
        else:
            return unentered_tiles_with_a_piece[-1].pieces[0] 

    def switch_turns(self):
        self.current_player = 'black' if self.current_player == 'white' else 'white'

        # check if player must move an unentered piece
        self.must_move_unentered = self.get_unentered_piece()

        self.roll_dice()
        self.moved_piece = None


    def check_game_over(self):
        saved_pieces = self.get_tile(10,0).pieces
        white_saved_count = sum(1 for piece in saved_pieces if piece.player == 'white')
        black_saved_count = sum(1 for piece in saved_pieces if piece.player == 'black')

        # Check if all pieces of one color are saved
        if white_saved_count == TOTAL_PIECES:
            return ['white', TOTAL_PIECES - black_saved_count]
        elif black_saved_count == TOTAL_PIECES:
            return ['black', TOTAL_PIECES - white_saved_count]
        else:
            return None

    def initialize_pieces(self):
        for r in range (8,10):
            player = 'white' if r == 8 else 'black'
            pieces = [Piece(player, s+1) for s in range(14)]  # Create all pieces for this player
            
            random.shuffle(pieces)  # Shuffle the pieces
            tiles = [self.get_tile(r, s) for s in range(14)]  # Create a list of tiles
            for tile, piece in zip(tiles, pieces):
                tile.add_piece(piece)
                self.pieces.append(piece)
    
    def get_reachable_tiles_by_dice(self, piece, index = False):   
        # in this version (unlike the Pygame version) I'm indexing the dictionary by the die roll number, not by 1 and 2; also assigning the result as a piece attribute, not just returning it
        reachable_tiles = {self.dice[0].number: [], self.dice[1].number: [], 'total': []}
        
        if piece.tile.ring in (8,9):   # if an unentered piece, start from the home tile
            start_tile = self.tiles[0]
        else:
            start_tile = piece.tile

        if not self.dice[0].used:
            # Reachable tiles for the first die
            reachable_tiles[self.dice[0].number] = self.get_reachable_tiles(start_tile, self.dice[0].number, index)
            if self.moved_piece and piece == self.moved_piece:  # if the piece has already been moved, restrict the reachable tiles to those it could reach with the total roll, to enforce shortest-path rule
                reachable_tiles[self.dice[0].number] = [tile for tile in reachable_tiles[self.dice[0].number] if tile in piece.reachable_tiles['total']]

        if not self.dice[1].used:
            # Reachable tiles for the second die
            reachable_tiles[self.dice[1].number] = self.get_reachable_tiles(start_tile, self.dice[1].number, index)
            if self.moved_piece and piece == self.moved_piece:
                reachable_tiles[self.dice[1].number] = [tile for tile in reachable_tiles[self.dice[1].number] if tile in piece.reachable_tiles['total']]
                
        if not self.dice[0].used and not self.dice[1].used:
            # Reachable tiles for the sum of both dice
            reachable_tiles['total'] = self.get_reachable_tiles(start_tile, self.dice[0].number+self.dice[1].number, index)

        if piece.tile.type == 'save' and self.game_stages[piece.player] != 'opening':
            save_roll = self.get_saving_die(piece)
            if save_roll:
                reachable_tiles[save_roll].append(self.tile_to_index(10,0))

        piece.reachable_tiles = reachable_tiles

    def create_tiles(self):
        # Central circle as 'home'
        self.tiles.append(Tile('home', 0, 1, self))
        
        goal_tile_numbers = [1, 4, 2, 5, 3, 6]  # Sequence for "save" tiles with numbers

        for r in range(7):   # rings
            for s in range(12):  # segments
                if r == 6:  # Special handling for the outermost ring
                    if s % 4 == 0:  # Splitting every other "nogo" into three "field" tiles
                        for mini_tile in range(3):
                            self.tiles.append(Tile('field', r + 1, (s+4) * 3 + mini_tile + 1, self))  # s+4 ensures no overlapping numbers
                    else:
                        tile_type = 'nogo' if s % 2 == 0 else 'save'
                        number = goal_tile_numbers[s // 2 % len(goal_tile_numbers)] if tile_type == 'save' else None
                        self.tiles.append(Tile(tile_type, r + 1, s + 1, self, number))
                else:
                    # Apply nogo tile logic for rings 1-6
                    if r == 0 and s % 4 == 0:  # Every 4th tile in Ring 1
                        tile_type = 'nogo'
                    elif r in [1, 4] and (s + 2) % 4 == 0:  # Every 4th tile offset by 2 in Ring 2
                        tile_type = 'nogo'
                    elif r in [3, 5] and s % 2 == 0:  # Every other tile in Rings 4 and 6
                        tile_type = 'nogo'
                    elif r == 4 and s % 4 == 0:  # Every 4th tile in Ring 5
                        for mini_tile in range(2):
                            self.tiles.append(Tile('field', r + 1, (s+6) * 2 + mini_tile + 1, self))  # s+6 ensures no overlapping numbers
                        continue  # Skip adding the original tile
                    else:
                        tile_type = 'field'
                    self.tiles.append(Tile(tile_type, r + 1, s + 1, self))
        
        # unentered racks: ring 8 for white, 9 for black
        for r in range (8,10):
            for s in range(14):
                self.tiles.append(Tile('unentered', r, s, self))

        # saved rack
        self.tiles.append(Tile('saved', 10, 0, self))

    def move_from_tuple(self, move_tuple):
        if move_tuple == (0, 0, 0):  # pass move
            self.switch_turns()
            return
        
        # if tuple is of the form (0, tile_index, 0), save opponent's block
        if move_tuple[0] == 0 and move_tuple[2] == 0:
            tile_index = move_tuple[1]
            tile = self.tiles[tile_index]
            print(tile.pieces)
            for piece in tile.pieces[:]:
                self.move_piece(piece, self.get_tile(10,0), 0)
            self.switch_turns()
            return

        piece, destination, roll = move_tuple
        self.move_piece(self.pieces[piece-1], self.tiles[destination], roll)

    def move_piece(self, piece, new_tile, roll):
        # Remove piece from the current tile
        current_tile = piece.tile
        current_tile.remove_piece(piece)

        # Capture if appropriate
        if new_tile.type == 'field' and len(new_tile.pieces) == 1 and new_tile.pieces[0].player != piece.player:
            captured_piece = new_tile.pieces[0]
            new_tile.remove_piece(captured_piece)
            self.tiles[0].add_piece(captured_piece)  # Move captured piece to home

        # Add piece to the new tile
        new_tile.add_piece(piece)

        # if this was the last unentered piece of the same color, move to midgame

        if current_tile.ring in (8,9):
            unentered_tiles = [self.get_tile(current_tile.ring, pos) for pos in range(0,14)]
            if all(len(tile.pieces) == 0 for tile in unentered_tiles):
                self.game_stages[self.current_player] = 'midgame'
                print(f"Player {self.current_player} has entered the midgame")

        self.moved_piece = piece
        self.must_move_unentered = None

        for die in self.dice:
            if die.number == roll and not die.used:
                die.used = True
                break

        # if moved out a captured or unentered piece, don't have to move another one
        if current_tile.type == 'home':
            self.must_move_unentered = False

        self.check_for_endgame(piece.player)

        game_over = self.check_game_over()
        if game_over:
            board.game_over = True
            winner = game_over[0]
            score = game_over[1]
            print('Game over!', winner, score)
            # code here to end the game

        if all(die.used for die in self.dice):
            self.switch_turns()
            return


    def get_saving_die(self, piece):
        # Verify the piece is on a 'save' tile
        current_tile = piece.tile
        if current_tile.type == 'save' and (piece.number > 6 or piece.number == current_tile.number):
            
            if self.game_stages[piece.player] == 'endgame':
                if piece.number > 6:
                    # In endgame, unnumbered pieces can be saved with a die roll >= the number on the goal tile
                    # Only if there are no higher numbered goal tiles occupied by this player's pieces
                    highest_occupied_goal_number = max((tile.number for tile in self.tiles if tile.type == 'save' and len(tile.pieces) > 0 and any(p.player == piece.player for p in tile.pieces)), default=0)
                    valid_dice = [die for die in self.dice if (not die.used) and die.number == current_tile.number or (die.number > current_tile.number and current_tile.number >= highest_occupied_goal_number)]
                else:
                    # In endgame with a numbered piece, the piece must be saved with the exact die roll
                    valid_dice = [die for die in self.dice if (not die.used) and die.number == current_tile.number]
            else:
                
                # Not in endgame, pieces must be saved with the exact die roll
                valid_dice = [die for die in self.dice if (not die.used) and die.number == current_tile.number]

            if valid_dice:
                # First, try to find a die that matches the current tile number exactly
                matching_die = next((die for die in valid_dice if die.number == current_tile.number), None)
                # If no exact match, use the die with the highest number (just in endgame)
                if matching_die:
                    die = matching_die
                else:
                    die = max(valid_dice, key=lambda die: die.number)
                return die.number
            else:
                return False  # The piece cannot be saved with the current dice rolls
            
    def can_piece_be_saved(self, piece):
        tile = piece.tile
        if tile.ring == 10:
            return True  # Piece is already saved

        if tile.type == 'save':
            if piece.number > 6 or (piece.number == tile.number):
                return True
        return False
    
    def check_for_endgame(self, player_color):
        player_pieces = [p for p in self.pieces if p.player == player_color]
        if all(self.can_piece_be_saved(piece) for piece in player_pieces):
            self.game_stages[player_color] = 'endgame'
            print(f"{player_color.capitalize()} is in the endgame stage.")
        else:
            # Ensure the stage is set back if previously set to endgame
            if self.game_stages[player_color] == 'endgame':
                self.game_stages[player_color] = 'midgame'  
                print(f"{player_color.capitalize()} is no longer in the endgame stage.")

    def get_valid_moves(self, readable=False):
        # if must move captured piece(s), do so
        captured_pieces = [piece for piece in self.tiles[0].pieces if piece.player == self.current_player]
        if captured_pieces:
            for piece in captured_pieces:
                self.get_reachable_tiles_by_dice(piece, index = True)
            self.destinations_by_piece = {piece.index: piece.reachable_tiles for piece in captured_pieces}

        # if must move unentered piece, do so
        elif self.must_move_unentered:
            piece = self.must_move_unentered

            # place it on the home tile
            #piece.tile.remove_piece(piece)
            #self.tiles[0].add_piece(piece)
    
            self.get_reachable_tiles_by_dice(piece, index = True)
            self.destinations_by_piece = {piece.index: piece.reachable_tiles}
            

        else:
            player_pieces = [p for p in self.pieces if p.player == self.current_player and p.tile.type in ['field', 'save']]

            # check if there's an unentered piece which can enter, and if so add it to the list of pieces
            unentered_piece = self.get_unentered_piece()
            if unentered_piece:
                player_pieces.append(unentered_piece)

            for piece in player_pieces:
                self.get_reachable_tiles_by_dice(piece, index = True)
        
            self.destinations_by_piece = {piece.index: piece.reachable_tiles for piece in player_pieces}



        # transform the dictionary so that items are tuples of (piece, tile, roll)
        tuples_list = []
        for piece, moves in self.destinations_by_piece.items():
            for roll, destinations in moves.items():
                if roll != 'total' and destinations:  # Ignore 'total' rolls and empty destinations
                    for destination in destinations:
                        tuples_list.append((piece, destination, roll))

        tuples_list.append((0, 0, 0))  # add a pass move

        # add tuples of form (0, tile_index, 0) for saving opponent's block
        for tile in self.tiles:
            if tile.type == 'field' and tile.pieces and len(tile.pieces) > 1 and tile.pieces[0].player != self.current_player:
                tuples_list.append((0, tile.index, 0))
  
        if not readable:
            return tuples_list
        
        # human-readable output
        adjusted_tuples_list = []
        for piece, destination, roll in tuples_list:
            piece = piece if piece <= 14 else piece - 14
            destination = self.index_to_tile(destination)
            adjusted_tuples_list.append((piece, destination, roll))
        
        tuple_dict = {adjusted_tuple: original_tuple for adjusted_tuple, original_tuple in zip(adjusted_tuples_list, tuples_list)}
        return tuple_dict

    def roll_dice(self):
        for die in self.dice:
            die.roll()

    def get_all_possible_moves(self):
        destination_tiles = [tile.index for tile in self.tiles if tile.type in ['field','save','saved']]
        pieces = range(len(self.pieces))
        die_rolls = range(1,7)
        all_possible_moves = list(itertools.product(pieces, destination_tiles, die_rolls))
        all_possible_moves.insert(0, (0, 0, 0))  # Add the tuple (0,0,0) for passing
        for destination in destination_tiles:
            all_possible_moves.append((0, destination, 0))  # Add an extra tuple for saving each tile: form (0, tile_index, 0)
        return all_possible_moves

    def render(self):
        for tile in self.tiles:
            if len(tile.pieces) > 0:
                pieces_str = ', '.join(piece.player[0].capitalize() + str(piece.number) for piece in tile.pieces)
                print(f'{tile}: {pieces_str}')


    def update_state(self, state):
        self.current_player = state['currentPlayer']
        for die in state['dice']:
            self.dice[die['index']].number = die['value']
            self.dice[die['index']].used = die['used']

class Die:
    def __init__(self, board):
        self.board = board
        self.roll()

    def roll(self):
        self.number = random.randint(1, 6)  
        self.used = False
        






# observation space: for each of 14*2 pieces, its location; for each field tile, whether it's blocked; game stage for each player; # of saved pieces for each player; current player?; dice rolls
# action space: for each of 14 pieces, its destination; for each field tile, whether to save the opponent's block; pass move (~1000) ; die roll?

