from game import Board

index_to_roll_mapping = {}

def move_to_index(move, board):
    if move == (0, 0, 0):
        return 0  # Pass move

    piece_id, destination, roll = move
    _, number = piece_id
    piece_offset = (number - 1)

    if destination == 'save':
        destination_offset = len(board.tiles)  # Separate index for 'save' move
    else:
        ring, pos = destination
        destination_offset = board.get_tile(ring, pos).index

    piece = next((p for p in board.pieces if (p.player, p.number) == piece_id), None)
    current_tile = piece.tile
    if current_tile and current_tile.type == 'save':
        is_on_goal_tile = 1 
    else:
        is_on_goal_tile = 0

    index = (
        piece_offset * (len(board.tiles) + 1) * 2 +  # +1 for the 'save' move index, *2 for the flag
        destination_offset * 2 + 
        is_on_goal_tile
    )

    index_to_roll_mapping[index] = roll
    return index

def index_to_move(index, board):

    if index == 0:
        return (0, 0, 0, board.current_player)  # Pass move

    num_tiles = len(board.tiles) + 1  # +1 for the 'save' move index
    is_on_goal_tile = index % 2
    index //= 2

    piece_offset = index // num_tiles
    destination_offset = index % num_tiles

    piece_number = piece_offset + 1  # Since piece_offset = (number - 1)

    if destination_offset == len(board.tiles):
        destination = 'save'
    else:
        tile = board.tiles[destination_offset]
        ring = tile.ring
        pos = tile.pos
        destination = (ring, pos)

    #roll = index_to_roll_mapping.get(index * 2 + is_on_goal_tile, None)
    try:
        roll = index_to_roll_mapping[index * 2 + is_on_goal_tile]
    except KeyError:
        print(f"KeyError: {index * 2 + is_on_goal_tile}, index: {index}")
        print(f"index: {index}, piece_number {piece_number}, destination_offset: {destination_offset}, is_on_goal_tile: {is_on_goal_tile}")
        print(f"Destination: {destination}")

    current_player = board.current_player
    piece = (current_player, piece_number)

    return (piece, destination, roll, current_player)


    
board = Board()

all_tiles = [(tile.ring, tile.pos) for tile in board.tiles]
all_pieces = [(piece.player, piece.number) for piece in board.pieces]
destinations = all_tiles + ['save']
dice = [1, 2, 3, 4, 5, 6]

all_moves = [(piece, destination, roll) for piece in all_pieces for destination in destinations for roll in dice]

#print(move_to_index((('white', 1), (0, 0), 1), board))
#print((index_to_move(1, board)))
#print(move_to_index((('white', 1), (0, 0), 1), board))

bad_moves = []
for move in all_moves:
    index = move_to_index(move, board)
    if index == 1208:
        print('Action 1208:', move)
    move_ = index_to_move(index, board)
    #if move[1] == (0,0):
   #     print(index, move, move_)
    if (move[1], move[2]) != (move_[1], move_[2]):
        bad_moves.append((move, move_))
      
      #  print(index, move, move_)

print((index_to_move(1, board)))
