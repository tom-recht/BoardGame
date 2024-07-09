from GCN import *
from game import Board

board = Board()
print(move_to_action_index((('black', 13), (1, 8), 5), board))
print(move_to_action_index((('white', 13), (1, 8), 5), board))
