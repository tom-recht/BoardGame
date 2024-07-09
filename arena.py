import copy
from agent import Agent
from model import GameNet, get_action_mask, index_to_move, load_model_for_inference, select_action
from game import Board

# Assuming you have classes or functions for the agents and the board

def play_game(agent1, policy_net, board):
    player = 'white'  # Assuming player 1 starts
    winner, score = board.check_game_over()
    while not winner:
        if player == 'white':
            # Agent 1's turn, selecting two moves -- rule-based agent
            valid_moves = board.get_valid_moves()
            move1, move2 = agent1.select_move_pair(valid_moves, board, player)
            board.apply_move(move1)
            winner, score = board.check_game_over()
            if not winner:  # Check if the game ended after the first move
                board.apply_move(move2)
        else:
            # Agent 2's turn, selecting one move at a time -- model-based agent
            for _ in range(2):  # Two moves for two dice rolls
                current_state = board.encode_state()
                mask = get_action_mask(board)
                action = select_action(current_state, policy_net, 0, mask)
                chosen_move = index_to_move(action, board)[:3]
                board.apply_move(chosen_move)
                winner, score = board.check_game_over()
                if winner:  # Check if the game ended after this move
                    break

        # Print or log the board state after each turn if needed
        print("Board state after player", player, "move(s):", board)

        # Switch player
        player = 'white' if player == 'black' else 'black'

    # Determine the winner or result
    winner, score = board.check_game_over()
    print("Game over. Winner:", winner, "Score:", score)
    return winner, score


agent1 = Agent()

model_path = "policy_net_260_v0_offgoalflag_doubleactionspace_withinreachreward.pth"
input_size = 97   # based on game.encode_state()
hidden_size = 128  
output_size = 4004
policy_net = load_model_for_inference(model_path, input_size, hidden_size, output_size)

board = Board()

play_game(agent1, policy_net, board)
