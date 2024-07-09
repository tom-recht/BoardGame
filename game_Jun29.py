def step(self, move_and_player, step_penalty=0.5, transition_factor=0.1):
    piece, destination, roll, player = move_and_player
    move = (piece, destination, roll)
    self.apply_move(move)

    winner, score = self.check_game_over()
    next_state = self.encode_state()
    
    if winner:
        print(f"Game over! {winner} wins with a score of {score}.")
        done = True
        reward = score * 100000 if winner == player else score * -100000

    else:  # intermediate rewards
        player_saved_rack = self.white_saved if player == 'white' else self.black_saved
        opponent_saved_rack = self.white_saved if player == 'black' else self.black_saved
        intermediate_reward = (len(player_saved_rack) - len(opponent_saved_rack)) * 10
        intermediate_reward += len(player_saved_rack) * 5
        
        player_saveable_pieces = len([p for p in self.pieces if p.can_be_saved() and p.player == player])
        opponent_saveable_pieces = len([p for p in self.pieces if p.can_be_saved() and p.player != player])
        intermediate_reward += (opponent_saveable_pieces - player_saveable_pieces) * 4

        if destination == 'save':
            intermediate_reward += 500
            if isinstance(piece, tuple):
                piece_object = next((p for p in self.pieces if (p.player, p.number) == piece), None)
                if piece_object.number <= 6:
                    intermediate_reward += piece_object.number * 50
        else:
            if isinstance(destination, tuple):
                tile = self.get_tile(destination[0], destination[1])
                if tile.type == 'save':
                    intermediate_reward += 50
        if self.game_stages[player] == 'endgame':
            intermediate_reward += 100
            intermediate_reward = intermediate_reward * 10
        
        # Apply step penalty
        intermediate_reward -= step_penalty

        # Blend intermediate and final rewards
        reward = (1 - transition_factor) * intermediate_reward + transition_factor * score

        done = False

    return next_state, reward, done
