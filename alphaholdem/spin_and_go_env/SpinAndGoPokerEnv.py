import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.game_state_utils import restore_game_state
from pypokerengine.engine.hand_evaluator import HandEvaluator


class MockPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        # Human action input
        print("Valid actions: {}".format(valid_actions))
        action = input("Choose action: ")
        return action, 0  # Action and the amount (for 'raise')


class SpinAndGoPokerEnv(gym.Env):
    def __init__(self):
        super(SpinAndGoPokerEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # fold, call, raise
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, 1), dtype=np.float32)

        self.config = setup_config(max_round=5, initial_stack=1500, small_blind_amount=50)

        for i in range(1, 4):
            self.config.register_player(name=f"Player{i}", algorithm=MockPlayer())

    def step(self, action):
        # Convert Gym action to PokerEngine action
        # In this case, assume the action is already in the PokerEngine format
        poker_action = action

        # Process action and update the game state
        self.emulator.apply_action(poker_action)

        # Retrieve the new state and info
        game_state = self.emulator.game_state
        round_state = game_state['round_state']
        is_round_finished = round_state['action_histories'] is None

        # Define the reward
        # For simplicity, this example does not calculate a sophisticated reward
        reward = 0

        # Check if the game/round is finished
        done = self._is_game_done(game_state)

        # Additional info about the game
        info = {
            'round_state': round_state,
            'player_states': game_state['player_states']
        }

        # Get the next observation
        observation = self._get_observation(game_state)

        return observation, reward, done, info

    def _is_game_done(self, game_state):
        # Implement logic to determine if the game is finished
        # For example, check if there is only one player with chips remaining
        return len([p for p in game_state['player_states'] if p['stack'] > 0]) <= 1

    def reset(self):
        # Reset the environment to a new game
        # Reset the config or game state as necessary
        return self._get_observation()

    def render(self, mode='human'):
        # Render the game state
        print(f"Current state: {self.current_state}")

    def close(self):
        # Perform any cleanup
        pass

    def _get_observation(self):
        # Define how you generate the current observation from the game state
        pass


if __name__ == '__main__':
    env = SpinAndGoPokerEnv()
    observation = env.reset()

    for _ in range(10):
        action = env.action_space.sample()  # Replace this with your action
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            break
    env.close()
