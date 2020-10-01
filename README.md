# Converting-DQNs-to-SNNs
Python version>=3.6.0
Pyorch version>=1.4.0
Need to intall openai/gym(https://github.com/openai/gym) and openai/baselines(https://github.com/openai/baselines)

game_model_episode_time_seed_percentile.py is used for converting DQNs to SNNs and test the SNNs on Atari games.
The function of converting ANNs to SNNs is in conversion/conversion.py.

Here is a exmaple to run the code:
python game_model_episode_time_seed_percentile.py --game BreakoutNoFrameskip-v4 --model breakout_dqn.pt --episode 50 --time 500 --seed 10086 --percentile 99.0
