import os
import numpy as np
import torch
from sac import Agent
import gym
from utils.plot_curve import plot_learning_curve
import pybullet_envs

if __name__ == '__main__':
    env = gym.make('InvertedPendulumBulletEnv-v0')
    agent = Agent(input_dims=env.observation_space.shape,
                  env=env,
                  n_actions=env.action_space.shape[0])
    n_games = 2500
    filename = 'invertedpendulum.png'
    fig_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    # Main game loop
    for i in range(n_games):
        observation= env.reset()
        done = False
        score = 0

        while not done:
            action = agent.action_selection(observation=observation)
            n_observation, reward, done, info = env.step(action)
            score += reward
            agent.memorize(observation, action, reward, n_observation, done)
            if not load_checkpoint:
                agent.learn()
            observation = n_observation
        score_history.append(score)
        avg_score = torch.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.load_models()
        print(f'EP: {i} | Score: {score} | Average_score: {avg_score}')

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, fig_file)

