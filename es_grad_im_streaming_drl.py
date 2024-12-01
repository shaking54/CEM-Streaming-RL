from copy import deepcopy
import argparse

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import pandas as pd

import os, pickle, argparse
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from torch.distributions import Normal
from sdrl.optim import ObGD as Optimizer
from sdrl.time_wrapper import AddTimeInfo
from sdrl.normalization_wrappers import NormalizeObservation, ScaleReward
from sdrl.sparse_init import sparse_init

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from ES import sepCMAES, sepCEM, sepMCEM
from models import RLNN
from collections import namedtuple
from random_process import GaussianNoise
from memory import Memory, Archive
from samplers import IMSampler
from util import *


Sample = namedtuple('Sample', ('params', 'score',
                               'gens', 'start_pos', 'end_pos', 'steps'))
Theta = namedtuple('Theta', ('mu', 'cov', 'samples'))
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor


def evaluate(agent, env, noise=None, render=False, entropy_coeff=None, overshooting_info=None, trials=3, debug=False):
    """
    Computes the score of an actor on a given number of runs,
    fills the memory if needed
    """

    returns, term_time_steps = [], []
    steps = 0
    
    while trials: 
        score = 0
        s, _ = deepcopy(env.reset())
        terminated = False
        while not terminated:    
            # get next action and act
            a = agent.sample_action(s)
            s_prime, r, terminated, truncated, info = env.step(a)
            score += r
            agent.update_params(s, a, r, s_prime,  terminated or truncated, entropy_coeff, overshooting_info)
            s = s_prime
            
            if terminated or truncated:
                if debug:
                    print("Episodic Return: {}, Time Step {}".format(info['episode']['r'][0], t))
                returns.append(info['episode']['r'][0])
                terminated, truncated = False, False
                s, _ = env.reset()
            steps += 1

        returns.append(score)
        trials -= 1

    return np.mean(returns), steps


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)

class Actor(RLNN):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128):
        super(Actor, self).__init__()
        self.fc_layer   = nn.Linear(n_obs, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.linear_mu = nn.Linear(hidden_size, n_actions)
        self.linear_std = nn.Linear(hidden_size, n_actions)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.fc_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        mu = self.linear_mu(x)
        pre_std = self.linear_std(x)
        std = F.softplus(pre_std)
        return mu, std

class Critic(RLNN):
    def __init__(self, n_obs=11, hidden_size=128):
        super(Critic, self).__init__()
        self.fc_layer   = nn.Linear(n_obs, hidden_size)
        self.hidden_layer  = nn.Linear(hidden_size, hidden_size)
        self.linear_layer  = nn.Linear(hidden_size, 1)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.fc_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)      
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        return self.linear_layer(x)

class StreamAC(RLNN):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128, lr=1.0, gamma=0.99, lamda=0.8, kappa_policy=3.0, kappa_value=2.0):
        super(StreamAC, self).__init__()
        self.gamma = gamma
        self.actor = Actor(n_obs=n_obs, n_actions=n_actions, hidden_size=hidden_size)
        self.critic = Critic(n_obs=n_obs, hidden_size=hidden_size)
        self.optimizer_policy = Optimizer(self.actor.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_policy)
        self.optimizer_value = Optimizer(self.critic.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)

    def pi(self, x):
        return self.actor(x)

    def v(self, x):
        return self.critic(x)

    def sample_action(self, s):
        x = torch.from_numpy(s).float()
        mu, std = self.pi(x)
        dist = Normal(mu, std)
        return dist.sample().numpy()

    def update_params(self, s, a, r, s_prime, done, entropy_coeff, overshooting_info=False):
        done_mask = 0 if done else 1
        s, a, r, s_prime, done_mask = torch.tensor(np.array(s), dtype=torch.float), torch.tensor(np.array(a)), \
                                         torch.tensor(np.array(r)), torch.tensor(np.array(s_prime), dtype=torch.float), \
                                         torch.tensor(np.array(done_mask), dtype=torch.float)

        v_s, v_prime = self.v(s), self.v(s_prime)
        td_target = r + self.gamma * v_prime * done_mask
        delta = td_target - v_s

        mu, std = self.pi(s)
        dist = Normal(mu, std)

        log_prob_pi = -(dist.log_prob(a)).sum()
        value_output = -v_s
        entropy_pi = -entropy_coeff * dist.entropy().sum() * torch.sign(delta).item()
        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()
        value_output.backward()
        (log_prob_pi + entropy_pi).backward()
        self.optimizer_policy.step(delta.item(), reset=done)
        self.optimizer_value.step(delta.item(), reset=done)

        if overshooting_info:
            v_s, v_prime = self.v(s), self.v(s_prime)
            td_target = r + self.gamma * v_prime * done_mask
            delta_bar = td_target - v_s
            if torch.sign(delta_bar * delta).item() == -1:
                print("Overshooting Detected!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', type=str,)
    parser.add_argument('--env', default='HalfCheetah-v2', type=str)
    parser.add_argument('--start_steps', default=10000, type=int)

    # DDPG parameters
    parser.add_argument('--actor_lr', default=0.001, type=float)
    parser.add_argument('--critic_lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--reward_scale', default=1., type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--layer_norm', dest='layer_norm', action='store_true')

    # TD3 parameters
    parser.add_argument('--use_td3', dest='use_td3', action='store_true')
    parser.add_argument('--policy_noise', default=0.2, type=float)
    parser.add_argument('--noise_clip', default=0.5, type=float)
    parser.add_argument('--policy_freq', default=2, type=int)

    # Gaussian noise parameters
    parser.add_argument('--gauss_sigma', default=0.1, type=float)

    # Streaming DRL parameters
    parser.add_argument('--lamda', default=0.8, type=float)
    parser.add_argument('--kappa_policy', default=3.0, type=float)
    parser.add_argument('--kappa_value', default=2.0, type=float)
    parser.add_argument('--entropy_coeff', default=0.01, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)

    # OU process parameters
    parser.add_argument('--ou_noise', dest='ou_noise', action='store_true')
    parser.add_argument('--ou_theta', default=0.15, type=float)
    parser.add_argument('--ou_sigma', default=0.2, type=float)
    parser.add_argument('--ou_mu', default=0.0, type=float)

    # ES parameters
    parser.add_argument('--pop_size', default=10, type=int)
    parser.add_argument('--elitism', dest="elitism",  action='store_true')
    parser.add_argument('--n_grad', default=5, type=int)
    parser.add_argument('--sigma_init', default=1e-3, type=float)
    parser.add_argument('--damp', default=1e-3, type=float)
    parser.add_argument('--damp_limit', default=1e-5, type=float)
    parser.add_argument('--mult_noise', dest='mult_noise', action='store_true')

    # Sampler parameters
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--epsilon', type=float, default=0)
    parser.add_argument('--k', type=int, default=1)

    # Training parameters
    parser.add_argument('--n_episodes', default=1, type=int)
    parser.add_argument('--max_steps', default=1000000, type=int)
    parser.add_argument('--mem_size', default=1000000, type=int)
    parser.add_argument('--n_noisy', default=0, type=int)

    # Testing parameters
    parser.add_argument('--filename', default="", type=str)
    parser.add_argument('--n_test', default=1, type=int)

    # misc
    parser.add_argument('--output', default='results/', type=str)
    parser.add_argument('--period', default=5000, type=int)
    parser.add_argument('--n_eval', default=10, type=int)
    parser.add_argument('--save_all_models',
                        dest="save_all_models", action="store_true")
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--render', dest='render', action='store_true')

    args = parser.parse_args()
    # args.output = get_output_folder(args.output, args.env)
    # with open(args.output + "/parameters.txt", 'w') as file:
    #     for key, value in vars(args).items():
    #         file.write("{} = {}\n".format(key, value))

    # environment
    env = gym.make(args.env)
    # env = gym.make(args.env, render_mode='None')
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = ScaleReward(env, gamma=args.gamma)
    env = NormalizeObservation(env)
    env = AddTimeInfo(env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])

    agent = StreamAC(n_obs=env.observation_space.shape[0], n_actions=env.action_space.shape[0], lr=args.actor_lr, gamma=args.gamma, lamda=args.lamda, kappa_policy=args.kappa_policy, kappa_value=args.kappa_value)
    a_noise = GaussianNoise(action_dim, sigma=args.gauss_sigma)
    entropy_coeff = args.entropy_coeff

    if USE_CUDA:
        agent.cuda()

    # CEM
    es = sepCEM(agent.actor.get_size(), mu_init=agent.actor.get_params(), sigma_init=args.sigma_init, damp=args.damp, damp_limit=args.damp_limit,
                pop_size=args.pop_size, antithetic=not args.pop_size % 2, parents=args.pop_size // 2, elitism=args.elitism)
    sampler = IMSampler(es)

    # stuff to save
    df = pd.DataFrame(columns=["total_steps", "average_score",
                               "average_score_rl", "average_score_ea", "best_score"])

    # training
    step_cpt = 0
    total_steps = 0
    actor_steps = 0
    reused_steps = 0

    es_params = []
    fitness = []
    n_steps = []
    n_start = []

    old_es_params = []
    old_fitness = []
    old_n_steps = []
    old_n_start = []

    while total_steps < args.max_steps:

        fitness = np.zeros(args.pop_size)
        n_start = np.zeros(args.pop_size)
        n_steps = np.zeros(args.pop_size)
        es_params, n_r, idx_r = sampler.ask(args.pop_size, old_es_params)
        print("Reused {} samples".format(n_r))

        actor_steps = 0
        reused_steps = 0

        # evaluate noisy actor(s)
        for i in range(args.n_noisy):
            agent.actor.set_params(es_params[i])
            agent.optimizer_policy = Optimizer(agent.actor.parameters(), lr=args.actor_lr, gamma=args.gamma, lamda=args.lamda, kappa=args.kappa_policy)
            f, steps = evaluate(agent, env, trials=1, entropy_coeff=entropy_coeff)
            actor_steps += steps
            prCyan('Noisy actor {} fitness:{}'.format(i, f))

        # evaluate all actors
        for i in range(args.pop_size):

            # evaluate new actors
            if i < args.n_grad or (i >= args.n_grad and (i - args.n_grad) >= n_r):

                agent.actor.set_params(es_params[i])
                f, steps = evaluate(agent, env, trials=1, entropy_coeff=entropy_coeff)
                actor_steps += steps

                # updating arrays
                fitness[i] = f
                n_steps[i] = steps
                # print scores
                prLightPurple('Actor {}, fitness:{}'.format(i, f))

            # reusing actors
            else:
                idx = idx_r[i - args.n_grad]
                fitness[i] = old_fitness[idx]
                n_steps[i] = old_n_steps[idx]
                n_start[i] = old_n_start[idx]

                # adding old_steps
                reused_steps += old_n_steps[idx]

                # print reused score
                prGreen('Actor {}, fitness:{}'.format(
                    i, fitness[i]))

        # update ea
        es.tell(es_params, fitness)

        # update step counts
        total_steps += actor_steps
        step_cpt += actor_steps

        # update sampler stuff
        old_fitness = deepcopy(fitness)
        old_n_steps = deepcopy(n_steps)
        old_n_start = deepcopy(n_start)
        old_es_params = deepcopy(es_params)

        # save stuff
        if step_cpt >= args.period:

            # evaluate mean actor over several runs. Memory is not filled
            # and steps are not counted
            agent.actor.set_params(es.mu)
            f_mu, _ = evaluate(agent.actor, env, trials=1, entropy_coeff=entropy_coeff)
            prRed('Actor Mu Average Fitness:{}'.format(f_mu))

            df.to_pickle(args.output + "/log.pkl")
            res = {"total_steps": total_steps,
                   "average_score": np.mean(fitness),
                   "average_score_half": np.mean(np.partition(fitness, args.pop_size // 2 - 1)[args.pop_size // 2:]),
                   "average_score_rl": np.mean(fitness[:args.n_grad]) if args.n_grad > 0 else None,
                   "average_score_ea": np.mean(fitness[args.n_grad:]),
                   "best_score": np.max(fitness),
                   "mu_score": f_mu,
                   "n_reused": n_r}

            if args.save_all_models:
                os.makedirs(args.output + "/{}_steps".format(total_steps),
                            exist_ok=True)
                agent.save_model(
                    args.output + "/{}_steps".format(total_steps), "critic")
                agent.actor.set_params(es.mu)
                agent.actor.save_model(
                    args.output + "/{}_steps".format(total_steps), "actor_mu")
            else:
                agent.critic.save_model(args.output, "critic")
                agent.actor.set_params(es.mu)
                agent.actor.save_model(args.output, "actor")
            df = df.append(res, ignore_index=True)
            step_cpt = 0
            print(res)

        print("Total steps", total_steps)
