# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool
from tqdm import tqdm

import gym
from gymnasium.spaces import Space
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from env import OpticEnv
from transformer import Seq2SeqTransformer
from stable_baselines3.common.save_util import save_to_pkl, load_from_pkl
from pathlib import Path


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, action_space=12, obs_space=34 + 48):
        super().__init__()
        self.fc1 = nn.Linear(
            action_space + obs_space, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, action_space=12, obs_space=34 + 48):
        super().__init__()
        self.fc1 = nn.Linear(obs_space, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_space)
        self.fc_logstd = nn.Linear(256, action_space)
        self.active_action_air = np.array([1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        self.active_action_lins = np.array([1.] * 11 + [0.])
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor([0.5, 3.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                         dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor([0.5, 3.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                        dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x, info=None):
        if info is not None:
            mask_action = torch.tensor([self.active_action_lins if x['type'] == 'lins' else self.active_action_air
                                        for x in info])
        else:
            mask_action = torch.tensor([self.active_action_air if x % 2 else self.active_action_lins
                                        for x in range(x.shape[0])]).type(torch.float)
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = (y_t * self.action_scale + self.action_bias) * mask_action
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = (torch.tanh(mean) * self.action_scale + self.action_bias) * mask_action
        return action, log_prob, mean


class OptPredictor:
    def __init__(self, autotune=True, alpha=0.2, target_network_frequency=1, policy_frequency=2,
                 q_lr=1e-3, policy_lr=3e-4, backbone_lr=7e-4, learning_starts=1e3, batch_size=64, gamma=0.99, tau=0.005,
                 buffer_size=1e6, total_timesteps=1000000, cuda=False, torch_deterministic=True):
        super().__init__()
        run_name = f"OpticSac__{int(time.time())}"
        self.writer = SummaryWriter(f"runs/{run_name}")
        torch.backends.cudnn.deterministic = torch_deterministic
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
        self.env = OpticEnv()
        self.max_action = 1.
        self.gamma = gamma
        self.tau = tau
        self.target_network_frequency = target_network_frequency
        self.policy_frequency = policy_frequency
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.total_timesteps = total_timesteps
        self.t = 0
        self.plato = 0
        self.padding = float('inf')
        self.best_loss = float('inf')
        self.actor = Actor()
        self.backbone = Seq2SeqTransformer(self.actor, 3, 3, 48, 6, 5, 5, 128)
        self.qf1 = SoftQNetwork()
        self.qf2 = SoftQNetwork()
        self.qf1_target = SoftQNetwork()
        self.qf2_target = SoftQNetwork()
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) +
                                      list(self.qf2.parameters()), lr=q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()),
                                          lr=policy_lr)
        self.backbone_optimizer = optim.Adam(list(self.backbone.parameters()),
                                             lr=backbone_lr)
        self.autotune = autotune
        self.metric_track = dict()
        if autotune:
            self.target_entropy = -torch.tensor(12.).to(self.device).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=q_lr)
        else:
            self.alpha = alpha
        self.rb = ReplayBuffer(int(buffer_size), spaces.Box(low=0, high=1, shape=[14, 12]),
                               spaces.Box(low=0, high=1, shape=[1, 34]),
                               self.device, handle_timeout_termination=True)

    def update_target_network(self):
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def _alpha_and_alpha_loss(self, observations):
        with torch.no_grad():
            _, log_pi, _ = self.actor.get_action(observations)
        alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()
        self.metric_track.update({'alpha_loss': alpha_loss.item()})
        return alpha_loss

    def _policy_loss(self, observations):
        pi, log_pi, _ = self.actor.get_action(observations)
        qf1_pi = self.qf1(observations, pi)
        qf2_pi = self.qf2(observations, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        self.metric_track.update({'actor_loss': actor_loss.item(), 'alpha': self.alpha})
        return actor_loss

    def _q_loss(self, observations, actions, rewards, next_observations, dones):
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(next_observations)
            qf1_next_target = self.qf1_target(next_observations, next_state_actions)
            qf2_next_target = self.qf2_target(next_observations, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards.flatten() + (1 - dones.flatten()) * self.gamma * min_qf_next_target.view(
                -1)
        qf1_a_values = self.qf1(observations, actions).view(-1)
        qf2_a_values = self.qf2(observations, actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        self.metric_track.update({'qf1_values': qf1_a_values.mean().item(), 'qf2_values': qf2_a_values.mean().item(),
                                  'qf1_loss': qf1_loss.item(), 'qf2_loss': qf2_loss.item(),
                                  'qf_loss': qf_loss.item() / 2})
        return qf_loss

    def train(self, batch):
        (
            observations,
            actions,
            next_observations,
            dones,
            rewards,
        ) = batch
        self.metric_track = dict()
        self.metric_track.update({'reward': torch.mean(rewards).item()})
        self.t += 1
        # print(type(observations[0][-1][-1]))
        len_observations = list()
        for obs in observations:
            for i, vec in enumerate(obs):
                if self.padding == vec[0].item():
                    break
            if i == 13:
                i = 14
            len_observations.append(i)

        observations = [observations[i][:len_observations[i]] for i in range(observations.shape[0])]

        len_actions = list()
        for obs in next_observations:
            for i, vec in enumerate(obs):
                if self.padding == vec[0].item():
                    break
            if i == 13:
                i = 14
            len_actions.append(i)
        next_observations = [next_observations[i][:len_actions[i]] for i in range(next_observations.shape[0])]
        # observations = list(map(lambda y: torch.tensor(list(filter(lambda x: self.padding not in x, y))), observations))

        # next_observations = list(map(lambda y: torch.tensor(list(filter(lambda x: self.padding not in x, y))),
        #                              next_observations))
        # len_actions = list(map(len, next_observations))
        full_observation = self.backbone.decode_with_target(observations, actions, next_observations)
        print(full_observation.shape)
        actions = torch.cat(next_observations)
        full_observation = full_observation[:, :-1, :]
        observations = full_observation[:, :-1, :]
        next_observations = full_observation[:, 1:, :]
        observations = torch.cat([observations[i][:len_actions[i]] for i in range(observations.shape[0])])
        next_observations = torch.cat(
            [next_observations[i][:len_actions[i]] for i in range(next_observations.shape[0])])
        dones = torch.cat([torch.tensor([0.] * (len_act - 1) + [1.]) for len_act in len_actions])
        rewards = torch.cat(
            [torch.tensor([0.] * (len_act - 1) + [rewards[i]]) for i, len_act in enumerate(len_actions)])
        qf_loss = self._q_loss(observations, actions, rewards, next_observations, dones)

        self.backbone_optimizer.zero_grad()
        self.q_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.q_optimizer.step()

        """ Q function loss """

        if self.t % self.policy_frequency == 0:
            for _ in range(self.policy_frequency):
                policy_loss = self._policy_loss(observations)
                self.actor_optimizer.zero_grad()
                policy_loss.backward(retain_graph=True)
                self.actor_optimizer.step()

                if self.autotune:
                    alpha_loss = self._alpha_and_alpha_loss(observations)
                    self.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()

        self.backbone_optimizer.step()
        if self.t % self.target_network_frequency == 0:
            self.update_target_network()

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.qf1.state_dict(),
            "critic2": self.qf2.state_dict(),
            "critic1_target": self.qf1_target.state_dict(),
            "critic2_target": self.qf2_target.state_dict(),
            "critic_optimizer": self.q_optimizer.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "sac_log_alpha": self.log_alpha,
            "sac_log_alpha_optim": self.a_optimizer.state_dict(),
            "transformer": self.backbone.state_dict(),
            "transformer_optimizer": self.backbone_optimizer.state_dict(),
            "total_it": self.t,
        }

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict=state_dict["actor"])
        self.qf1.load_state_dict(state_dict=state_dict["critic1"])
        self.qf2.load_state_dict(state_dict=state_dict["critic2"])

        self.qf1_target.load_state_dict(state_dict=state_dict["critic1_target"])
        self.qf2_target.load_state_dict(state_dict=state_dict["critic2_target"])

        self.q_optimizer.load_state_dict(
            state_dict=state_dict["critic_optimizer"]
        )
        self.actor_optimizer.load_state_dict(state_dict=state_dict["actor_optim"])

        self.log_alpha = state_dict["sac_log_alpha"]
        self.a_optimizer.load_state_dict(
            state_dict=state_dict["sac_log_alpha_optim"]
        )
        self.backbone.load_state_dict(state_dict=state_dict['transformer'])
        self.backbone_optimizer.load_state_dict(state_dict=state_dict['transformer_optimizer'])
        self.t = state_dict["total_it"]

    def save_info(self):
        [self.writer.add_scalar(f"losses/{key}", value, self.t) for
         key, value in self.metric_track.items()]

    def get_data(self, feature_env, feature_loss):
        if self.t < self.learning_starts or self.plato > 10:
            actions = self.env.sample()
            # full_obs = self.backbone.decode_with_target(feature_env, feature_loss, torch.tensor(actions))
            # full_obs = full_obs[:, :-1, :]
        else:
            actions, full_obs = self.backbone.decode_without_target(feature_env, feature_loss)
            actions = actions.squeeze().detach().cpu().numpy()
        # obs = full_obs[:, :-1, :].squeeze()
        # next_obs = full_obs[:, 1:, :].squeeze()
        obs = feature_env.detach().numpy()
        loss_obs = feature_loss.reshape(1, -1)
        # TRY NOT TO MODIFY: execute the game and log data.
        return_feature_env, return_feature_loss, loss, reward = self.env.step_m1(actions)
        if obs.shape[0] < 14:
            obs = np.concatenate([obs, np.array([[self.padding] * (obs.shape[1])] * (14 - obs.shape[0]))])
        if actions.shape[0] < 14:
            actions = np.concatenate(
                [actions, np.array([[self.padding] * (actions.shape[1])] * (14 - actions.shape[0]))])

        if loss is None:
            return_feature_env, return_feature_loss, loss = feature_env, feature_loss, float('inf')
        elif loss < self.best_loss:
            self.plato = 0
            self.best_loss = loss
        elif self.plato > 10 or self.t < self.learning_starts:
            self.plato = 0
            self.best_loss = loss
        else:
            self.plato += 1
        # dones = np.array([False] * (actions.shape[1] - 1) + [True])
        # infos = [{'type': 'lins'}, {'type': 'air'}] * (actions.shape[1] // 2)
        # rewards = [0.] * (actions.shape[1] - 1) + [reward]

        # TRY NOT TO MODIFY: record rewards for plotting purposes

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        # print(obs.shape)
        # print(actions.shape)
        self.rb.add(obs, actions, loss_obs, reward, np.array([True]), [{}])

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        return torch.tensor(return_feature_env), torch.tensor(return_feature_loss), loss

    def start(self, load_model=False, load_rb=False):
        if load_rb:
            self.rb = load_from_pkl('sac_replay_buffer', 0)

        if load_model:
            policy_file = Path(os.path.join('model', f"checkpoint_{self.t}.pt"))
            self.load_state_dict(torch.load(policy_file))

        feature_env, feature_loss, self.best_loss = self.env.reset()
        save_loss = self.best_loss
        for _ in tqdm(range(self.total_timesteps)):
            self.t += 1
            feature_env, feature_loss, loss = self.get_data(feature_env, feature_loss)
            if loss < save_loss:
                with open('best_5.txt', 'a') as file:
                    file.write(f'{loss=}, best_param={feature_env} \n')
                save_loss = loss

            if self.t % 100 == 0:
                save_to_pkl('sac_replay_buffer', self.rb, 0)

            if self.t > self.learning_starts:
                self.train(self.rb.sample(self.batch_size))
                if self.t % 10 == 0:
                    self.save_info()
                if self.t % 100 == 0:
                    torch.save(
                        self.state_dict(),
                        os.path.join('model', f"checkpoint_{self.t}.pt"),
                    )
        self.writer.close()
