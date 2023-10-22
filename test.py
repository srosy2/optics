import time
import unittest

import torch

from env import OpticEnv
from rl_model import Actor, SoftQNetwork
from transformer import Seq2SeqTransformer
from ray.util.multiprocessing import Pool
import ray
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_reset_env(self):
        env = OpticEnv()
        feature_env, feature_loss, loss = env.reset()
        self.assertEqual(feature_env.shape[1], 12)
        print(feature_loss.shape)
        print(loss)

    def test_env_sample(self):
        env = OpticEnv()
        feature_env, feature_loss, loss = env.reset()
        actions = env.sample()
        print(actions)
        return_feature_env, return_feature_loss, loss, reward = env.step_m1(actions)
        print(reward, return_feature_env.shape, return_feature_loss.shape)

    def test_transformer_with_target(self):
        actor = Actor()
        model = Seq2SeqTransformer(actor, 3, 3, 48, 6, 5, 5, 128)
        env = OpticEnv()
        feature_env, feature_loss, loss = env.reset()
        actions = env.sample()
        full_obs = model.decode_with_target(feature_env, feature_loss, torch.tensor(actions))
        full_obs = full_obs.squeeze()
        full_obs = full_obs[:-1, :]
        print(full_obs.shape)

    def test_transformer_without_target(self):
        actor = Actor()
        model = Seq2SeqTransformer(actor, 3, 3, 48, 6, 5, 5, 128)
        env = OpticEnv()
        feature_env, feature_loss, loss = env.reset()
        actions, full_obs = model.decode_without_target(feature_env, feature_loss)
        print(full_obs.shape, actions.shape)

    def test_multiprocessing(self):
        ray.init(num_cpus=6, _temp_dir="/Users/19936244/test",
                 include_dashboard=False, ignore_reinit_error=True)
        # from multiprocessing import Pool
        pool = Pool(5)
        # with Pool(5) as pool:
        start = time.time()
        pool.starmap(test,  [[1, 1], [2, 2]])
        print(f'time {time.time() - start}')
        start = time.time()
        pool.map(test, *zip([range(10), range(10)]))
        print(f'time {time.time() - start}')
        test(1, 1)


def test(n, b):
    print(n, b)
    return 1


if __name__ == '__main__':
    import torch.optim as optim
    with open('mistakes.npy', 'rb') as f:
        x = np.load(f)
    with open('min_qf.npy', 'rb') as f:
        y = np.load(f)
    print(y)
    print(np.min(np.abs(y)))
    print(np.min(np.abs(x)))
    x = torch.tensor(x)
    print(x)
    act = Actor()
    critic = SoftQNetwork()
    pi, log_pi, _ = act.get_action(x)
    actor_optimizer = optim.Adam(list(act.parameters()),
                                      lr=3e-4)
    print(torch.max(torch.abs(pi)), torch.min(torch.abs(pi)))
    print(torch.max(torch.abs(log_pi)), torch.min(torch.abs(log_pi)))
    print((pi==pi).all(), (log_pi==log_pi).all())
    qf1_pi = critic(x, pi)
    actor_loss = ((0.2 * log_pi) - qf1_pi).mean()
    print(actor_loss)
    actor_optimizer.zero_grad()
    actor_loss.backward(retain_graph=True)
    actor_optimizer.step()
    with torch.no_grad():
        _, log_pi, _ = act.get_action(x)
    # pi, log_pi, _ = act.get_action(x)
    print((log_pi == log_pi).all())
    # unittest.main()
