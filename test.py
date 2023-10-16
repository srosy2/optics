import unittest

import torch

from env import OpticEnv
from rl_model import Actor
from transformer import Seq2SeqTransformer


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


if __name__ == '__main__':
    unittest.main()
