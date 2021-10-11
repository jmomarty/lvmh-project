import unittest
from thompson_sampling import ThompsonSampler
from recommendation_system import ThompsonSamplingRecSys
import numpy as np

#unittest.TestLoader.sortTestMethodsUsing = None


class TestThompsonSampling(unittest.TestCase):

    def test_thompson_sampler_class(self):
        ts = ThompsonSampler()

        N = np.random.randint(1, 100)
        ts = ThompsonSampler(N)
        assert ts.number_of_arms == N
        assert hasattr(ts, 'alphas')
        assert hasattr(ts, 'betas')
        assert ts.alphas.shape == (ts.number_of_arms,)
        assert ts.betas.shape == (ts.number_of_arms,)
        assert np.allclose(ts.alphas, np.ones(ts.number_of_arms))
        assert np.allclose(ts.betas, np.ones(ts.number_of_arms))

        assert hasattr(ts, '_sample_from_beta_dist')
        assert hasattr(ts, 'choose_arm')

        ts.choose_arm()
        assert ts.thetas is not None
        assert ts.thetas.shape == (ts.number_of_arms,)
        ts.choose_arm()
        assert hasattr(ts, 'chosen_arm')
        assert ts.chosen_arm in range(ts.number_of_arms)
        assert ts.chosen_arm == np.argmax(ts.thetas)

        assert hasattr(ts, 'update_beta_dist')
        assert hasattr(ts, 'get_reward')

        reward = np.random.randint(0, 1)
        ts.get_reward(reward)
        assert hasattr(ts, 'current_reward')
        assert ts.current_reward == reward
        ts.get_reward(1 - reward)
        assert ts.current_reward == 1 - reward

        reward = np.random.randint(0, 1)
        ts.get_reward(reward)
        alphas, betas = ts.alphas.copy(), ts.betas.copy()
        ts.update_beta_dist()
        assert any(np.concatenate([ts.alphas, ts.betas]) != np.concatenate([alphas, betas])), \
        f"""current alphas : {ts.alphas} \n current betas : {ts.betas}
            last alphas : {ts.alphas} \n last betas : {ts.betas}
        """

        if reward:
            assert ts.alphas[ts.chosen_arm] > alphas[ts.chosen_arm]
            assert ts.betas[ts.chosen_arm] == betas[ts.chosen_arm]
        else:
            assert ts.betas[ts.chosen_arm] > betas[ts.chosen_arm]
            assert ts.alphas[ts.chosen_arm] == alphas[ts.chosen_arm]

    def test_recsys_class(self):

        rs = ThompsonSamplingRecSys(config_fn='test_config.json')
        assert hasattr(rs, 'ts')
        assert hasattr(rs, '_load_config')
        assert hasattr(rs, 'config')
        assert hasattr(rs, 'rec_files')
        assert hasattr(rs, '_parse_rec_files')
        assert hasattr(rs, 'rec_files')
        assert hasattr(rs, 'recs')
        assert rs.ts.number_of_arms == len(rs.recs.keys())

        assert hasattr(rs, 'recommend')
        reco = rs.recommend()
        print(reco)
        rs.get_feedback(1)
        reco = rs.recommend()
        print(reco)
        rs.get_feedback(0)
        reco = rs.recommend()
        print(reco)
        rs.get_feedback(0)
        reco = rs.recommend()
        print(reco)


if __name__ == '__main__':
    unittest.main()
