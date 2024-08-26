from lrhc_control.training_algs.dummy.dummy_test_algo_base import DummyTestAlgoBase

import torch 
import torch.nn as nn
import torch.nn.functional as F

import os

import time

class Dummy(DummyTestAlgoBase):

    def __init__(self,
            env, 
            debug = False,
            remote_db = False,
            seed: int = 1):

        super().__init__(env=env, 
                    debug=debug,
                    remote_db=remote_db,
                    seed=seed)

        self._this_child_path = os.path.abspath(__file__) # overrides parent
    
    def _collect_transition(self):
        
        # experience collection
        self._switch_training_mode(train=False)

        obs = self._env.get_obs(clone=True) # we beed cloned obs (they are modified upon env stepping by the
        # env itself

        actions = self._agent.get_action(x=obs)
        actions = actions.detach()
        env_step_ok = self._env.step(actions)

        return env_step_ok