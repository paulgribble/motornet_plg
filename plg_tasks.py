import copy
import numpy as np
import tensorflow as tf
from motornet.nets.losses import PositionLoss, L2xDxActivationLoss, L2xDxRegularizer
from motornet.tasks import Task, CentreOutReach
from typing import Union
import time

class CentreOutFF(CentreOutReach):
    def __init__(
            self,
            network,
            name: str = 'CentreOutFF',
            **kwargs
    ):
        super().__init__(network, **kwargs)
        self.FF_matvel = tf.convert_to_tensor(kwargs.get('FF_matvel', np.array([[0,0],[0,0]])), dtype=tf.float32)

    def generate(self, batch_size, n_timesteps, condition="test"):
        """
        condition = "train": learn to reach to random targets in workspace in a NF
                    "test" : centre-out reaches to each target in a given FF/NF
                    "adapt": re-learn centre-out reaches in a given FF/NF
        """
        catch_trial = np.zeros(batch_size, dtype='float32')
        if (condition=="train"): # train net to reach to random targets in workspace in a NF
            init_states   = self.get_initial_state(batch_size=batch_size)
            goal_states_j = self.network.plant.draw_random_uniform_states(batch_size=batch_size)
            goal_states   = self.network.plant.joint2cartesian(goal_states_j)
            p             = int(np.floor(batch_size * self.catch_trial_perc / 100))
            catch_trial[np.random.permutation(catch_trial.size)[:p]] = 1.
            
        elif (condition=="test"): # centre-out reaches to each target in a given FF/NF
            angle_set   = np.deg2rad(np.arange(0, 360, self.angular_step))
            reps        = int(np.ceil(batch_size / len(angle_set)))
            angle       = np.tile(angle_set, reps=reps)
            batch_size  = reps * len(angle_set)
            start_jpv   = np.concatenate([self.start_position, np.zeros_like(self.start_position)], axis=1)
            start_cpv   = self.network.plant.joint2cartesian(start_jpv)
            end_cp      = self.reaching_distance * np.stack([np.cos(angle), np.sin(angle)], axis=-1)
            init_states = self.network.get_initial_state(batch_size=batch_size, inputs=start_jpv)
            goal_states = start_cpv + np.concatenate([end_cp, np.zeros_like(end_cp)], axis=-1)
            catch_trial = np.zeros(batch_size, dtype='float32')

        elif (condition=="adapt"): # re-learn centre-out reaches in a given FF/NF
            angle_set   = np.deg2rad(np.arange(0, 360, self.angular_step))
            reps        = int(np.ceil(batch_size / len(angle_set)))
            angle       = np.tile(angle_set, reps=reps)
            batch_size  = reps * len(angle_set)
            start_jpv   = np.concatenate([self.start_position, np.zeros_like(self.start_position)], axis=1)
            start_cpv   = self.network.plant.joint2cartesian(start_jpv)
            end_cp      = self.reaching_distance * np.stack([np.cos(angle), np.sin(angle)], axis=-1)
            init_states = self.network.get_initial_state(batch_size=batch_size, inputs=start_jpv)
            goal_states = start_cpv + np.concatenate([end_cp, np.zeros_like(end_cp)], axis=-1)
            p             = int(np.floor(batch_size * self.catch_trial_perc / 100))
            catch_trial[np.random.permutation(catch_trial.size)[:p]] = 1.

        startpos     = self.network.plant.joint2cartesian(init_states[0][:, :])
        go_cue       = np.ones([batch_size, n_timesteps, 1])
        targets      = self.network.plant.state2target(state=goal_states, n_timesteps=n_timesteps).numpy()
        inputs_targ  = copy.deepcopy(targets[:, :, :self.network.plant.space_dim])
        tmp          = np.repeat(startpos[:, np.newaxis, :self.network.plant.space_dim], n_timesteps, axis=1)
        inputs_start = copy.deepcopy(tmp)
        for i in range(batch_size):
            if ((condition=="train") or (condition=="adapt")):
                go_cue_time = int(np.random.uniform(self.go_cue_range[0], self.go_cue_range[1]))
            elif (condition=="test"):
                go_cue_time = int(self.go_cue_range[0] + np.diff(self.go_cue_range) / 2)
            if catch_trial[i] > 0.:
                targets[i, :, :] = startpos[i, np.newaxis, :]
            else:
                targets[i, :go_cue_time, :] = startpos[i, np.newaxis, :]
                inputs_start[i, go_cue_time + self.network.visual_delay:, :] = 0.
                go_cue[i, go_cue_time + self.network.visual_delay:, 0] = 0.

        return [
            {"inputs": np.concatenate([inputs_start, inputs_targ, go_cue], axis=-1)},
            self.convert_to_tensor(targets), init_states
        ]

    def recompute_inputs(self, inputs, states):
        jstate, cstate, mstate, gstate = self.network.unpack_plant_states(states)
        inputs['endpoint_load'] = tf.transpose(tf.matmul(self.FF_matvel, tf.transpose(cstate[:, 2:4])))  # [2x2] x [2xbatch] = [2xbatch]
        return inputs

