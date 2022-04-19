import copy
import numpy as np
import tensorflow as tf
from motornet.nets.losses import PositionLoss, L2xDxActivationLoss, L2xDxRegularizer
from motornet.tasks import Task, CentreOutReach

class CentreOutReachFF(Task):
    def __init__(self, network, **kwargs):
        super().__init__(network, **kwargs)
        self.__name__ = 'CentreOutReach'

        self.angle_step = kwargs.get('reach_angle_step_deg', 15)
        self.catch_trial_perc = kwargs.get('catch_trial_perc', 50)
        self.reaching_distance = kwargs.get('reaching_distance', 0.1)
        self.start_position = kwargs.get('start_joint_position', None)
        if not self.start_position:
            # start at the center of the workspace
            lb = np.array(self.network.plant.pos_lower_bound)
            ub = np.array(self.network.plant.pos_upper_bound)
            self.start_position = lb + (ub - lb) / 2
        self.start_position = np.array(self.start_position).reshape(1, -1)

        deriv_weight = kwargs.get('deriv_weight', 0.)
        max_iso_force = self.network.plant.muscle.max_iso_force
        dt = self.network.plant.dt
        muscle_loss = L2xDxActivationLoss(max_iso_force=max_iso_force, dt=dt, deriv_weight=deriv_weight)
        gru_loss = L2xDxRegularizer(deriv_weight=0.05, dt=self.network.plant.dt)
        self.add_loss('gru_hidden0', loss_weight=0.1, loss=gru_loss)
        self.add_loss('muscle state', loss_weight=5, loss=muscle_loss)
        self.add_loss('cartesian position', loss_weight=1., loss=PositionLoss())

        go_cue_range = np.array(kwargs.get('go_cue_range', [0.05, 0.25])) / dt
        self.go_cue_range = [int(go_cue_range[0]), int(go_cue_range[1])]
        self.delay_range = self.go_cue_range

        self.FF_matvel = tf.convert_to_tensor(kwargs.get('FF_matvel', np.array([[0,0],[0,0]])), dtype=tf.float32)

    def generate(self, batch_size, n_timesteps, **kwargs):
        catch_trial = np.zeros(batch_size, dtype='float32')
        validation = kwargs.get('validation', False)
        if not validation:
            init_states = self.get_initial_state(batch_size=batch_size)
            goal_states_j = self.network.plant.draw_random_uniform_states(batch_size=batch_size)
            goal_states = self.network.plant.joint2cartesian(goal_states_j)
            p = int(np.floor(batch_size * self.catch_trial_perc / 100))
            catch_trial[np.random.permutation(catch_trial.size)[:p]] = 1.
        else:
            angle_set = np.deg2rad(np.arange(0, 360, self.angle_step))
            reps = int(np.ceil(batch_size / len(angle_set)))
            angle = np.tile(angle_set, reps=reps)
            batch_size = reps * len(angle_set)
            catch_trial = np.zeros(batch_size, dtype='float32')

            start_jpv = np.concatenate([self.start_position, np.zeros_like(self.start_position)], axis=1)
            start_cpv = self.network.plant.joint2cartesian(start_jpv)
            end_cp = self.reaching_distance * np.stack([np.cos(angle), np.sin(angle)], axis=-1)
            init_states = self.network.get_initial_state(batch_size=batch_size, inputs=start_jpv)
            goal_states = start_cpv + np.concatenate([end_cp, np.zeros_like(end_cp)], axis=-1)

        center = self.network.plant.joint2cartesian(init_states[0][:, :])
        go_cue = np.ones([batch_size, n_timesteps, 1])
        targets = self.network.plant.state2target(state=goal_states, n_timesteps=n_timesteps).numpy()
        inputs_targ = copy.deepcopy(targets[:, :, :self.network.plant.space_dim])
        inputs_start = copy.deepcopy(np.repeat(center[:, np.newaxis, :self.network.plant.space_dim], n_timesteps, axis=1))
        for i in range(batch_size):
            if not validation:
                go_cue_time = int(np.random.uniform(self.go_cue_range[0], self.go_cue_range[1]))
            else:
                go_cue_time = int(0.150 / self.network.plant.dt)

            if catch_trial[i] > 0.:
                targets[i, :, :] = center[i, np.newaxis, :]
            else:
                targets[i, :go_cue_time, :] = center[i, np.newaxis, :]
                inputs_start[i, go_cue_time + self.network.visual_delay:, :] = 0.
                go_cue[i, go_cue_time + self.network.visual_delay:, 0] = 0.

        return [{"inputs": np.concatenate([inputs_start, inputs_targ, go_cue], axis=-1)},
                self.convert_to_tensor(targets), init_states]

    def recompute_inputs(self, inputs, states):
        jstate, cstate, mstate, gstate = self.network.unpack_plant_states(states)
        inputs['endpoint_load'] = tf.transpose(tf.matmul(self.FF_matvel, tf.transpose(cstate[:, 2:4])))  # [2x2] x [2xbatch] = [2xbatch]
        return inputs


class Gribble1999(Task):
    # Gribble PL, Ostry DJ (1999)
    # Compensation for interaction torques during single- and multijoint limb movement.
    # J Neurophysiol 82:2310–2326

    def __init__(self, network, **kwargs):
        super().__init__(network, **kwargs)
        self.__name__ = 'Gribble1999'

        self.angle_step = kwargs.get('reach_angle_step_deg', 15)
        self.catch_trial_perc = kwargs.get('catch_trial_perc', 50)
        self.reaching_distance = kwargs.get('reaching_distance', 0.1)
        self.start_position = kwargs.get('start_joint_position', None)
        if not self.start_position:
            # start at the center of the workspace
            lb = np.array(self.network.plant.pos_lower_bound)
            ub = np.array(self.network.plant.pos_upper_bound)
            self.start_position = lb + (ub - lb) / 2
        self.start_position = np.array(self.start_position).reshape(1, -1)

        deriv_weight = kwargs.get('deriv_weight', 0.)
        max_iso_force = self.network.plant.muscle.max_iso_force
        dt = self.network.plant.dt
        muscle_loss = L2xDxActivationLoss(max_iso_force=max_iso_force, dt=dt, deriv_weight=deriv_weight)
        gru_loss = L2xDxRegularizer(deriv_weight=0.05, dt=self.network.plant.dt)
        self.add_loss('gru_hidden0', loss_weight=0.1, loss=gru_loss)
        self.add_loss('muscle state', loss_weight=5, loss=muscle_loss)
        self.add_loss('cartesian position', loss_weight=1., loss=PositionLoss())

        go_cue_range = np.array(kwargs.get('go_cue_range', [0.100, 0.300])) / dt
        self.go_cue_range = [int(go_cue_range[0]), int(go_cue_range[1])]
        self.delay_range = self.go_cue_range

    def generate(self, batch_size, n_timesteps, **kwargs):
        catch_trial = np.zeros(batch_size, dtype='float32')
        experiment = kwargs.get('experiment', False)
        if experiment not in [1,2,3]:
            init_states = self.get_initial_state(batch_size=batch_size)
            goal_states_j = self.network.plant.draw_random_uniform_states(batch_size=batch_size)
            goal_states = self.network.plant.joint2cartesian(goal_states_j)
            p = int(np.floor(batch_size * self.catch_trial_perc / 100))
            catch_trial[np.random.permutation(catch_trial.size)[:p]] = 1.
        else: # Gribble 1999 experimental conditions (s,e)
            # Experiment 1: elbow alone    : (50,60) deg + (20,40,60) deg elbow flexion
            # Experiment 2: shoulder alone : (10,80) deg + (20,40,60) deg shoulder flexion
            # Experiment 3: elbow,shoulder combinations : (50,70) + (+20,+30), (+20,-30), (-20,+30), (-20,-30)

            if (experiment==1):
                start_jpv = tf.repeat(np.array([[50,60,0,0]])*np.pi/180, 3, axis=0)
                start_cpv = self.network.plant.joint2cartesian(start_jpv)
                mov_jpv = np.array([[0,20,0,0],[0,40,0,0],[0,60,0,0]])*np.pi/180
                end_jpv = start_jpv + mov_jpv
                end_cpv = self.network.plant.joint2cartesian(end_jpv)
            elif (experiment==2):
                start_jpv = tf.repeat(np.array([[10,80,0,0]])*np.pi/180, 3, axis=0)
                start_cpv = self.network.plant.joint2cartesian(start_jpv)
                mov_jpv = np.array([[30,0,0,0],[50,0,0,0],[70,0,0,0]])*np.pi/180
                end_jpv = start_jpv + mov_jpv
                end_cpv = self.network.plant.joint2cartesian(end_jpv)
            elif (experiment==3):
                start_jpv = tf.repeat(np.array([[50,70,0,0]])*np.pi/180, 4, axis=0)
                start_cpv = self.network.plant.joint2cartesian(start_jpv)
                mov_jpv = np.array([[20,30,0,0],[20,-30,0,0],[-20,30,0,0],[-20,-30,0,0]])*np.pi/180
                end_jpv = start_jpv + mov_jpv
                end_cpv = self.network.plant.joint2cartesian(end_jpv)

            batch_size = tf.shape(start_cpv)[0]
            catch_trial = np.zeros(batch_size, dtype='float32') # catch trial flags to zero
            init_states = self.network.get_initial_state(batch_size=batch_size, inputs=start_jpv)
            goal_states = end_cpv

        start_cp = self.network.plant.joint2cartesian(init_states[0][:, :])
        go_cue = np.ones([batch_size, n_timesteps, 1])
        targets = self.network.plant.state2target(state=goal_states, n_timesteps=n_timesteps).numpy()
        inputs_targ = copy.deepcopy(targets[:, :, :self.network.plant.space_dim])
        inputs_start = copy.deepcopy(np.repeat(start_cp[:, np.newaxis, :self.network.plant.space_dim], n_timesteps, axis=1))
        for i in range(batch_size):
            if experiment not in [1,2,3]:
                go_cue_time = int(np.random.uniform(self.go_cue_range[0], self.go_cue_range[1]))
            else:
                go_cue_time = int(self.go_cue_range[0] + np.diff(self.go_cue_range) / 2)

            if catch_trial[i] > 0.:
                targets[i, :, :] = start_cp[i, np.newaxis, :]
            else:
                targets[i, :go_cue_time, :] = start_cp[i, np.newaxis, :]
                inputs_start[i, go_cue_time + self.network.visual_delay:, :] = 0.
                go_cue[i, go_cue_time + self.network.visual_delay:, 0] = 0.

        return [{"inputs": np.concatenate([inputs_start, inputs_targ, go_cue], axis=-1)},
                self.convert_to_tensor(targets), init_states]

