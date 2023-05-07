import os
import numpy as np
import random
import time
from copy import copy
from collections import deque
import argparse
import tensorflow as tf
import tensorflow.keras.backend as K
import geopy.distance
from bluesky.tools import geo
from operator import itemgetter
from shapely.geometry import LineString, Point
from vis.utils import utils
import numba as nb
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

################################
##                            ##
##      Marc Brittain         ##
##  marcbrittain.github.io    ##
##                            ##
################################


LOSS_CLIPPING = 0.2
ENTROPY_LOSS = 1e-4
HIDDEN_SIZE = 32

import time


@nb.njit()
def discount(r, discounted_r, cumul_r):
    """ Compute the gamma-discounted rewards over an episode
    """
    for t in range(len(r) - 1, -1, -1):
        cumul_r = r[t] + cumul_r * 0.99
        discounted_r[t] = cumul_r
    return discounted_r


def dist_goal(states, traf, i):
    olat, olon = states
    ilat, ilon = traf.ap.route[i].wplat[0], traf.ap.route[i].wplon[0]
    dist = geo.latlondist(olat, olon, ilat, ilon) / geo.nm
    return dist


def getClosestAC_Distance(self, state, traf, route_keeper):
    olat, olon, ID = state[:3]
    index = int(ID[2:])
    rte = int(route_keeper[index])
    lat, lon, glat, glon, h = self.positions[rte]
    size = traf.lat.shape[0]
    index = np.arange(size).reshape(-1, 1)
    ownship_obj = LineString([[olon, olat, 31000], [glon, glat, 31000]])
    # line_test = LineString([[olon, olat], [glon, glat]])
    # plt.plot(line_test)
    d = geo.latlondist_matrix(np.repeat(olat, size), np.repeat(olon, size), traf.lat, traf.lon)
    d = d.reshape(-1, 1)

    dist = np.concatenate([d, index], axis=1)

    dist = sorted(np.array(dist), key=itemgetter(0))[1:]
    if len(dist) > 0:
        for i in range(len(dist)):

            index = int(dist[i][1])
            ID_ = traf.id[index]
            index_route = int(ID_[2:])

            rte_int = route_keeper[index_route]
            lat, lon, glat, glon, h = self.positions[rte_int]
            int_obj = LineString([[traf.lon[index], traf.lat[index], 31000], [glon, glat, 31000]])

            if not ownship_obj.intersects(int_obj):
                continue

            if not rte_int in self.intersection_distances[rte].keys() and rte_int != rte:
                continue

            if dist[i][0] > 100:
                continue

            return dist[i][0]


    else:
        return np.inf

    return np.inf


def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        r = prob / (old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING,
                                                       max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(
                prob * K.log(prob + 1e-10)))

    return loss


# initalize the PPO agent
class PPO_Agent:
    def __init__(self, state_size, action_size, num_routes, numEpisodes, positions, num_intruders):

        self.state_size = state_size
        self.action_size = action_size
        self.positions = positions
        self.gamma = 0.99  # discount rate
        self.numEpisodes = numEpisodes
        self.max_time = 500
        self.num_intruders = num_intruders

        self.episode_count = 0
        self.speeds = np.array([156, 0, 346])
        self.max_agents = 0
        self.num_routes = num_routes
        self.experience = {}
        self.dist_close = {}
        self.dist_goal = {}
        self.tas_max = 253.39054470774
        self.tas_min = 118.54804803287088
        self.lr = 0.0001
        self.value_size = 1
        self.getRouteDistances()
        self.model_check = []
        self.model = self._build_PPO()
        self.model_saliency = self._build_PPO_saliency()

        self.count = 0
        self.intersect_time = 0  # runtime of 'intersects' function
        self.calState_time = 0  # runtime of 'getClosestAC' function
        self.dist_time = 0  # runtime of distance calculation function
        self.intruder_time = 0
        self.forward_time = 0

    def getRouteDistances(self):
        self.intersections = {}
        self.intersection_distances = {}
        self.route_distances = []
        self.conflict_routes = {}
        for i in range(len(self.positions)):
            olat, olon, glat, glon, h = self.positions[i]
            _, d = geo.qdrdist(olat, olon, glat, glon)
            self.route_distances.append(d)
            own_obj = LineString([[olon, olat, 31000], [glon, glat, 31000]])
            self.conflict_routes[i] = []
            for j in range(len(self.positions)):
                if i == j: continue
                olat, olon, glat, glon, h = self.positions[j]
                other_obj = LineString([[olon, olat, 31000], [glon, glat, 31000]])
                self.conflict_routes[i].append(j)
                if own_obj.intersects(other_obj):
                    intersect = own_obj.intersection(other_obj)
                    try:
                        Ilon, Ilat, alt = list(list(intersect.boundary[0].coords)[0])
                    except:
                        Ilon, Ilat, alt = list(list(intersect.coords)[0])

                    try:
                        self.intersections[i][j] = [Ilat, Ilon]
                        # self.intersections[i].append([j, [Ilat, Ilon]])
                    except:
                        self.intersections[i] = {j: [Ilat, Ilon]}
                        # self.intersections[i] = [[j, [Ilat, Ilon]]]

        for route in self.intersections.keys():
            olat, olon, glat, glon, h = self.positions[i]

            for intersection in self.intersections[route].keys():
                conflict_route, location = intersection, self.intersections[route][intersection]
                Ilat, Ilon = location
                _, d = geo.qdrdist(Ilat, Ilon, glat, glon)
                try:
                    self.intersection_distances[route][conflict_route] = d
                except:
                    self.intersection_distances[route] = {conflict_route: d}

        # import pickle
        # with open('intersections_E2.pkl', 'wb') as f:
        #     pickle.dump(self.intersections, f)
        # f.close()
        self.max_d = max(self.route_distances)

    def normalize_that(self, value, what, context=False, state=False, id_=None):

        if what == 'spd':

            if value > self.tas_max:
                self.tas_max = value

            if value < self.tas_min:
                self.tas_min = value
            return (value - self.tas_min) / (self.tas_max - self.tas_min)

        if what == 'rt':
            return value / (self.num_routes - 1)

        if what == 'state':
            dgoal = self.dist_goal[id_] / self.max_d  # distance to the goal
            spd = self.normalize_that(value[2], 'spd')  # airspeed
            rt = self.normalize_that(value[3], 'rt')  # route identifier
            acc = value[4] + 0.5  # why plus 0.5??? [-0.5,0,0.5] --> [0,0.5,1]
            rt_own = int(value[3])
            # [distance to the goal, airspeed, route identifier, acceleration,
            # loss of separation], see the definition of s_t^o in the "attention
            # network version of separation assurance" paper(Page 11) for reference
            norm_array = np.array([dgoal, spd, rt, acc, 3 / self.max_d])

            return norm_array

        if what == 'context':
            # "value" is the normalized intruder state
            # "context" is the unnormalized intruder state
            # "state" is the unnormalized ownship state

            rt_own = int(state[3])  # route identifier of ownship
            dgoal = self.dist_goal[id_] / self.max_d  # intruder's distance to the goal
            spd = self.normalize_that(context[2], 'spd')  # intruder's airspeed
            rt = self.normalize_that(context[3], 'rt')  # route identifier of intruder
            acc = context[4] + 0.5  # acceleration of intruder
            rt_int = int(context[3])  # route identifier of intruder

            if rt_own == rt_int:
                # if the ownship and the intruder are on the same route,
                # we don't consider the intersection
                dist_away = abs(value[0] - dgoal)
                dist_own_intersection = 0
                dist_int_intersection = 0

            else:
                # if the ownship and the intruder are on different routes,
                # we should consider the intersection
                dist_own_intersection = abs(
                    self.intersection_distances[rt_own][rt_int] / self.max_d - value[0])
                dist_int_intersection = abs(
                    self.intersection_distances[rt_int][rt_own] / self.max_d - dgoal)
                d = geo.latlondist(state[0], state[1], context[0], context[1]) / geo.nm
                dist_away = d / self.max_d

            # [distance to the goal, airspeed, route identifier, acceleration,
            # distance from ownship to intruder, distance from ownship to intersection,
            # distance from intruder to intersection], see the definition of h_t^o(i) in
            # the "attention network version of separation assurance" paper(Page 11)
            # for reference
            context_arr = np.array([dgoal, spd, rt, acc, dist_away,
                                    dist_own_intersection, dist_int_intersection])

            return context_arr.reshape(1, 1, 7)

    def _build_PPO(self):

        I = tf.keras.layers.Input(shape=(self.state_size,), name='states')

        context = tf.keras.layers.Input(shape=(self.num_intruders, 7), name='context')
        empty = tf.keras.layers.Input(shape=(HIDDEN_SIZE,), name='empty')

        advantage = tf.keras.layers.Input(shape=(1,), name='A')
        old_prediction = tf.keras.layers.Input(shape=(self.action_size,), name='old_pred')

        flatten_context = tf.keras.layers.Flatten()(context)
        # encoding other_state into 32 values
        H1_int = tf.keras.layers.Dense(HIDDEN_SIZE, activation='relu')(flatten_context)
        # now combine them
        combined = tf.keras.layers.concatenate([I, H1_int], axis=1)

        H2 = tf.keras.layers.Dense(256, activation='relu')(combined)
        H3 = tf.keras.layers.Dense(256, activation='relu')(H2)

        output = tf.keras.layers.Dense(self.action_size + 1, activation=None)(H3)

        # Split the output layer into policy and value
        policy = tf.keras.layers.Lambda(lambda x: x[:, :self.action_size], output_shape=(self.action_size,))(output)
        value = tf.keras.layers.Lambda(lambda x: x[:, self.action_size:], output_shape=(self.value_size,))(output)

        # now I need to apply activation
        policy_out = tf.keras.layers.Activation('softmax', name='policy_out')(policy)
        value_out = tf.keras.layers.Activation('linear', name='value_out')(value)

        # Using Adam optimizer, RMSProp's successor.
        opt = tf.keras.optimizers.Adam(lr=self.lr)

        model = tf.keras.models.Model(inputs=[I, context, empty, advantage, old_prediction],
                                      outputs=[policy_out, value_out])

        self.predictor = tf.keras.models.Model(inputs=[I, context, empty], outputs=[policy_out, value_out])

        # The model is trained on 2 different loss functions
        model.compile(optimizer=opt, loss={'policy_out': proximal_policy_optimization_loss(
            advantage=advantage,
            old_prediction=old_prediction), 'value_out': 'mse'})

        print(model.summary())

        return model

    def _build_PPO_saliency(self):

        I = tf.keras.layers.Input(shape=(self.state_size,), name='states')

        context = tf.keras.layers.Input(shape=(self.num_intruders, 7), name='context')
        empty = tf.keras.layers.Input(shape=(HIDDEN_SIZE,), name='empty')

        advantage = tf.keras.layers.Input(shape=(1,), name='A')
        old_prediction = tf.keras.layers.Input(shape=(self.action_size,), name='old_pred')

        flatten_context = tf.keras.layers.Flatten()(context)
        # encoding other_state into 32 values
        H1_int = tf.keras.layers.Dense(HIDDEN_SIZE, activation='relu')(flatten_context)
        # now combine them
        combined = tf.keras.layers.concatenate([I, H1_int], axis=1)

        H2 = tf.keras.layers.Dense(256, activation='relu')(combined)
        H3 = tf.keras.layers.Dense(256, activation='relu')(H2)

        output = tf.keras.layers.Dense(self.action_size + 1, activation=None)(H3)

        # Split the output layer into policy and value
        policy = tf.keras.layers.Lambda(lambda x: x[:, :self.action_size], output_shape=(self.action_size,))(output)
        value = tf.keras.layers.Lambda(lambda x: x[:, self.action_size:], output_shape=(self.value_size,))(output)

        # now I need to apply activation
        policy_out = tf.keras.layers.Activation('linear', name='policy_out')(policy)
        value_out = tf.keras.layers.Activation('linear', name='value_out')(value)

        # Using Adam optimizer, RMSProp's successor.
        opt = tf.keras.optimizers.Adam(lr=self.lr)

        model = tf.keras.models.Model(inputs=[I, context, empty, advantage, old_prediction],
                                      outputs=[policy_out, value_out])

        self.predictor_saliency = tf.keras.models.Model(inputs=[I, context, empty], outputs=[policy_out, value_out])

        # The model is trained on 2 different loss functions
        model.compile(optimizer=opt, loss={'policy_out': proximal_policy_optimization_loss(
            advantage=advantage,
            old_prediction=old_prediction), 'value_out': 'mse'})

        print(model.summary())

        return model

    def store(self, state, action, next_state, traf, id_, route_keeper, term=0):
        reward = 0
        done = False

        if term == 0:
            lat = traf.lat[traf.id2idx(id_)]
            lon = traf.lon[traf.id2idx(id_)]

            dist = self.dist_close[id_]

            if dist < 10 and dist > 3:
                reward = -0.1 + 0.05 * (dist / 10)

        if term == 1:
            reward = -1
            done = True

        if term == 2:
            reward = 0
            done = True

        state, context = state
        state = state.reshape((1, 5))
        context = context.reshape((1, -1, 7))

        if context.shape[1] > self.num_intruders:
            context = context[:, -self.num_intruders:, :]

        self.max_agents = max(self.max_agents, context.shape[1])

        if not id_ in self.experience.keys():
            self.experience[id_] = {}

        try:
            self.experience[id_]['state'] = np.append(self.experience[id_]['state'], state, axis=0)

            if self.max_agents > self.experience[id_]['context'].shape[1]:
                self.experience[id_]['context'] = np.append(
                    tf.keras.preprocessing.sequence.pad_sequences(self.experience[id_]['context'], self.max_agents,
                                                                  dtype='float32'), context, axis=0)
            else:
                self.experience[id_]['context'] = np.append(self.experience[id_]['context'],
                                                            tf.keras.preprocessing.sequence.pad_sequences(context,
                                                                                                          self.max_agents,
                                                                                                          dtype='float32'),
                                                            axis=0)

            self.experience[id_]['action'] = np.append(self.experience[id_]['action'], action)
            self.experience[id_]['reward'] = np.append(self.experience[id_]['reward'], reward)
            self.experience[id_]['done'] = np.append(self.experience[id_]['done'], done)




        except:
            self.experience[id_]['state'] = state
            if self.max_agents > context.shape[1]:
                self.experience[id_]['context'] = tf.keras.preprocessing.sequence.pad_sequences(context,
                                                                                                self.max_agents,
                                                                                                dtype='float32')
            else:
                self.experience[id_]['context'] = context

            self.experience[id_]['action'] = [action]
            self.experience[id_]['reward'] = [reward]
            self.experience[id_]['done'] = [done]

    def train(self):

        """Grab samples from batch to train the network"""

        total_state = []
        total_reward = []
        total_A = []
        total_advantage = []
        total_context = []
        total_policy = []

        total_length = 0

        for transitions in self.experience.values():
            episode_length = transitions['state'].shape[0]
            total_length += episode_length

            state = transitions['state']  # .reshape((episode_length,self.state_size))
            context = transitions['context']
            reward = transitions['reward']
            done = transitions['done']
            action = transitions['action']

            discounted_r, cumul_r = np.zeros_like(reward), 0
            discounted_rewards = discount(reward, discounted_r, cumul_r)
            policy, values = self.predictor.predict(
                {'states': state, 'context': context, 'empty': np.zeros((len(state), HIDDEN_SIZE))}, batch_size=256)
            advantages = np.zeros((episode_length, self.action_size))
            index = np.arange(episode_length)
            advantages[index, action] = 1
            A = discounted_rewards - values[:, 0]

            if len(total_state) == 0:

                total_state = state
                if context.shape[1] == self.max_agents:
                    total_context = context
                else:
                    total_context = tf.keras.preprocessing.sequence.pad_sequences(context, self.max_agents,
                                                                                  dtype='float32')
                total_reward = discounted_rewards
                total_A = A
                total_advantage = advantages
                total_policy = policy

            else:
                total_state = np.append(total_state, state, axis=0)
                if context.shape[1] == self.max_agents:
                    total_context = np.append(total_context, context, axis=0)
                else:
                    total_context = np.append(total_context,
                                              tf.keras.preprocessing.sequence.pad_sequences(context, self.max_agents,
                                                                                            dtype='float32'), axis=0)
                total_reward = np.append(total_reward, discounted_rewards, axis=0)
                total_A = np.append(total_A, A, axis=0)
                total_advantage = np.append(total_advantage, advantages, axis=0)
                total_policy = np.append(total_policy, policy, axis=0)

        total_A = (total_A - total_A.mean()) / (total_A.std() + 1e-8)
        self.model.fit({'states': total_state, 'context': total_context, 'empty': np.zeros((total_length, HIDDEN_SIZE)),
                        'A': total_A, 'old_pred': total_policy},
                       {'policy_out': total_advantage, 'value_out': total_reward}, shuffle=True,
                       batch_size=total_state.shape[0], epochs=8, verbose=0)

        self.max_agents = 0
        self.experience = {}

    def load(self, name):
        print('Loading weights...')
        self.model.load_weights(name)
        print('Successfully loaded model weights from {}'.format(name))

    def save(self, best=False, case_study='A'):

        if best:

            self.model.save_weights('best_model_{}.h5'.format(case_study))


        else:

            self.model.save_weights('model_{}.h5'.format(case_study))

    # action implementation for the agent
    def act(self, state, context):

        context = context.reshape((state.shape[0], -1, 7))

        if context.shape[1] > self.num_intruders:
            context = context[:, -self.num_intruders:, :]
        if context.shape[1] < self.num_intruders:
            context = tf.keras.preprocessing.sequence.pad_sequences(context, self.num_intruders, dtype='float32')

        policy, value = self.predictor.predict(
            {'states': state, 'context': context, 'empty': np.zeros((state.shape[0], HIDDEN_SIZE))},
            batch_size=state.shape[0])

        return policy

    # action implementation for the agent (for computing saliency)
    def act_saliency(self, state, context):

        context = context.reshape((state.shape[0], -1, 7))

        if context.shape[1] > self.num_intruders:
            context = context[:, -self.num_intruders:, :]
        if context.shape[1] < self.num_intruders:
            context = tf.keras.preprocessing.sequence.pad_sequences(
                context, self.num_intruders, dtype='float32')

        start = time.time()
        policy, value = self.predictor_saliency.predict(
            {'states': state, 'context': context, 'empty': np.zeros(
                (state.shape[0], HIDDEN_SIZE))}, batch_size=state.shape[0])
        end = time.time()
        runtime = end - start

        return policy, runtime

    def update(self, traf, index, route_keeper):
        """calulate reward and determine if terminal or not"""
        T = 0
        type_ = 0
        dist = getClosestAC_Distance(self, [traf.lat[index], traf.lon[index], traf.id[index]], traf, route_keeper)
        if dist < 3:
            T = True
            type_ = 1

        self.dist_close[traf.id[index]] = dist

        d_goal = dist_goal([traf.lat[index], traf.lon[index]], traf, index)

        if d_goal < 5 and T == 0:
            T = True
            type_ = 2

        self.dist_goal[traf.id[index]] = d_goal

        return T, type_

    def compute_simple_gradient_saliency(self, state, context, original_state,
                                         intruder_idx, max_ac, case_study='A'):
        """reference paper: Deep inside convolutional networks: Visualising image
        classification models and saliency maps"""

        if not self.load_best_model_flag:
            self.predictor.load_weights('best_model_{}_5.h5'.format(case_study))
            self.predictor_saliency.load_weights('best_model_{}_5.h5'.format(case_study))
            self.load_best_model_flag = True
        policy = self.act(state, context)
        select_ac = np.random.choice(state.shape[0], 1)[0]  # aircraft of interest
        # select_ac = 0  # aircraft of interest
        _policy = policy[select_ac]  # policy of aircraft of interest
        # action = np.random.choice(self.action_size, 1, p=_policy)[0]  # choose an action based on policy distribution
        action = np.argsort(_policy)[::-1]  # choose the action with max prob
        action_idx = action[0]

        layer_idx = utils.find_layer_idx(self.predictor_saliency, 'policy_out')
        # print(self.predictor_saliency.layers[layer_idx].activation)

        o_i = 1  # ownship(0) or intruder(1)

        layer_input = self.predictor_saliency.input
        # print(self.predictor_saliency.layers[layer_idx].output.shape)
        loss = self.predictor_saliency.layers[layer_idx].output[select_ac, action_idx]
        grad_tensor_own = K.gradients(loss, layer_input)[0]
        grad_tensor_int = K.gradients(loss, layer_input)[1]
        derivative_fn_own = K.function([layer_input], [grad_tensor_own])
        derivative_fn_int = K.function([layer_input], [grad_tensor_int])

        context = context.reshape((state.shape[0], -1, 7))
        if context.shape[1] > self.num_intruders:
            context = context[:, -self.num_intruders:, :]
        if context.shape[1] < self.num_intruders:
            context = tf.keras.preprocessing.sequence.pad_sequences(
                context, self.num_intruders, dtype='float32')

        grad_own = derivative_fn_own([state, context, np.zeros((state.shape[0],
                                                                HIDDEN_SIZE))])[0]
        grad_int = derivative_fn_int([state, context, np.zeros((state.shape[0],
                                                                HIDDEN_SIZE))])[0]
        # grad_eval_by_hand1 = derivative_fn([state, context])[0]
        # assert np.all(np.abs(grad_eval_by_hand1 - grad_eval_by_hand) < 0.00001)

        # saliency of ownships' or N intruders' states
        grad_own = grad_own[select_ac]
        grad_int = grad_int[select_ac]

        # draw aircraft position
        # if case_study == 'A':
        #     self.draw_aircraft_position_A(original_state, select_ac, intruder_idx,
        #                                   policy, max_ac)
        # elif case_study == 'B':
        #     self.draw_aircraft_position_B(original_state, select_ac, intruder_idx,
        #                                   policy, max_ac)
        # elif case_study == 'C':
        #     self.draw_aircraft_position_C(original_state, select_ac, intruder_idx,
        #                                   policy, max_ac)

        context = np.array(context[select_ac])
        inserted = np.zeros(context.shape[0])
        context = np.insert(context, context.shape[1], values=inserted, axis=1)
        context[:, context.shape[1] - 1] = -0.00001
        if len(intruder_idx[select_ac]) > 0:
            for i in range(len(intruder_idx[select_ac])):
                offset = len(intruder_idx[select_ac]) - 1 - i
                context[context.shape[0] - 1 - offset][context.shape[1] - 1] = \
                    intruder_idx[select_ac][i]

            grad_own = np.abs(grad_own)
            arr_min, arr_max = np.min(grad_own), np.max(grad_own)
            grad_own = (grad_own - arr_min) / (arr_max - arr_min + K.epsilon())

            grad_int = grad_int[(grad_int.shape[0] - len(intruder_idx[select_ac])
                                 ):(grad_int.shape[0]), :]
            grad_int = np.abs(grad_int)
            arr_min, arr_max = np.min(grad_int), np.max(grad_int)
            grad_int = (grad_int - arr_min) / (arr_max - arr_min + K.epsilon())

        return policy

    def draw_aircraft_position_A(self, original_state, ownship_idx, intruder_idx, policy,
                                 max_ac):

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # img = plt.imread('aircraft.png')
        # img = OffsetImage(img, zoom=0.5)
        plt.clf()
        ax = plt.gca()
        ax.set_aspect(1)

        route1 = np.load('route0_A.npy')
        route2 = np.load('route1_A.npy')

        route1_s_x, route1_e_x = route1[0][0], route1[0][len(route1[0]) - 1]
        route1_s_y, route1_e_y = route1[1][0], route1[1][len(route1[1]) - 1]
        route2_s_x, route2_e_x = route2[0][0], route2[0][len(route2[0]) - 1]
        route2_s_y, route2_e_y = route2[1][0], route2[1][len(route2[1]) - 1]

        route1_x = route1[0]
        route1_y = route1[1]
        # ax.plot(route1_x, route1_y)
        # ax.plot(route1_e_x, route1_e_y, marker=(3, 0, 186), markersize=12,
        #          markerfacecolor='b', markeredgecolor='b')
        # ax.text(route1_e_x, route1_e_y - 0.04, 'R0', fontsize='large',
        #          fontweight='bold')
        plt.plot(route1_x, route1_y)
        plt.plot(route1_e_x, route1_e_y, marker=(3, 0, 40.4977), markersize=12,
                 markerfacecolor='b', markeredgecolor='b')
        plt.text(route1_e_x, route1_e_y - 0.04, 'R0', fontsize='large',
                 fontweight='bold')

        route2_x = route2[0]
        route2_y = route2[1]
        # ax.plot(route2_x, route2_y)
        # ax.plot(route2_e_x, route2_e_y, marker=(3, 0, 232), markersize=12,
        #          markerfacecolor='b', markeredgecolor='b')
        # ax.text(route2_e_x, route2_e_y + 0.02, 'R1', fontsize='large',
        #          fontweight='bold')
        plt.plot(route2_x, route2_y)
        plt.plot(route2_e_x, route2_e_y, marker=(3, 0, 22.7517), markersize=12,
                 markerfacecolor='b', markeredgecolor='b')
        plt.text(route2_e_x, route2_e_y + 0.02, 'R1', fontsize='large',
                 fontweight='bold')

        lat, lon = original_state[:, 0], original_state[:, 1]
        plt.plot(lon[ownship_idx], lat[ownship_idx], 'g*', markersize=12, label='ownship')
        # img = AnnotationBbox(img, (lon[ownship_idx], lat[ownship_idx]), xycoords='data',
        #                      frameon=False)
        # ax.add_artist(img)
        if len(intruder_idx[ownship_idx]) > 0:
            for i in range(len(intruder_idx[ownship_idx])):
                plt.plot(lon[intruder_idx[ownship_idx][i]],
                         lat[intruder_idx[ownship_idx][i]], 'r*', markersize=12,
                         label='intruder')
                plt.text(lon[intruder_idx[ownship_idx][i]] - 0.035,
                         lat[intruder_idx[ownship_idx][i]] - 0.01,
                         '{}'.format(i + 1), color='w', fontsize=8.5)
        for i in range(len(lat)):
            if i != ownship_idx and i not in intruder_idx[ownship_idx]:
                plt.plot(lon[i], lat[i], 'b*', markersize=12, label='other')

        # for i in range(len(lat)):
        #     plt.text(lat[i], lon[i] + 0.1, 'KL{}'.format(i))
        _policy = []
        for i in range(len(policy)):
            _policy_temp = np.argsort(policy[i])[::-1]
            _policy.append(_policy_temp[0])
        for i in range(len(_policy)):
            if _policy[i] == 0:
                plt.text(lon[i] - 0.2, lat[i] + 0.02, '{}(-)'.format(i))
            elif _policy[i] == 1:
                plt.text(lon[i] - 0.2, lat[i] + 0.02, '{}(0)'.format(i))
            elif _policy[i] == 2:
                plt.text(lon[i] - 0.2, lat[i] + 0.02, '{}(+)'.format(i))

        plt.title('N_Closest, Case study A, {} aircraft'.format(max_ac))
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        # plt.show()

    def draw_aircraft_position_B(self, original_state, ownship_idx, intruder_idx, policy,
                                 max_ac):

        plt.clf()
        ax = plt.gca()
        ax.set_aspect(1)

        route1_s_x, route1_e_x = self.positions[0][1], self.positions[0][3]
        route1_s_y, route1_e_y = self.positions[0][0], self.positions[0][2]
        route2_s_x, route2_e_x = self.positions[1][1], self.positions[1][3]
        route2_s_y, route2_e_y = self.positions[1][0], self.positions[1][2]
        route3_s_x, route3_e_x = self.positions[2][1], self.positions[2][3]
        route3_s_y, route3_e_y = self.positions[2][0], self.positions[2][2]

        route1_x = np.arange(route1_e_x, route1_s_x, 0.001)
        route1_y = []
        for item in route1_x:
            route1_y.append(route1_s_y)
        plt.plot(route1_x, route1_y)
        plt.plot(route1_e_x, route1_e_y, marker=(3, 0, 90), markersize=12,
                 markerfacecolor='b', markeredgecolor='b')
        plt.text(route1_e_x - 0.02, route1_e_y - 0.15, 'R0', fontsize='large',
                 fontweight='bold')

        route2_x = np.arange(route2_s_x, route2_e_x, 0.001)
        route2_y = []
        for item in route2_x:
            route2_y.append(route2_s_y)
        plt.plot(route2_x, route2_y)
        plt.plot(route2_e_x, route2_e_y, marker=(3, 0, 270), markersize=12,
                 markerfacecolor='b', markeredgecolor='b')
        plt.text(route2_e_x - 0.02, route2_e_y - 0.15, 'R1', fontsize='large',
                 fontweight='bold')

        route3_x = []
        route3_y = np.arange(route3_e_y, route3_s_y, 0.001)
        for item in route3_y:
            route3_x.append(route3_s_x)
        plt.plot(route3_x, route3_y)
        plt.plot(route3_e_x, route3_e_y, marker=(3, 0, 180), markersize=12,
                 markerfacecolor='b', markeredgecolor='b')
        plt.text(route3_e_x + 0.06, route3_e_y, 'R2', fontsize='large',
                 fontweight='bold')

        lat, lon = original_state[:, 0], original_state[:, 1]
        plt.plot(lon[ownship_idx], lat[ownship_idx], 'g*', markersize=12, label='ownship')
        if len(intruder_idx[ownship_idx]) > 0:
            for i in range(len(intruder_idx[ownship_idx])):
                plt.plot(lon[intruder_idx[ownship_idx][i]],
                         lat[intruder_idx[ownship_idx][i]], 'r*', markersize=12,
                         label='intruder')
                plt.text(lon[intruder_idx[ownship_idx][i]] - 0.018,
                         lat[intruder_idx[ownship_idx][i]] - 0.02,
                         '{}'.format(i + 1), color='w', fontsize=8.5)
        for i in range(len(lat)):
            if i != ownship_idx and i not in intruder_idx[ownship_idx]:
                plt.plot(lon[i], lat[i], 'b*', markersize=12, label='other')

        # for i in range(len(lat)):
        #     plt.text(lat[i], lon[i] + 0.1, 'KL{}'.format(i))
        _policy = []
        for i in range(len(policy)):
            _policy_temp = np.argsort(policy[i])[::-1]
            _policy.append(_policy_temp[0])
        for i in range(len(_policy)):
            if _policy[i] == 0:
                plt.text(lon[i] - 0.1, lat[i] + 0.06, 'KL{}(-)'.format(i))
            elif _policy[i] == 1:
                plt.text(lon[i] - 0.1, lat[i] + 0.06, 'KL{}(0)'.format(i))
            elif _policy[i] == 2:
                plt.text(lon[i] - 0.1, lat[i] + 0.06, 'KL{}(+)'.format(i))

        plt.title('N_Closest, Case study B, {} aircraft'.format(max_ac))
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        # plt.show()

    def draw_aircraft_position_C(self, original_state, ownship_idx, intruder_idx, policy,
                                 max_ac):

        plt.clf()
        ax = plt.gca()
        ax.set_aspect(1)

        route1 = np.load('route0_C.npy')
        route2 = np.load('route1_C.npy')
        route3 = np.load('route2_C.npy')

        route1_s_x, route1_e_x = route1[0][0], route1[0][len(route1[0]) - 1]
        route1_s_y, route1_e_y = route1[1][0], route1[1][len(route1[1]) - 1]
        route2_s_x, route2_e_x = route2[0][0], route2[0][len(route2[0]) - 1]
        route2_s_y, route2_e_y = route2[1][0], route2[1][len(route2[1]) - 1]
        route3_s_x, route3_e_x = route3[0][0], route3[0][len(route3[0]) - 1]
        route3_s_y, route3_e_y = route3[1][0], route3[1][len(route3[1]) - 1]

        route1_x = route1[0]
        route1_y = route1[1]
        plt.plot(route1_x, route1_y)
        plt.plot(route1_e_x, route1_e_y, marker=(3, 0, 180), markersize=12,
                 markerfacecolor='b', markeredgecolor='b')
        plt.text(route1_e_x - 0.02, route1_e_y - 0.25, 'R0', fontsize='large',
                 fontweight='bold')

        route2_x = route2[0]
        route2_y = route2[1]
        plt.plot(route2_x, route2_y)
        plt.plot(route2_e_x, route2_e_y, marker=(3, 0, 143), markersize=12,
                 markerfacecolor='b', markeredgecolor='b')
        plt.text(route2_e_x - 0.02, route2_e_y - 0.23, 'R1', fontsize='large',
                 fontweight='bold')

        route3_x = route3[0]
        route3_y = route3[1]
        plt.plot(route3_x, route3_y)
        plt.plot(route3_e_x, route3_e_y, marker=(3, 0, 140), markersize=12,
                 markerfacecolor='b', markeredgecolor='b')
        plt.text(route3_e_x + 0.08, route3_e_y, 'R2', fontsize='large',
                 fontweight='bold')

        lat, lon = original_state[:, 0], original_state[:, 1]
        plt.plot(lon[ownship_idx], lat[ownship_idx], 'g*', markersize=12, label='ownship')
        if len(intruder_idx[ownship_idx]) > 0:
            for i in range(len(intruder_idx[ownship_idx])):
                plt.plot(lon[intruder_idx[ownship_idx][i]],
                         lat[intruder_idx[ownship_idx][i]], 'r*', markersize=12,
                         label='intruder')
                plt.text(lon[intruder_idx[ownship_idx][i]] - 0.035,
                         lat[intruder_idx[ownship_idx][i]] - 0.035,
                         '{}'.format(i + 1), color='w', fontsize=8.5)
        for i in range(len(lat)):
            if i != ownship_idx and i not in intruder_idx[ownship_idx]:
                plt.plot(lon[i], lat[i], 'b*', markersize=12, label='other')

        # for i in range(len(lat)):
        #     plt.text(lat[i], lon[i] + 0.1, 'KL{}'.format(i))
        _policy = []
        for i in range(len(policy)):
            _policy_temp = np.argsort(policy[i])[::-1]
            _policy.append(_policy_temp[0])
        for i in range(len(_policy)):
            if _policy[i] == 0:
                plt.text(lon[i] - 0.1, lat[i] + 0.09, 'KL{}(-)'.format(i))
            elif _policy[i] == 1:
                plt.text(lon[i] - 0.1, lat[i] + 0.09, 'KL{}(0)'.format(i))
            elif _policy[i] == 2:
                plt.text(lon[i] - 0.1, lat[i] + 0.09, 'KL{}(+)'.format(i))

        plt.title('N_Closest, Case study C, {} aircraft'.format(max_ac))
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        # plt.show()

    def LEG_heatmap(self, leg, ownship_id, intruder_idx, sign, spd, num_valid):
        aircraft_id = []
        features = ["lat/lon", "tas"]

        aircraft_id.append("{} (ownship)".format(ownship_id))
        for int_id in intruder_idx:
            aircraft_id.append("{} ({})".format(int_id, intruder_idx.index(int_id)+1))

        fig, ax = plt.subplots(1, 3)  # self.action_size = 3
        ax[0].imshow(leg[0], cmap='jet')
        ax[1].imshow(leg[1], cmap='jet')
        im = ax[2].imshow(leg[2], cmap='jet')
        plt.colorbar(im)

        for i in range(3):  # self.action_size = 3
            ax[i].set_xticks(np.arange(len(features)))
            ax[i].set_yticks(np.arange(len(aircraft_id)))
            ax[i].set_xticklabels(features)
            ax[i].set_yticklabels(aircraft_id)
            plt.setp(ax[i].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(3):  # self.action_size = 3
            for j in range(len(aircraft_id)):
                for k in range(len(features)):
                    ax[i].text(k, j, "{}\n({}, {})".format(
                        round(leg[i, j, k], 5), sign[i, j, k], round(spd[j], 2)), ha="center", va="center",
                                   color='w') if k == 1 else ax[i].text(k, j, "{}\n({})".format(
                        round(leg[i, j, k], 5), sign[i, j, k]), ha="center", va="center", color='w')

        plt.suptitle('LEG (min-max normalized), {} samples'.format(num_valid))
        ax[0].set_title('dec')
        ax[1].set_title('hold')
        ax[2].set_title('acc')
        # plt.show()
