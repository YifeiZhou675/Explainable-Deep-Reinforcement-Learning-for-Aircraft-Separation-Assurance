""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """
import numpy as np
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, settings, navdb, traf, sim, scr, tools
from bluesky import navdb
from bluesky.tools.aero import ft
from bluesky.tools import geo, areafilter
from Multi_Agent.PPO import PPO_Agent, dist_goal
import geopy.distance
import tensorflow as tf
import random
import pandas as pd
from operator import itemgetter
from shapely.geometry import LineString
import numba as nb
import time
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K


## For running on GPU
# from keras.backend.tensorflow_backend import set_session
# from shapely.geometry import LineString
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
#
# sess = tf.Session(config=config)
# set_session(sess)


### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    global num_ac
    global counter
    global ac
    global max_ac
    global positions
    global agent
    global best_reward
    global num_success
    global success
    global collisions
    global num_collisions
    global ac_counter
    global route_queue
    global n_states
    global route_keeper
    global previous_action
    global last_observation
    global observation
    global num_success_train
    global num_collisions_train
    global choices
    global positions
    global start
    global route0
    global route1
    global N
    global regular
    global tas_lvl

    num_success_train = []
    num_collisions_train = []

    num_success = []
    num_collisions = []
    previous_action = {}
    last_observation = {}
    observation = {}
    collisions = 0
    success = 0

    num_ac = 0
    max_ac = 30
    best_reward = -10000000
    ac_counter = 0
    n_states = 5
    route_keeper = np.zeros(max_ac, dtype=int)
    num_intruders = 5
    regular = 0.15  # regular noise level of longitude
    tas_lvl = 0.1  # noise level of tas

    positions = np.load('./routes/case_study_a_route.npy')
    choices = [20, 25, 30]  # 4 minutes, 5 minutes, 6 minutes
    route_queue = random.choices(choices, k=positions.shape[0])

    agent = PPO_Agent(n_states, 3, positions.shape[0], 100000, positions, num_intruders)
    agent.predictor.load_weights('best_model_A_5.h5')
    agent.predictor_saliency.load_weights('best_model_A_5.h5')
    counter = 0
    start = time.time()

    route0 = [[], []]  # (lon, lat)
    route1 = [[], []]  # (lon, lat)
    R0 = np.load('route0_A.npy')
    R1 = np.load('route1_A.npy')
    N = np.zeros((2, 3))  # Normal

    R0_s_lon, R0_e_lon = R0[0][0], R0[0][-1]
    R0_s_lat, R0_e_lat = R0[1][0], R0[1][-1]
    R1_s_lon, R1_e_lon = R1[0][0], R1[0][-1]
    R1_s_lat, R1_e_lat = R1[1][0], R1[1][-1]

    R0_se = np.array([[R0_s_lon, R0_e_lon], [R0_s_lat, R0_e_lat]])
    R0_se[0] += 360
    R0_se[1] = 90 - R0_se[1]
    R0_se = np.radians(R0_se)

    x_s0 = np.sin(R0_se[1][0]) * np.cos(R0_se[0][0])
    y_s0 = np.sin(R0_se[1][0]) * np.sin(R0_se[0][0])
    z_s0 = np.cos(R0_se[1][0])
    x_e0 = np.sin(R0_se[1][1]) * np.cos(R0_se[0][1])
    y_e0 = np.sin(R0_se[1][1]) * np.sin(R0_se[0][1])
    z_e0 = np.cos(R0_se[1][1])
    N[0] = np.cross([x_s0, y_s0, z_s0], [x_e0, y_e0, z_e0])

    R1_se = np.array([[R1_s_lon, R1_e_lon], [R1_s_lat, R1_e_lat]])
    R1_se[0] += 360
    R1_se[1] = 90 - R1_se[1]
    R1_se = np.radians(R1_se)

    x_s1 = np.sin(R1_se[1][0]) * np.cos(R1_se[0][0])
    y_s1 = np.sin(R1_se[1][0]) * np.sin(R1_se[0][0])
    z_s1 = np.cos(R1_se[1][0])
    x_e1 = np.sin(R1_se[1][1]) * np.cos(R1_se[0][1])
    y_e1 = np.sin(R1_se[1][1]) * np.sin(R1_se[0][1])
    z_e1 = np.cos(R1_se[1][1])
    N[1] = np.cross([x_s1, y_s1, z_s1], [x_e1, y_e1, z_e1])

    # Addtional initilisation code
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name': 'CASE_STUDY_A',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type': 'sim',

        # Update interval in seconds. By default, your plugin's update function(s)
        # are called every timestep of the simulation. If your plugin needs less
        # frequent updates provide an update interval.
        'update_interval': 12.0,

        # The update function is called after traffic is updated. Use this if you
        # want to do things as a result of what happens in traffic. If you need to
        # something before traffic is updated please use preupdate.

        'update': update}

    # If your plugin has a state, you will probably need a reset function to
    # clear the state in between simulations.
    # 'reset':         reset
    # }

    stackfunctions = {
    }

    return config, stackfunctions


### Periodic update functions that are called by the simulation. You can replace
### this by anything, so long as you communicate this in init_plugin


def update():
    """given a current state in the simulation, allow the agent to select an action.
     "" Then send the action to the bluesky command line ""
    """
    global num_ac
    global counter
    global ac
    global max_ac
    global positions
    global agent
    global success
    global collisions
    global ac_counter
    global route_queue
    global n_states
    global route_keeper
    global previous_action
    global choices
    global start
    global route0
    global route1

    store_terminal = {}

    if ac_counter < max_ac:  ## maybe spawn a/c based on time, not based on this update interval

        if ac_counter == 0:
            for i in range(len(positions)):
                lat, lon, glat, glon, h = positions[i]
                stack.stack('CRE KL{}, A320, {}, {}, {}, 25000, 251'.format(ac_counter, lat, lon, h))
                stack.stack('ADDWPT KL{} {}, {}'.format(ac_counter, glat, glon))
                route_keeper[ac_counter] = i
                num_ac += 1
                ac_counter += 1

        else:
            for k in range(len(route_queue)):
                if counter == route_queue[k]:
                    lat, lon, glat, glon, h = positions[k]
                    stack.stack('CRE KL{}, A320, {}, {}, {}, 25000, 251'.format(ac_counter, lat, lon, h))
                    stack.stack('ADDWPT KL{} {}, {}'.format(ac_counter, glat, glon))
                    route_keeper[ac_counter] = k

                    num_ac += 1
                    ac_counter += 1

                    route_queue[k] = counter + random.choices(choices, k=1)[0]

                    if ac_counter == max_ac:
                        break

    store_terminal = np.zeros(len(traf.id), dtype=int)
    for i in range(len(traf.id)):
        T, type_ = agent.update(traf, i, route_keeper)
        id_ = traf.id[i]

        if T:
            stack.stack('DEL {}'.format(id_))
            num_ac -= 1
            if type_ == 1:
                collisions += 1
            if type_ == 2:
                success += 1

            store_terminal[i] = 1

            agent.store(last_observation[id_], previous_action[id_],
                        [np.zeros(last_observation[id_][0].shape), np.zeros(last_observation[id_][1].shape)], traf, id_,
                        route_keeper, type_)

            del last_observation[id_]

    if ac_counter == max_ac and num_ac == 0:
        reset()
        return

    if num_ac == 0 and ac_counter != max_ac:
        return

    if not len(traf.id) == 0:
        ids = []
        new_actions = {}
        n_ac = len(traf.id)
        state = np.zeros((n_ac, 5))

        id_sub = np.array(traf.id)[store_terminal != 1]
        ind = np.array([int(x[2:]) for x in traf.id])
        route = route_keeper[ind]

        state[:, 0] = traf.lat  # latitude
        state[:, 1] = traf.lon  # longitude
        state[:, 2] = traf.tas  # speed provided by the ASAS (eas) [m/s]
        state[:, 3] = route  # route identifier
        state[:, 4] = traf.ax  # [m/s2] absolute value of longitudinal acceleration

        # if 'KL0' in id_sub:
        #     route0[0].append(state[0][1])  # lon
        #     route0[1].append(state[0][0])  # lat
        # if 'KL1' in id_sub:
        #     route1[0].append(state[1][1])  # lon
        #     route1[1].append(state[1][0])  # lat
        # if 'KL0' not in id_sub and 'KL1' not in id_sub:
        #     np.save('route0_A.npy', route0)
        #     np.save('route1_A.npy', route1)
        #     print('save!!!')

        norm_state, norm_context, intruder_idx = getClosestAC(state, traf, route_keeper,
                                                              previous_action, n_states,
                                                              store_terminal, agent,
                                                              last_observation, observation)

        # if ac_counter == 30 and num_ac == state.shape[0]:
        #     print('ac_counter = {}, num_ac = {}'.format(ac_counter, num_ac))
        #     LEGp_v1(state, 1000, intruder_idx, traf, agent, norm_state, norm_context)

        # if norm_state.shape[0] == 0:
        #     import ipdb; ipdb.set_trace()

        policy = agent.act(norm_state, norm_context)
        # policy = agent.compute_simple_gradient_saliency(norm_state, norm_context, state,
        #                                                 intruder_idx, max_ac)

        for j in range(len(id_sub)):
            id_ = id_sub[j]

            # This is for updating s, sp, ...
            if not id_ in last_observation.keys():
                last_observation[id_] = [norm_state[j], norm_context[j]]

            if not id_ in observation.keys() and id_ in previous_action.keys():
                observation[id_] = [norm_state[j], norm_context[j]]

                agent.store(last_observation[id_], previous_action[id_], observation[id_], traf, id_, route_keeper)
                last_observation[id_] = observation[id_]

                del observation[id_]

            action = np.random.choice(agent.action_size, 1, p=policy[j].flatten())[0]
            speed = agent.speeds[action]  # why set speed manually?
            index = traf.id2idx(id_)

            if action == 1:  # hold
                speed = int(np.round((traf.cas[index] / tools.geo.nm) * 3600))  # ?

            stack.stack('{} SPD {}'.format(id_, speed))
            new_actions[id_] = action

        previous_action = new_actions

    counter += 1


def reset():
    global best_reward
    global counter
    global num_ac
    global num_success
    global success
    global collisions
    global num_collisions
    global ac_counter
    global route_queue
    global n_states
    global route_keeper
    global previous_action
    global last_observation
    global observation
    global num_success_train
    global num_collisions_train
    global choices
    global positions
    global start

    if (agent.episode_count + 1) % 5 == 0:
        agent.train()

    end = time.time()

    print(end - start)
    goals_made = success

    num_success_train.append(success)
    num_collisions_train.append(collisions)

    success = 0
    collisions = 0

    counter = 0
    num_ac = 0
    ac_counter = 0

    route_queue = random.choices([20, 25, 30], k=positions.shape[0])

    previous_action = {}
    route_keeper = np.zeros(max_ac, dtype=int)
    last_observation = {}
    observation = {}

    t_success = np.array(num_success_train)
    t_coll = np.array(num_collisions_train)
    np.save('success_train_A.npy', t_success)
    np.save('collisions_train_A.npy', t_coll)

    if agent.episode_count > 150:
        df = pd.DataFrame(t_success)
        if float(df.rolling(150, 150).mean().max()) >= best_reward:
            agent.save(True, case_study='A')
            best_reward = float(df.rolling(150, 150).mean().max())

    agent.save(case_study='A')

    print("Episode: {} | Reward: {} | Best Reward: {}".format(agent.episode_count, goals_made, best_reward))

    agent.episode_count += 1

    if agent.episode_count == agent.numEpisodes:
        stack.stack('STOP')

    stack.stack('IC multi_agent.scn')

    start = time.time()


def getClosestAC(state, traf, route_keeper, new_action, n_states, store_terminal,
                 agent, last_observation, observation):
    n_ac = traf.lat.shape[0]
    norm_state = np.zeros((len(store_terminal[store_terminal != 1]), 5))
    intruder_idx = []  # save indices of each ownship's intruders

    size = traf.lat.shape[0]
    index = np.arange(size).reshape(-1, 1)

    d = geo.latlondist_matrix(
        np.repeat(state[:, 0], n_ac), np.repeat(state[:, 1], n_ac),
        np.tile(state[:, 0], n_ac), np.tile(state[:, 1], n_ac)).reshape(n_ac, n_ac)
    d_temp = np.array(d)
    argsort = np.array(np.argsort(d, axis=1))  # sort by row

    total_closest_states = []
    route_count = 0
    i = 0
    j = 0

    max_agents = 1

    count = 0
    for i in range(d.shape[0]):  # "i" is ownship's index
        intruder_idx.append([])
        r = int(state[i][3])  # route identifier of ownship
        lat, lon, glat, glon, h = agent.positions[r]
        if store_terminal[i] == 1:
            continue
        ownship_obj = LineString([[
            state[i][1], state[i][0], 31000],
            [glon, glat, 31000]])  # ownship's current position to goal position, (lon, lat)

        # norm_state[:, 0]: distance to the goal
        # norm_state[:, 1]: airspeed
        # norm_state[:, 2]: route identifier
        # norm_state[:, 3]: acceleration
        # norm_state[:, 4]: loss of separation
        norm_state[count, :] = agent.normalize_that(state[i], 'state', id_=traf.id[i])
        closest_states = []
        count += 1

        route_count = 0

        intruder_count = 0

        for j in range(len(argsort[i])):  # "j" is intruder's index

            index = int(argsort[i][j])  # intruder

            if i == index:  # means the diagonal elements in "d", which are all 0
                continue

            if store_terminal[index] == 1:  # aircraft "index" has been terminal
                continue

            route = int(state[index][3])  # route identifier of intruder "index"

            if route == r and route_count == 2:  # no more than 2 intruders on the same route as ownship
                continue

            if route == r:
                route_count += 1

            lat, lon, glat, glon, h = agent.positions[route]
            # intruder's current position to goal position
            int_obj = LineString(
                [[state[index, 1], state[index, 0], 31000], [glon, glat, 31000]])

            if not ownship_obj.intersects(int_obj):
                # means that collision won't happen
                continue

            if not route in agent.intersection_distances[r].keys() and route != r:
                continue

            if d[i, index] > 100:
                # far away from ownship, won't consider
                continue

            max_agents = max(max_agents, j)

            if len(closest_states) == 0:
                closest_states = np.array(
                    [traf.lat[index], traf.lon[index], traf.tas[index], route,
                     traf.ax[index]])
                closest_states = agent.normalize_that(
                    norm_state[count - 1], 'context', closest_states, state[i],
                    id_=traf.id[index])
            else:
                adding = np.array(
                    [traf.lat[index], traf.lon[index], traf.tas[index], route,
                     traf.ax[index]])
                adding = agent.normalize_that(
                    norm_state[count - 1], 'context', adding, state[i],
                    id_=traf.id[index])

                closest_states = np.append(closest_states, adding, axis=1)

            intruder_count += 1
            intruder_idx[i].append(index)

            if intruder_count == agent.num_intruders:
                break

        if len(closest_states) == 0:
            closest_states = np.array([0, 0, 0, 0, 0, 0, 0]).reshape(1, 1, 7)

        if len(total_closest_states) == 0:
            total_closest_states = closest_states
        else:
            total_closest_states = np.append(
                tf.keras.preprocessing.sequence.pad_sequences(
                    total_closest_states, agent.num_intruders, dtype='float32'),
                tf.keras.preprocessing.sequence.pad_sequences(
                    closest_states, agent.num_intruders, dtype='float32'), axis=0)

    if len(total_closest_states) == 0:
        total_closest_states = np.array([0, 0, 0, 0, 0, 0, 0]).reshape(
            1, agent.num_intruders, 7)

    return norm_state, total_closest_states, intruder_idx


def LEG_getPState(state, traf, agent, ac_idx, ACint_idx):
    norm_state = agent.normalize_that(state[ac_idx], 'state', id_=traf.id[ac_idx])
    closest_states = []

    for i in range(len(ACint_idx)):
        index = ACint_idx[i]
        route = int(state[index, 3])
        if len(closest_states) == 0:
            closest_states = np.array([traf.lat[index], traf.lon[index], traf.tas[index],
                                       route, traf.ax[index]])
            closest_states = agent.normalize_that(norm_state, 'context', closest_states,
                                                  state[ac_idx], id_=traf.id[index])
        else:
            adding = np.array([traf.lat[index], traf.lon[index], traf.tas[index], route,
                               traf.ax[index]])
            adding = agent.normalize_that(norm_state, 'context', adding, state[ac_idx],
                                          id_=traf.id[index])
            closest_states = np.append(closest_states, adding, axis=1)

    if len(closest_states) == 0:  # in case that there are no intruders
        closest_states = np.array([0, 0, 0, 0, 0, 0, 0]).reshape((1, 1, 7))

    return norm_state, closest_states


def LEGp_v1(state, num_sample, intruder_idx, traf, agent, onorm_state, onorm_context):
    """Only consider intruders change"""

    global max_ac
    global N
    global regular
    global tas_lvl

    r_id = state[:, 3].astype(int)  # route identifier
    # ori_policy, rt = agent.act_saliency(onorm_state, onorm_context)
    ori_policy = agent.act(onorm_state, onorm_context)
    LEG_list = []

    for ac_idx in range(state.shape[0]):
        # ac_idx = 7
        agent.draw_aircraft_position_A(state, ac_idx, intruder_idx, ori_policy, max_ac)

        ACint_idx = intruder_idx[ac_idx]  # intruders of current aircraft
        if len(ACint_idx) == 0:
            continue

        num_intruder = len(ACint_idx)
        lon_lvl = np.array([regular] * (num_intruder + 1))  # noise level of longitude
        min_d = 100.0  # minimum difference of longitude between ownship and intersections
        r = int(state[ac_idx][3])  # ownship's route identifier
        for inter in agent.intersections[r]:
            lon_diff = abs(state[ac_idx, 1] - agent.intersections[r][inter][1])
            if min_d > lon_diff:
                min_d = lon_diff
        if min_d < regular:
            lon_lvl[0] = min_d
        for i in range(agent.num_intruders):
            if onorm_context[ac_idx][i][6] != 0.0:
                route = int(onorm_context[ac_idx][i][2])  # intruder's route identifier
                idx = i - agent.num_intruders + num_intruder  # actual index of intruder
                lon_diff = abs(state[ACint_idx[idx], 1] - agent.intersections[route][r][1])
                if lon_diff < regular:
                    lon_lvl[idx + 1] = lon_diff
        var_lon = lon_lvl ** 2 / 3
        var_tas = tas_lvl ** 2 / 3
        sigma = np.identity((num_intruder + 1) * 2)
        sigma *= var_tas
        for i in range(num_intruder + 1):
            sigma[i, i] = var_lon[i]
        inv_sigma = np.linalg.inv(sigma)
        # the first row of 'noise' is for ownship, others are for intruders
        noise = np.zeros((num_intruder + 1, num_sample))
        noise[0, :] = np.random.uniform(-lon_lvl[0], lon_lvl[0], size=num_sample)
        for i in range(num_intruder):
            noise[i + 1, :] = np.random.uniform(-lon_lvl[i + 1], lon_lvl[i + 1], size=num_sample)

        p_own_lon = noise[0, :] + state[ac_idx][1]
        p_own_lon += 360
        p_own_lat = np.arctan(-N[r_id[ac_idx]][2] / (N[r_id[ac_idx]][0] * np.cos(
            np.radians(p_own_lon)) + N[r_id[ac_idx]][1] * np.sin(np.radians(p_own_lon))))
        p_own_lat = 90 - np.degrees(p_own_lat)
        p_own_lon -= 360

        p_intru_lon = np.zeros((num_intruder, num_sample))
        p_intru_lat = np.zeros((num_intruder, num_sample))
        for i in range(num_intruder):
            p_intru_lon[i, :] = noise[i + 1, :] + state[ACint_idx[i]][1]
            p_intru_lon[i, :] += 360
            p_intru_lat[i, :] = np.arctan(-N[r_id[ACint_idx[i]]][2] / (N[r_id[ACint_idx[i]]][0] * np.cos(
                np.radians(p_intru_lon[i, :])) + N[r_id[ACint_idx[i]]][1] * np.sin(np.radians(p_intru_lon[i, :]))))
            p_intru_lat[i, :] = 90 - np.degrees(p_intru_lat[i, :])
            p_intru_lon[i, :] -= 360

        p_state = np.array(state)  # perturbed state features (all features)
        # num_valid = 0  # number of valid perturbations
        z = np.zeros((num_intruder + 1, 2))  # see equation (4) in LEG paper
        target_y = np.zeros((agent.action_size, num_intruder + 1, 2))
        _ori_policy = ori_policy[ac_idx]
        for sample_idx in range(num_sample):
            p_state[ac_idx][0], p_state[ac_idx][1] = p_own_lat[sample_idx], p_own_lon[sample_idx]
            for i in range(num_intruder):
                p_state[ACint_idx[i]][0] = p_intru_lat[i][sample_idx]
                p_state[ACint_idx[i]][1] = p_intru_lon[i][sample_idx]

            # we don't need to consider 'd_goal < 5' here, cuz we're using the waypoint file
            # (loaded in the initialization function)
            # the end points are already actual exit points
            # But we still need to update d_goal for corresponding aircraft!!!!!
            traf.lat = p_state[:, 0]
            traf.lon = p_state[:, 1]
            d_goal = dist_goal([traf.lat[ac_idx], traf.lon[ac_idx]], traf, ac_idx)
            agent.dist_goal[traf.id[ac_idx]] = d_goal
            for i in range(num_intruder):
                d_goal = dist_goal([traf.lat[ACint_idx[i]], traf.lon[ACint_idx[i]]],
                                   traf, ACint_idx[i])
                agent.dist_goal[traf.id[ACint_idx[i]]] = d_goal

            pNorm_state, pNorm_context = LEG_getPState(p_state, traf, agent, ac_idx, ACint_idx)
            # num_valid += 1
            spd_noise = np.random.uniform(-tas_lvl, tas_lvl)
            pNorm_state[1] += spd_noise
            z[0][1] = spd_noise
            for i in range(num_intruder):
                spd_noise = np.random.uniform(-tas_lvl, tas_lvl)
                pNorm_context[0][i][1] += spd_noise
                z[i + 1][1] = spd_noise

            pNorm_state = np.array([pNorm_state])
            # p_policy, t = agent.act_saliency(pNorm_state, pNorm_context)
            p_policy = agent.act(pNorm_state, pNorm_context)
            y_diff = p_policy[0] - _ori_policy

            # longitude noise
            z[0][0] = noise[0][sample_idx]
            for i in range(num_intruder):
                z[i + 1][0] = noise[i + 1][sample_idx]

            for i in range(agent.action_size):
                target_y[i] += y_diff[i] * z

        LEG = np.zeros((agent.action_size, num_intruder + 1, 2))
        for i in range(agent.action_size):
            target_y[i] /= num_sample
            temp = target_y[i].flatten('F')
            LEG[i] = (inv_sigma @ temp).reshape(num_intruder + 1, 2, order='F')

        # normalized LEG
        sign = (LEG > 0).astype(str)  # sign of each element in LEG
        for i in range(agent.action_size):
            for j in range(num_intruder + 1):
                for k in range(2):
                    if sign[i, j, k] == 'True':
                        sign[i, j, k] = '+'
                    elif LEG[i, j, k] == 0:
                        sign[i, j, k] = '0'
                    else:
                        sign[i, j, k] = '-'
        for i in range(agent.action_size):
            LEG[i] = np.abs(LEG[i])
            LEG_min, LEG_max = np.min(LEG[i]), np.max(LEG[i])
            LEG[i] = (LEG[i] - LEG_min) / (LEG_max - LEG_min + K.epsilon())
        spd = np.zeros(num_intruder + 1)
        spd[0] = onorm_state[ac_idx][1]
        for i in range(num_intruder):
            spd[num_intruder - i] = onorm_context[ac_idx][agent.num_intruders - 1 - i][1]
        agent.LEG_heatmap(LEG, ac_idx, ACint_idx, sign, spd, num_sample)
        print('1')

    # restore the states to original states
    traf.lat = state[:, 0]
    traf.lon = state[:, 1]
    for i in range(state.shape[0]):
        agent.dist_goal[traf.id[i]] = onorm_state[i][0] * agent.max_d
    print('2')
