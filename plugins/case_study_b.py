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
    num_intruders = 5

    num_ac = 0
    max_ac = 30
    best_reward = -10000000
    ac_counter = 0
    n_states = 5
    route_keeper = np.zeros(max_ac, dtype=int)
    regular = [0.15, 0.15, 0.15]  # regular noise level of lat/lon (R0,R1,R2)
    tas_lvl = 0.15  # noise level of tas

    positions = np.load('./routes/case_study_b_route.npy')
    choices = [20, 25, 30]  # 4 minutes, 5 minutes, 6 minutes
    route_queue = random.choices(choices, k=positions.shape[0])

    agent = PPO_Agent(n_states, 3, positions.shape[0], 100000, positions, num_intruders)
    agent.predictor.load_weights('best_model_B_5.h5')
    agent.predictor_saliency.load_weights('best_model_B_5.h5')
    counter = 0
    start = time.time()

    # Addtional initilisation code
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name': 'CASE_STUDY_B',

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

    store_terminal = {}

    if ac_counter < max_ac:  ## maybe spawn a/c based on time, not based on this update interval

        if ac_counter == 0:
            for i in range(len(positions)):
                lat, lon, glat, glon, h = positions[i]  # 'h' means heading angle
                stack.stack('CRE KL{}, A320, {}, {}, {}, 25000, 251'.format(ac_counter, lat, lon, h))
                stack.stack('ADDWPT KL{} {}, {}'.format(ac_counter, glat, glon))
                route_keeper[ac_counter] = i
                num_ac += 1
                ac_counter += 1

        else:
            for k in range(len(route_queue)):
                if counter == route_queue[k]:
                    lat, lon, glat, glon, h = positions[k]  # 'h' means heading angle
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

        norm_state, norm_context, intruder_idx = getClosestAC(state, traf, route_keeper, previous_action, n_states,
                                                              store_terminal, agent, last_observation, observation)
        if ac_counter == 30 and num_ac == state.shape[0]:
            print('ac_counter = {}, num_ac = {}'.format(ac_counter, num_ac))
            LEGp_v1(state, 1000, intruder_idx, traf, agent, norm_state, norm_context)

        # if norm_state.shape[0] == 0:
        #     import ipdb; ipdb.set_trace()

        policy = agent.act(norm_state, norm_context)
        # policy = agent.compute_simple_gradient_saliency(norm_state, norm_context, state,
        #                                                 intruder_idx, max_ac,
        #                                                 case_study='B')

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
            speed = agent.speeds[action]
            index = traf.id2idx(id_)

            if action == 1:  # hold
                speed = int(np.round((traf.cas[index] / tools.geo.nm) * 3600))

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
    np.save('success_train_B.npy', t_success)
    np.save('collisions_train_B.npy', t_coll)

    if agent.episode_count > 150:
        df = pd.DataFrame(t_success)
        if float(df.rolling(150, 150).mean().max()) >= best_reward:
            agent.save(True, case_study='B')
            best_reward = float(df.rolling(150, 150).mean().max())

    agent.save(case_study='B')

    print("Episode: {} | Reward: {} | Best Reward: {}".format(agent.episode_count, goals_made, best_reward))

    agent.episode_count += 1

    if agent.episode_count == agent.numEpisodes:
        stack.stack('STOP')

    stack.stack('IC multi_agent.scn')

    start = time.time()


def getClosestAC(state, traf, route_keeper, new_action, n_states, store_terminal, agent, last_observation, observation):
    n_ac = traf.lat.shape[0]
    norm_state = np.zeros((len(store_terminal[store_terminal != 1]), 5))
    intruder_idx = []

    size = traf.lat.shape[0]
    index = np.arange(size).reshape(-1, 1)

    # dist_start = time.time()
    d = geo.latlondist_matrix(np.repeat(state[:, 0], n_ac), np.repeat(state[:, 1], n_ac), np.tile(state[:, 0], n_ac),
                              np.tile(state[:, 1], n_ac)).reshape(n_ac, n_ac)
    argsort = np.array(np.argsort(d, axis=1))
    # dist_end = time.time()
    # agent.dist_time += dist_end - dist_start

    total_closest_states = []
    route_count = 0
    i = 0
    j = 0

    max_agents = 1

    count = 0
    for i in range(d.shape[0]):
        intruder_idx.append([])
        r = int(state[i][3])
        lat, lon, glat, glon, h = agent.positions[r]
        if store_terminal[i] == 1:
            continue
        ownship_obj = LineString([[state[i][1], state[i][0], 31000], [glon, glat, 31000]])

        norm_state[count, :] = agent.normalize_that(state[i], 'state', id_=traf.id[i])
        closest_states = []
        count += 1

        route_count = 0

        intruder_count = 0

        for j in range(len(argsort[i])):

            index = int(argsort[i][j])

            if i == index:
                continue

            if store_terminal[index] == 1:
                continue

            route = int(state[index][3])

            if route == r and route_count == 2:
                continue

            if route == r:
                route_count += 1

            lat, lon, glat, glon, h = agent.positions[route]
            ##### time consuming!! 13~15s #####
            int_obj = LineString([[state[index, 1], state[index, 0], 31000], [glon, glat, 31000]])
            ###################################

            ##### time consuming!! 18~20s #####
            intersect_start = time.time()
            if not ownship_obj.intersects(int_obj):
                intersect_end = time.time()
                agent.intersect_time += intersect_end - intersect_start
                continue
            intersect_end = time.time()
            agent.intersect_time += intersect_end - intersect_start
            #################################

            if not route in agent.intersection_distances[r].keys() and route != r:
                continue

            if d[i, index] > 100:
                continue

            max_agents = max(max_agents, j)

            ##### time consuming!! 21~22s #####
            if len(closest_states) == 0:
                closest_states = np.array([traf.lat[index], traf.lon[index], traf.tas[index], route, traf.ax[index]])
                closest_states = agent.normalize_that(norm_state[count - 1], 'context', closest_states, state[i],
                                                      id_=traf.id[index])
            else:
                adding = np.array([traf.lat[index], traf.lon[index], traf.tas[index], route, traf.ax[index]])
                adding = agent.normalize_that(norm_state[count - 1], 'context', adding, state[i], id_=traf.id[index])

                closest_states = np.append(closest_states, adding, axis=1)
            ###################################

            intruder_count += 1
            intruder_idx[i].append(index)

            if intruder_count == agent.num_intruders:
                break

        if len(closest_states) == 0:
            closest_states = np.array([0, 0, 0, 0, 0, 0, 0]).reshape(1, 1, 7)

        ##### time consuming!! 12~13s #####
        if len(total_closest_states) == 0:
            total_closest_states = closest_states
        else:
            intruder_start = time.time()
            total_closest_states = np.append(
                tf.keras.preprocessing.sequence.pad_sequences(total_closest_states, agent.num_intruders,
                                                              dtype='float32'),
                tf.keras.preprocessing.sequence.pad_sequences(closest_states, agent.num_intruders, dtype='float32'),
                axis=0)
            intruder_end = time.time()
            agent.intruder_time += intruder_end - intruder_start
        ###################################

    if len(total_closest_states) == 0:
        total_closest_states = np.array([0, 0, 0, 0, 0, 0, 0]).reshape(1, agent.num_intruders, 7)

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
    global regular
    global tas_lvl

    r_id = state[:, 3].astype(int)  # route identifier
    pos_idx = [1, 1, 0]  # R0,R1: longitude; R2: latitude
    # ori_policy, rt = agent.act_saliency(onorm_state, onorm_context)
    ori_policy = agent.act(onorm_state, onorm_context)
    LEG_list = []

    for ac_idx in range(state.shape[0]):
        # ac_idx = 5
        agent.draw_aircraft_position_B(state, ac_idx, intruder_idx, ori_policy, max_ac)

        ACint_idx = intruder_idx[ac_idx]  # intruders of current aircraft
        if len(ACint_idx) == 0:
            continue

        num_intruder = len(ACint_idx)
        min_d = 100.0  # minimum difference of lon/lat between ownship and intersections
        r = int(state[ac_idx][3])  # ownship's route identifier
        pos_lvl = np.array([regular[r]] * (num_intruder + 1))  # noise level of lon/lat
        for inter in agent.intersections[r]:
            pos_diff = abs(state[ac_idx, pos_idx[r]] - agent.intersections[r][inter][pos_idx[r]])
            if min_d > pos_diff:
                min_d = pos_diff
        if min_d < regular[r]:
            pos_lvl[0] = min_d
        for i in range(agent.num_intruders):
            if onorm_context[ac_idx][i][6] != 0.0:
                route = int(onorm_context[ac_idx][i][2] * 2)  # intruder's route identifier: [0,0.5,1.0] --> [0,1,2]
                idx = i - agent.num_intruders + num_intruder  # actual index of intruder
                pos_diff = abs(state[ACint_idx[idx], pos_idx[route]] - agent.intersections[route][r][pos_idx[route]])
                if pos_diff < regular[route]:
                    pos_lvl[idx + 1] = pos_diff
                else:
                    pos_lvl[idx + 1] = regular[route]
        var_pos = pos_lvl ** 2 / 3
        var_tas = tas_lvl ** 2 / 3
        sigma = np.identity((num_intruder + 1) * 2)
        sigma *= var_tas
        for i in range(num_intruder + 1):
            sigma[i, i] = var_pos[i]
        inv_sigma = np.linalg.inv(sigma)
        noise = np.zeros((num_intruder + 1, num_sample))
        noise[0, :] = np.random.uniform(-pos_lvl[0], pos_lvl[0], size=num_sample)
        for i in range(num_intruder):
            noise[i + 1, :] = np.random.uniform(-pos_lvl[i + 1], pos_lvl[i + 1], size=num_sample)

        p_own_pos = noise[0, :] + state[ac_idx][pos_idx[r]]
        p_intru_pos = np.zeros((num_intruder, num_sample))
        for i in range(num_intruder):
            route = r_id[ACint_idx[i]]  # intruder's route identifier
            p_intru_pos[i, :] = noise[i + 1, :] + state[ACint_idx[i]][pos_idx[route]]

        p_state = np.array(state)  # perturbed state features (all features)
        # num_valid = 0  # number of valid perturbations
        z = np.zeros((num_intruder + 1, 2))  # see equation (4) in LEG paper
        target_y = np.zeros((agent.action_size, num_intruder + 1, 2))
        _ori_policy = ori_policy[ac_idx]
        for sample_idx in range(num_sample):
            p_state[ac_idx][pos_idx[r]] = p_own_pos[sample_idx]
            for i in range(num_intruder):
                route = r_id[ACint_idx[i]]  # intruder's route identifier
                p_state[ACint_idx[i]][pos_idx[route]] = p_intru_pos[i][sample_idx]

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
            # p_policy = np.array(_ori_policy)
            # y_diff = p_policy - _ori_policy

            # lon/lat noise
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


def LEG_perturbation(state, noise_lvl, num_sample, intruder_idx, traf, agent, onorm_state,
                     onorm_context):
    global max_ac

    # 'x' for longitude, 'y' for latitude
    route0_s_x, route0_e_x = agent.positions[0][1], agent.positions[0][3]
    route1_s_x, route1_e_x = agent.positions[1][1], agent.positions[1][3]
    route2_s_y, route2_e_y = agent.positions[2][0], agent.positions[2][2]
    routes = np.zeros((3, 2))
    routes[0][0], routes[0][1] = route0_s_x, route0_e_x
    routes[1][0], routes[1][1] = route1_e_x, route1_s_x
    routes[2][0], routes[2][1] = route2_s_y, route2_e_y

    latlon_idx = np.zeros((state.shape[0], 1))
    for i in range(state.shape[0]):
        if state[i][3] == 0 or state[i][3] == 1:  # route0 or route1
            latlon_idx[i] = 1  # longitude
        else:  # route2
            latlon_idx[i] = 0  # latitude
    latlon_idx = latlon_idx.astype(int)  # indices must be integers

    r_id = state[:, 3].astype(int)  # route identifier
    original_policy, rt = agent.act_saliency(onorm_state, onorm_context)
    leg_list = []

    for ac_idx in range(state.shape[0]):
        agent.draw_aircraft_position_B(state, ac_idx, intruder_idx, original_policy, max_ac)

        ACint_idx = intruder_idx[ac_idx]  # intruders of current aircraft
        if len(ACint_idx) == 0:
            continue

        num_intruder = len(ACint_idx)
        sigma = np.identity((num_intruder + 1) * 2)
        sigma *= noise_lvl ** 2
        inv_sigma = np.linalg.inv(sigma)
        # the first row of 'noise' is for ownship, others are for intruders
        noise = np.random.normal(loc=0, scale=noise_lvl, size=(num_intruder + 1, num_sample))
        # validity of ownship and intruder perturbation (True or False)
        perturbation_validity = np.zeros((num_sample, 2))

        perturb_ownship_latORlon = noise[0, :] * abs(routes[r_id[ac_idx]][0] -
                                                     routes[r_id[ac_idx]][1]) + \
                                   state[ac_idx][latlon_idx[ac_idx]]
        for i in range(num_sample):
            if routes[r_id[ac_idx]][0] >= perturb_ownship_latORlon[i] >= \
                    routes[r_id[ac_idx]][1]:
                perturbation_validity[i][0] = True
            else:
                perturbation_validity[i][0] = False

        perturb_intruder_latORlon = np.zeros((num_intruder, num_sample))
        for i in range(num_intruder):
            perturb_intruder_latORlon[i, :] = noise[i + 1, :] * \
                                              abs(routes[r_id[ACint_idx[i]]][0] -
                                                  routes[r_id[ACint_idx[i]]][1]) + \
                                              state[ACint_idx[i]][latlon_idx[ACint_idx[i]]]
            for j in range(num_sample):
                if routes[r_id[ACint_idx[i]]][0] >= perturb_intruder_latORlon[i][j] >= \
                        routes[r_id[ACint_idx[i]]][1]:
                    perturbation_validity[j][1] = True
                else:
                    perturbation_validity[j][1] = False

        perturb_state = np.array(state)  # perturbed state features (all features)
        num_valid = 0  # number of valid perturbations
        is_sector = True  # if the aircraft is still in the sector
        z = np.zeros((num_intruder + 1, 2))  # see equation (4) in LEG paper
        target_y = np.zeros((num_intruder + 1, 2))
        _original_policy = original_policy[ac_idx]
        action_idx = np.argmax(_original_policy)
        for sample_idx in range(num_sample):
            if perturbation_validity[sample_idx][0] and perturbation_validity[sample_idx][1]:
                perturb_state[ac_idx][latlon_idx[ac_idx]] = perturb_ownship_latORlon[sample_idx]

                for i in range(num_intruder):
                    perturb_state[ACint_idx[i]][latlon_idx[ACint_idx[i]]] = \
                        perturb_intruder_latORlon[i][sample_idx]

                # calculate 'distance to the goal'
                traf.lat = perturb_state[:, 0]
                traf.lon = perturb_state[:, 1]
                d_goal = dist_goal([traf.lat[ac_idx], traf.lon[ac_idx]], traf, ac_idx)
                if d_goal < 5.0:
                    continue
                agent.dist_goal[traf.id[ac_idx]] = d_goal
                for i in range(num_intruder):
                    d_goal = dist_goal([traf.lat[ACint_idx[i]], traf.lon[ACint_idx[i]]],
                                       traf, ACint_idx[i])
                    if d_goal < 5.0:
                        is_sector = False
                        break
                    agent.dist_goal[traf.id[ACint_idx[i]]] = d_goal
                if not is_sector:
                    is_sector = True
                    continue

                calState_start = time.time()
                pnorm_state, pnorm_context, pACint_idx, pd_all = LEG_getClosestAC(
                    perturb_state, traf, agent, ac_idx)
                calState_end = time.time()
                agent.calState_time += calState_end - calState_start
                pd_all = np.array(pd_all)
                if set(ACint_idx) == set(pACint_idx):
                    for i in range(pd_all.shape[0]):
                        pd_all[i][i] = 100.0
                    if np.all(pd_all >= 3.0):  # loss of separation (m or nm??? need to check)
                        num_valid += 1
                        for i in range(pd_all.shape[0]):
                            pd_all[i][i] = 0.0
                        spd_noise = np.random.normal(loc=0, scale=noise_lvl)
                        while (pnorm_state[1] + spd_noise) < 0.0 or \
                                (pnorm_state[1] + spd_noise) > 1.0:
                            spd_noise = np.random.normal(loc=0, scale=noise_lvl)
                        pnorm_state[1] += spd_noise
                        z[0][1] = spd_noise

                        for i in range(num_intruder):
                            spd_noise = np.random.normal(loc=0, scale=noise_lvl)
                            while (pnorm_context[0][i][1] + spd_noise) < 0.0 \
                                    or (pnorm_context[0][i][1] + spd_noise) > 1.0:
                                spd_noise = np.random.normal(loc=0, scale=noise_lvl)
                            pnorm_context[0][i][1] += spd_noise
                            if ACint_idx[i] == pACint_idx[i]:
                                z[i + 1][1] = spd_noise
                            else:
                                ori_idx = ACint_idx.index(pACint_idx[i])
                                z[ori_idx + 1][1] = spd_noise

                        # forward_start = time.time()
                        pnorm_state = np.array([pnorm_state])
                        perturb_policy, runtime = agent.act_saliency(pnorm_state, pnorm_context)
                        # forward_end = time.time()
                        # agent.forward_time += forward_end - forward_start
                        agent.forward_time += runtime
                        # agent.draw_aircraft_position_B(perturb_state, aircraft_idx,
                        #                                pintruder_idx, perturb_policy,
                        #                                max_ac)
                        y_diff = perturb_policy[0][action_idx] - _original_policy[action_idx]

                        # lat/lon
                        z[0][0] = noise[0][sample_idx]
                        for i in range(num_intruder):
                            z[i + 1][0] = noise[i + 1][sample_idx]

                        target_y += y_diff * z
                    else:
                        continue
                else:
                    # agent.draw_aircraft_position_B(perturb_state, aircraft_idx, pintruder_idx,
                    #                                original_policy, max_ac)
                    continue
            else:
                continue

        target_y /= num_valid
        target_y = target_y.flatten()
        leg = inv_sigma @ target_y
        leg = leg.reshape(num_intruder + 1, 2)

        # normalized LEG
        leg = np.abs(leg)
        leg_min, leg_max = np.min(leg), np.max(leg)
        leg = (leg - leg_min) / (leg_max - leg_min + K.epsilon())
        leg_list.append(leg.tolist())
        agent.LEG_heatmap(leg, ac_idx, ACint_idx)

        print('aircraft {} num_valid: {}'.format(ac_idx, num_valid))

    # restore the states to original states
    traf.lat = state[:, 0]
    traf.lon = state[:, 1]
    for i in range(state.shape[0]):
        d_goal = dist_goal([traf.lat[i], traf.lon[i]], traf, i)
        agent.dist_goal[traf.id[i]] = d_goal

    return leg_list
