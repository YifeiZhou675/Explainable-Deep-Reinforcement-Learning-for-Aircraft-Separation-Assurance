import numpy as np
import matplotlib.pyplot as plt
import random


def perturb_case_A():
    plt.clf()
    ax = plt.gca()
    ax.set_aspect(1)

    R0 = np.load('../route0_A.npy')
    R1 = np.load('../route1_A.npy')

    R0_s_lon, R0_e_lon = R0[0][0], R0[0][-1]
    R0_s_lat, R0_e_lat = R0[1][0], R0[1][-1]
    R1_s_lon, R1_e_lon = R1[0][0], R1[0][-1]
    R1_s_lat, R1_e_lat = R1[1][0], R1[1][-1]

    # positions = np.load('../routes/case_study_a_route.npy')
    # R0_s_lat, R0_s_lon, R0_e_lat, R0_e_lon, h = positions[0]
    # R1_s_lat, R1_s_lon, R1_e_lat, R1_e_lon, h = positions[1]

    R0_lon = R0[0]
    R0_lat = R0[1]
    plt.plot(R0_lon, R0_lat)
    plt.plot(R0_e_lon, R0_e_lat, marker=(3, 0, 40.4977), markersize=12, markerfacecolor='b',
             markeredgecolor='b')
    plt.text(R0_e_lon, R0_e_lat - 0.04, 'R0', fontsize='large', fontweight='bold')

    R1_lon = R1[0]
    R1_lat = R1[1]
    plt.plot(R1_lon, R1_lat)
    plt.plot(R1_e_lon, R1_e_lat, marker=(3, 0, 22.7517), markersize=12, markerfacecolor='b',
             markeredgecolor='b')
    plt.text(R1_e_lon, R1_e_lat + 0.02, 'R1', fontsize='large', fontweight='bold')

    num_p = 15
    p_lons = np.zeros(num_p)
    for i in range(num_p):
        p_lons[i] = random.uniform(np.min([R0_s_lon, R0_e_lon]), np.max([R0_s_lon, R0_e_lon]))
        if p_lons[i] < 0:
            p_lons[i] += 360

    R0_se = np.array([[R0[0][0], R0[0][-1]], [R0[1][0], R0[1][-1]]])
    R0_se[0] += 360
    R0_se[1] = 90 - R0_se[1]
    R0_se = np.radians(R0_se)

    x_s0 = np.sin(R0_se[1][0]) * np.cos(R0_se[0][0])
    y_s0 = np.sin(R0_se[1][0]) * np.sin(R0_se[0][0])
    z_s0 = np.cos(R0_se[1][0])
    x_e0 = np.sin(R0_se[1][1]) * np.cos(R0_se[0][1])
    y_e0 = np.sin(R0_se[1][1]) * np.sin(R0_se[0][1])
    z_e0 = np.cos(R0_se[1][1])

    x_n, y_n, z_n = np.cross([x_s0, y_s0, z_s0], [x_e0, y_e0, z_e0])
    p_lats = np.arctan(-z_n / (x_n * np.cos(np.radians(p_lons)) + y_n *
                               np.sin(np.radians(p_lons))))
    p_lats = 90 - np.degrees(p_lats)

    for i in range(num_p):
        p_lons[i] -= 360
    plt.plot(p_lons, p_lats, 'r*', markersize=5)

    for i in range(num_p):
        p_lons[i] = random.uniform(np.min([R1_s_lon, R1_e_lon]), np.max([R1_s_lon, R1_e_lon]))
        if p_lons[i] < 0:
            p_lons[i] += 360

    R1_se = np.array([[R1[0][0], R1[0][-1]], [R1[1][0], R1[1][-1]]])
    R1_se[0] += 360
    R1_se[1] = 90 - R1_se[1]
    R1_se = np.radians(R1_se)

    x_s1 = np.sin(R1_se[1][0]) * np.cos(R1_se[0][0])
    y_s1 = np.sin(R1_se[1][0]) * np.sin(R1_se[0][0])
    z_s1 = np.cos(R1_se[1][0])
    x_e1 = np.sin(R1_se[1][1]) * np.cos(R1_se[0][1])
    y_e1 = np.sin(R1_se[1][1]) * np.sin(R1_se[0][1])
    z_e1 = np.cos(R1_se[1][1])

    x_n, y_n, z_n = np.cross([x_s1, y_s1, z_s1], [x_e1, y_e1, z_e1])
    p_lats = np.arctan(-z_n / (x_n * np.cos(np.radians(p_lons)) + y_n *
                               np.sin(np.radians(p_lons))))
    p_lats = 90 - np.degrees(p_lats)

    for i in range(num_p):
        p_lons[i] -= 360
    plt.plot(p_lons, p_lats, 'r*', markersize=5)

    plt.show()


def perturb_case_C():
    plt.clf()
    ax = plt.gca()
    ax.set_aspect(1)

    R0 = np.load('../route0_C.npy')
    R1 = np.load('../route1_C.npy')
    R2 = np.load('../route2_C.npy')

    R0_s_lon, R0_e_lon = R0[0][0], R0[0][-1]
    R0_s_lat, R0_e_lat = R0[1][0], R0[1][-1]
    R1_s_lon, R1_e_lon = R1[0][0], R1[0][-1]
    R1_s_lat, R1_e_lat = R1[1][0], R1[1][-1]
    R2_s_lon, R2_e_lon = R2[0][0], R2[0][-1]
    R2_s_lat, R2_e_lat = R2[1][0], R2[1][-1]

    R0_lon = R0[0]
    R0_lat = R0[1]
    plt.plot(R0_lon, R0_lat)
    plt.plot(R0_e_lon, R0_e_lat, marker=(3, 0, 61), markersize=12, markerfacecolor='b',
             markeredgecolor='b')
    plt.text(R0_e_lon - 0.02, R0_e_lat - 0.25, 'R0', fontsize='large', fontweight='bold')

    R1_lon = R1[0]
    R1_lat = R1[1]
    plt.plot(R1_lon, R1_lat)
    plt.plot(R1_e_lon, R1_e_lat, marker=(3, 0, 22.7517), markersize=12, markerfacecolor='b',
             markeredgecolor='b')
    plt.text(R1_e_lon - 0.02, R1_e_lat - 0.23, 'R1', fontsize='large', fontweight='bold')

    R2_lon = R2[0]
    R2_lat = R2[1]
    plt.plot(R2_lon, R2_lat)
    plt.plot(R2_e_lon, R2_e_lat, marker=(3, 0, 20), markersize=12, markerfacecolor='b',
             markeredgecolor='b')
    plt.text(R2_e_lon + 0.08, R2_e_lat, 'R2', fontsize='large', fontweight='bold')

    num_p = 15
    p_lons = np.zeros(num_p)
    for i in range(num_p):
        p_lons[i] = random.uniform(np.min([R0_s_lon, R0_e_lon]), np.max([R0_s_lon, R0_e_lon]))
        if p_lons[i] < 0:
            p_lons[i] += 360

    R0_se = np.array([[R0[0][0], R0[0][-1]], [R0[1][0], R0[1][-1]]])
    R0_se[0] += 360
    R0_se[1] = 90 - R0_se[1]
    R0_se = np.radians(R0_se)

    x_s0 = np.sin(R0_se[1][0]) * np.cos(R0_se[0][0])
    y_s0 = np.sin(R0_se[1][0]) * np.sin(R0_se[0][0])
    z_s0 = np.cos(R0_se[1][0])
    x_e0 = np.sin(R0_se[1][1]) * np.cos(R0_se[0][1])
    y_e0 = np.sin(R0_se[1][1]) * np.sin(R0_se[0][1])
    z_e0 = np.cos(R0_se[1][1])

    x_n, y_n, z_n = np.cross([x_s0, y_s0, z_s0], [x_e0, y_e0, z_e0])
    p_lats = np.arctan(-z_n / (x_n * np.cos(np.radians(p_lons)) + y_n *
                               np.sin(np.radians(p_lons))))
    p_lats = 90 - np.degrees(p_lats)

    for i in range(num_p):
        p_lons[i] -= 360
    plt.plot(p_lons, p_lats, 'r*', markersize=5)

    for i in range(num_p):
        p_lons[i] = random.uniform(np.min([R1_s_lon, R1_e_lon]), np.max([R1_s_lon, R1_e_lon]))
        if p_lons[i] < 0:
            p_lons[i] += 360

    R1_se = np.array([[R1[0][0], R1[0][-1]], [R1[1][0], R1[1][-1]]])
    R1_se[0] += 360
    R1_se[1] = 90 - R1_se[1]
    R1_se = np.radians(R1_se)

    x_s1 = np.sin(R1_se[1][0]) * np.cos(R1_se[0][0])
    y_s1 = np.sin(R1_se[1][0]) * np.sin(R1_se[0][0])
    z_s1 = np.cos(R1_se[1][0])
    x_e1 = np.sin(R1_se[1][1]) * np.cos(R1_se[0][1])
    y_e1 = np.sin(R1_se[1][1]) * np.sin(R1_se[0][1])
    z_e1 = np.cos(R1_se[1][1])

    x_n, y_n, z_n = np.cross([x_s1, y_s1, z_s1], [x_e1, y_e1, z_e1])
    p_lats = np.arctan(-z_n / (x_n * np.cos(np.radians(p_lons)) + y_n *
                               np.sin(np.radians(p_lons))))
    p_lats = 90 - np.degrees(p_lats)

    for i in range(num_p):
        p_lons[i] -= 360
    plt.plot(p_lons, p_lats, 'r*', markersize=5)

    R2_s_lon, R2_s_lat = R2_lon[5], R2_lat[5]  # we don't consider the small tail of R2
    for i in range(num_p):
        p_lons[i] = random.uniform(np.min([R2_s_lon, R2_e_lon]), np.max([R2_s_lon, R2_e_lon]))
        if p_lons[i] < 0:
            p_lons[i] += 360

    R2_se = np.array([[R2[0][5], R2[0][-1]], [R2[1][5], R2[1][-1]]])
    R2_se[0] += 360
    R2_se[1] = 90 - R2_se[1]
    R2_se = np.radians(R2_se)

    x_s2 = np.sin(R2_se[1][0]) * np.cos(R2_se[0][0])
    y_s2 = np.sin(R2_se[1][0]) * np.sin(R2_se[0][0])
    z_s2 = np.cos(R2_se[1][0])
    x_e2 = np.sin(R2_se[1][1]) * np.cos(R2_se[0][1])
    y_e2 = np.sin(R2_se[1][1]) * np.sin(R2_se[0][1])
    z_e2 = np.cos(R2_se[1][1])

    x_n, y_n, z_n = np.cross([x_s2, y_s2, z_s2], [x_e2, y_e2, z_e2])
    p_lats = np.arctan(-z_n / (x_n * np.cos(np.radians(p_lons)) + y_n *
                               np.sin(np.radians(p_lons))))
    p_lats = 90 - np.degrees(p_lats)

    for i in range(num_p):
        p_lons[i] -= 360
    plt.plot(p_lons, p_lats, 'r*', markersize=5)

    plt.show()


def perturb_case_E2():
    plt.clf()
    ax = plt.gca()
    ax.set_aspect(1)

    R0 = np.load('../route0_E2.npy')
    R1 = np.load('../route1_E2.npy')
    R2 = np.load('../route2_E2.npy')
    R3 = np.load('../route3_E2.npy')
    N = np.load('../N_E2.npy')

    R0_s_lon, R0_e_lon = R0[0][0], R0[0][-1]
    R0_s_lat, R0_e_lat = R0[1][0], R0[1][-1]
    R1_s_lon, R1_e_lon = R1[0][0], R1[0][-1]
    R1_s_lat, R1_e_lat = R1[1][0], R1[1][-1]
    R2_s_lon, R2_e_lon = R2[0][4], R2[0][-1]
    R2_s_lat, R2_e_lat = R2[1][4], R2[1][-1]
    R3_s_lon, R3_e_lon = R3[0][5], R3[0][-1]
    R3_s_lat, R3_e_lat = R3[1][5], R3[1][-1]

    R0_lon = R0[0]
    R0_lat = R0[1]
    plt.plot(R0_lon, R0_lat)
    plt.plot(R0_e_lon, R0_e_lat, marker=(3, 0, 61), markersize=12, markerfacecolor='b',
             markeredgecolor='b')
    plt.text(R0_e_lon - 0.02, R0_e_lat - 0.25, 'R0', fontsize='large', fontweight='bold')

    R1_lon = R1[0]
    R1_lat = R1[1]
    plt.plot(R1_lon, R1_lat)
    plt.plot(R1_e_lon, R1_e_lat, marker=(3, 0, 22.7517), markersize=12, markerfacecolor='b',
             markeredgecolor='b')
    plt.text(R1_e_lon - 0.02, R1_e_lat - 0.23, 'R1', fontsize='large', fontweight='bold')

    R2_lon = R2[0, 4:-1]
    R2_lat = R2[1, 4:-1]
    plt.plot(R2_lon, R2_lat)
    plt.plot(R2_e_lon, R2_e_lat, marker=(3, 0, 20), markersize=12, markerfacecolor='b',
             markeredgecolor='b')
    plt.text(R2_e_lon + 0.08, R2_e_lat, 'R2', fontsize='large', fontweight='bold')

    R3_lon = R3[0, 5:-1]
    R3_lat = R3[1, 5:-1]
    plt.plot(R3_lon, R3_lat)
    plt.plot(R3_e_lon, R3_e_lat, marker=(3, 0, 20), markersize=12, markerfacecolor='b',
             markeredgecolor='b')
    plt.text(R3_e_lon + 0.08, R3_e_lat, 'R3', fontsize='large', fontweight='bold')

    num_p = 15
    p_lons = np.zeros(num_p)
    for i in range(num_p):
        p_lons[i] = random.uniform(np.min([R0_s_lon, R0_e_lon]), np.max([R0_s_lon, R0_e_lon]))
        if p_lons[i] < 0:
            p_lons[i] += 360
    #
    # R0_se = np.array([[R0[0][0], R0[0][-1]], [R0[1][0], R0[1][-1]]])
    # R0_se[0] += 360
    # R0_se[1] = 90 - R0_se[1]
    # R0_se = np.radians(R0_se)
    #
    # x_s0 = np.sin(R0_se[1][0]) * np.cos(R0_se[0][0])
    # y_s0 = np.sin(R0_se[1][0]) * np.sin(R0_se[0][0])
    # z_s0 = np.cos(R0_se[1][0])
    # x_e0 = np.sin(R0_se[1][1]) * np.cos(R0_se[0][1])
    # y_e0 = np.sin(R0_se[1][1]) * np.sin(R0_se[0][1])
    # z_e0 = np.cos(R0_se[1][1])
    #
    # x_n, y_n, z_n = np.cross([x_s0, y_s0, z_s0], [x_e0, y_e0, z_e0])
    p_lats = np.arctan(-N[0, 2] / (N[0, 0] * np.cos(np.radians(p_lons)) + N[0, 1] *
                       np.sin(np.radians(p_lons))))
    p_lats = 90 - np.degrees(p_lats)

    for i in range(num_p):
        p_lons[i] -= 360
    plt.plot(p_lons, p_lats, 'r*', markersize=5)

    for i in range(num_p):
        p_lons[i] = random.uniform(np.min([R1_s_lon, R1_e_lon]), np.max([R1_s_lon, R1_e_lon]))
        if p_lons[i] < 0:
            p_lons[i] += 360

    # R1_se = np.array([[R1[0][0], R1[0][-1]], [R1[1][0], R1[1][-1]]])
    # R1_se[0] += 360
    # R1_se[1] = 90 - R1_se[1]
    # R1_se = np.radians(R1_se)
    #
    # x_s1 = np.sin(R1_se[1][0]) * np.cos(R1_se[0][0])
    # y_s1 = np.sin(R1_se[1][0]) * np.sin(R1_se[0][0])
    # z_s1 = np.cos(R1_se[1][0])
    # x_e1 = np.sin(R1_se[1][1]) * np.cos(R1_se[0][1])
    # y_e1 = np.sin(R1_se[1][1]) * np.sin(R1_se[0][1])
    # z_e1 = np.cos(R1_se[1][1])
    #
    # x_n, y_n, z_n = np.cross([x_s1, y_s1, z_s1], [x_e1, y_e1, z_e1])
    p_lats = np.arctan(-N[1, 2] / (N[1, 0] * np.cos(np.radians(p_lons)) + N[1, 1] *
                       np.sin(np.radians(p_lons))))
    p_lats = 90 - np.degrees(p_lats)

    for i in range(num_p):
        p_lons[i] -= 360
    plt.plot(p_lons, p_lats, 'r*', markersize=5)

    # R2_s_lon, R2_s_lat = R2_lon[5], R2_lat[5]  # we don't consider the small tail of R2
    for i in range(num_p):
        p_lons[i] = random.uniform(np.min([R2_s_lon, R2_e_lon]), np.max([R2_s_lon, R2_e_lon]))
        if p_lons[i] < 0:
            p_lons[i] += 360

    # R2_se = np.array([[R2[0][5], R2[0][-1]], [R2[1][5], R2[1][-1]]])
    # R2_se[0] += 360
    # R2_se[1] = 90 - R2_se[1]
    # R2_se = np.radians(R2_se)
    #
    # x_s2 = np.sin(R2_se[1][0]) * np.cos(R2_se[0][0])
    # y_s2 = np.sin(R2_se[1][0]) * np.sin(R2_se[0][0])
    # z_s2 = np.cos(R2_se[1][0])
    # x_e2 = np.sin(R2_se[1][1]) * np.cos(R2_se[0][1])
    # y_e2 = np.sin(R2_se[1][1]) * np.sin(R2_se[0][1])
    # z_e2 = np.cos(R2_se[1][1])

    # x_n, y_n, z_n = np.cross([x_s2, y_s2, z_s2], [x_e2, y_e2, z_e2])
    p_lats = np.arctan(-N[2, 2] / (N[2, 0] * np.cos(np.radians(p_lons)) + N[2, 1] *
                       np.sin(np.radians(p_lons))))
    p_lats = 90 - np.degrees(p_lats)

    for i in range(num_p):
        p_lons[i] -= 360
    plt.plot(p_lons, p_lats, 'r*', markersize=5)

    for i in range(num_p):
        p_lons[i] = random.uniform(np.min([R3_s_lon, R3_e_lon]), np.max([R3_s_lon, R3_e_lon]))
        if p_lons[i] < 0:
            p_lons[i] += 360

    p_lats = np.arctan(-N[3, 2] / (N[3, 0] * np.cos(np.radians(p_lons)) + N[3, 1] *
                       np.sin(np.radians(p_lons))))
    p_lats = 90 - np.degrees(p_lats)

    for i in range(num_p):
        p_lons[i] -= 360
    plt.plot(p_lons, p_lats, 'r*', markersize=5)

    plt.show()


perturb_case_E2()

# a = np.array([[1, 5], [2, 6], [3, 7], [4, 8]])
# print(a)
# b = a.flatten('F')
# print(b)
# c = b.reshape(a.shape, order='F')
# print(c)

# arr = np.array([0, 0, 0, 100, 100, 100.])
# arr[0], arr[1], arr[2] = 1, 2, 3
# print(arr)

# x = [10, 2, 29, 8]
# idx = np.argsort(x)[::-1]
# print(x, idx)
