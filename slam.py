from __future__ import division
import numpy as np
import slam_utils
import tree_extraction

from scipy.stats import chi2

def fix_P(P):
    P = slam_utils.make_symmetric(P)
    P += 1e-6*np.eye(P.shape[0])
    return P

def motion_model(u, dt, ekf_state, vehicle_params):
    """
    Computes the discretized motion model for the given vehicle as well as its Jacobian

    Returns:
        f(x,u), a 3x1 vector corresponding to motion x_{t+1} - x_t given the odometry u.

        df/dX, the 3x3 Jacobian of f with respect to the vehicle state (x, y, phi)
    """

    ###
    # Implement the vehicle model and its Jacobian you derived.
    ###
    # params
    a = vehicle_params['a']
    b = vehicle_params['b']
    H = vehicle_params['H']
    L = vehicle_params['L']

    # pose
    pose = ekf_state['x'][:3]
    xv = pose[0]
    yv = pose[1]
    phi = pose[2]

    # control
    ve = u[0]
    alpha = u[1]
    vc = ve/(1 - (np.tan(alpha)*H/L))

    const = vc*np.tan(alpha)/L
    # calculation
    motion = np.zeros((3, 1))
    motion[0] = xv + dt*(vc*np.cos(phi) - const*(a*np.sin(phi) + b*np.cos(phi)))
    motion[1] = yv + dt*(vc*np.sin(phi) + const*(a*np.cos(phi) - b*np.sin(phi)))
    motion[2] = phi + dt*const

    # G = np.zeros((3, 3))
    G = np.eye(3)
    G[0, 2] = -vc*dt*np.sin(phi) - dt*const*(a*np.cos(phi) - b*np.sin(phi))
    G[1, 2] = vc*dt*np.cos(phi) - dt*const*(a*np.sin(phi) + b*np.cos(phi))

    return motion, G


def odom_predict(u, dt, ekf_state, vehicle_params, sigmas):
    '''
    Perform the propagation step of the EKF filter given an odometry measurement u
    and time step dt where u = (ve, alpha) as shown in the vehicle/motion model.

    Returns the new ekf_state.
    '''

    ###
    # Implement the propagation
    ###
    motion, G = motion_model(u, dt, ekf_state, vehicle_params)
    # motion += np.vstack([ekf_state['x'][0], ekf_state['x'][1], ekf_state['x'][2]])
    # G = G + np.eye(3)

    P = ekf_state['P'][:3, :3]
    sigma_xy = sigmas['xy']**2
    sigma_phi = sigmas['phi']**2

    Q = sigma_xy*np.eye(3)
    Q[2, 2] = sigma_phi

    ekf_state['x'][:3] = motion.reshape(3,)
    ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
    ekf_state['P'][:3, :3] = np.dot(np.dot(G, P), G.T) + Q
    ekf_state['P'] = fix_P(ekf_state['P'])
    return ekf_state


def gps_update(gps, ekf_state, sigmas):
    '''
    Perform a measurement update of the EKF state given a GPS measurement (x,y).

    Returns the updated ekf_state.
    '''

    ###
    # Implement the GPS update.
    ###
    pose = ekf_state['x'][:3]
    P = ekf_state['P'][:3, :3]

    R = sigmas['gps']**2*np.eye(2);
    H = np.array([[1, 0, 0],
        [0, 1, 0]])

    measurement_error = (np.array([gps[0], gps[1]]) - pose[:2]).reshape(2, 1)
    S = np.dot(np.dot(H, P), H.T) + R

    d = np.dot(np.dot(measurement_error.T, np.linalg.inv(S)), measurement_error)
    if d > chi2.ppf(0.999,2):
        print("GPS data is thrown out")
        return ekf_state

    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))

    ekf_state['x'][:3] = pose + np.dot(K, measurement_error).reshape(3,)
    ekf_state['P'][:3, :3] = np.dot((np.eye(3) - np.dot(K, H)), P)
    ekf_state['P'] = fix_P(ekf_state['P'])
    return ekf_state


def laser_measurement_model(ekf_state, landmark_id):
    '''
    Returns the measurement model for a (range,bearing) sensor observing the
    mapped landmark with id 'landmark_id' along with its jacobian.

    Returns:
        h(x, l_id): the 2x1 predicted measurement vector [r_hat, theta_hat].

        dh/dX: For a measurement state with m mapped landmarks, i.e. a state vector of
                dimension 3 + 2*m, this should return the full 2 by 3+2m Jacobian
                matrix corresponding to a measurement of the landmark_id'th feature.
    '''

    ###
    # Implement the measurement model and its Jacobian you derived
    ###
    state = ekf_state['x']
    xv = state[0]
    yv = state[1]
    phi = state[2]
    xL = state[3+landmark_id*2]
    yL = state[3+landmark_id*2+1]

    zhat = np.zeros((2,1))
    zhat[0] = np.sqrt((xL - xv)**2+(yL - yv)**2)
    zhat[1] = slam_utils.clamp_angle(np.arctan2(yL - yv, xL- xv) - phi)

    H = np.zeros((2, 3+2*ekf_state['num_landmarks']))

    H[0, 0] = (xv - xL)/zhat[0]
    H[0, 1] = (yv - yL)/zhat[0]
    H[0, 2] = 0
    H[0, 3+2*landmark_id] = -H[0, 0]
    H[0, 3+2*landmark_id+1] = -H[0, 1]

    H[1, 0] = (yL - yv)/(zhat[0]**2)
    H[1, 1] = (xv - xL)/(zhat[0]**2)
    H[1, 2] = -1
    H[1, 3+2*landmark_id] = -H[1, 0]
    H[1, 3+2*landmark_id+1] = -H[1, 1]

    return zhat, H

def initialize_landmark(ekf_state, tree):
    '''
    Initialize a newly observed landmark in the filter state, increasing its
    dimension by 2.

    Returns the new ekf_state.
    '''

    ###
    # Implement this function.
    ###
    r = tree[0]
    b = tree[1]

    pose = ekf_state['x']
    xv = pose[0]
    yv = pose[1]
    phi = pose[2]

    xL = xv + r*np.cos(phi + b)
    yL = yv + r*np.sin(phi + b)

    ekf_state['x'] = np.append(ekf_state['x'], [xL, yL])
    mP, nP = ekf_state['P'].shape
    ekf_state['P'] = np.block([[ekf_state['P'], np.zeros((mP, 2))],
        [np.zeros((2, nP)), 5*np.eye(2)]])
    ekf_state['P'] = fix_P(ekf_state['P'])
    ekf_state['num_landmarks'] += 1

    return ekf_state


def compute_data_association(ekf_state, measurements, sigmas, params):
    '''
    Computes measurement data association.

    Given a robot and map state and a set of (range,bearing) measurements,
    this function should compute a good data association, or a mapping from
    measurements to landmarks.

    Returns an array 'assoc' such that:
        assoc[i] == j if measurement i is determined to be an observation of landmark j,
        assoc[i] == -1 if measurement i is determined to be a new, previously unseen landmark, or,
        assoc[i] == -2 if measurement i is too ambiguous to use and should be discarded.
    '''

    if ekf_state["num_landmarks"] == 0:
        # set association to init new landmarks for all measurements
        return [-1 for m in measurements]

    ###
    # Implement this function.
    ###
    state = ekf_state['x']
    xv = state[0]
    yv = state[1]
    phi = state[2]

    R = np.eye(2)
    R[0,0] = sigmas['range']**2
    R[1,1] = sigmas['bearing']**2

    num_measurements = len(measurements)
    num_landmarks = ekf_state["num_landmarks"]
    M = np.ones((num_measurements, num_measurements+num_landmarks))*chi2.ppf(0.95, 2)
    for i, m in enumerate(measurements):
        for j in range(num_landmarks):
            zhat, full_H = laser_measurement_model(ekf_state, j)

            # P = np.zeros((5,5))
            # P[:3, :3] = ekf_state['P'][:3, :3]
            # P[3:, 3:] = ekf_state['P'][3+j*2:3+(j+1)*2, 3+j*2:3+(j+1)*2]
            # P[3:, :3] = ekf_state['P'][3+j*2:3+(j+1)*2, :3]
            # P[:3, 3:] = ekf_state['P'][:3, 3+j*2:3+(j+1)*2]
            P = ekf_state['P']

            # H = np.zeros((2,5))
            # H[:2, :3] = full_H[:2, :3]
            # H[:2, 3:] = full_H[:2, 3+j*2:3+(j+1)*2]
            H = full_H

            r = np.vstack([m[0], m[1]]) - zhat
            S = np.dot(np.dot(H, P), H.T) + R

            M[i, j] = np.dot(np.dot(r.T, np.linalg.inv(S)), r)

    matchings = slam_utils.solve_cost_matrix_heuristic(M.copy())
    # row_ind, col_ind = linear_sum_assignment(M.copy())
    # matchings = list(zip(row_ind, col_ind))
    assoc = [-2 for m in measurements]
    for m in matchings:
        if m[1] >= num_landmarks and np.min(M[m[0],:num_landmarks]) > chi2.ppf(0.99, 2):
            assoc[m[0]] = -1
        elif m[1] >= num_landmarks:
            continue
        else:
            assoc[m[0]] = m[1]

    return assoc

def laser_update(trees, assoc, ekf_state, sigmas, params):
    '''
    Perform a measurement update of the EKF state given a set of tree measurements.

    trees is a list of measurements, where each measurement is a tuple (range, bearing, diameter).

    assoc is the data association for the given set of trees, i.e. trees[i] is an observation of the
    ith landmark. If assoc[i] == -1, initialize a new landmark with the function initialize_landmark
    in the state for measurement i. If assoc[i] == -2, discard the measurement as
    it is too ambiguous to use.

    The diameter component of the measurement can be discarded.

    Returns the ekf_state.
    '''

    ###
    # Implement the EKF update for a set of range, bearing measurements.
    ###
    R = np.eye(2)
    R[0,0] = sigmas['range']**2
    R[1,1] = sigmas['bearing']**2

    for i, m in enumerate(trees):
        if assoc[i] == -1:
            ekf_state = initialize_landmark(ekf_state, m)
        elif assoc[i] == -2:
            continue
        else:
            j = assoc[i]
            zhat, H = laser_measurement_model(ekf_state, j)

            # P = np.zeros((5,5))
            # P[:3, :3] = ekf_state['P'][:3, :3]
            # P[3:, 3:] = ekf_state['P'][3+j*2:3+(j+1)*2, 3+j*2:3+(j+1)*2]
            # P[:3, 3:] = ekf_state['P'][:3, 3+j*2:3+(j+1)*2]
            # P[3:, :3] = ekf_state['P'][3+j*2:3+(j+1)*2, :3]
            P = ekf_state['P']

            # H = np.zeros((2,5))
            # H[:2, :3] = full_H[:2, :3]
            # H[:2, 3:] = full_H[:2, 3+j*2:3+(j+1)*2]

            r = np.vstack([m[0], m[1]]) - zhat
            S = np.dot(np.dot(H, P), H.T) + R

            if np.dot(np.dot(r.T, np.linalg.inv(S)), r) > chi2.ppf(0.999,2):
                continue

            K = np.dot(np.dot(P, H.T), np.linalg.inv(S))

            x_update = np.dot(K, r).flatten()
            # ekf_state['x'][:3] += x_update[:3]
            # ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
            # ekf_state['x'][3+j*2:3+(j+1)*2] += x_update[3+j*2:3+(j+1)*2]
            ekf_state['x'] += x_update

            # P_update = np.dot((np.eye(5) - np.dot(K, H)), P)
            # ekf_state['P'][:3, :3] = P_update[:3, :3]
            # ekf_state['P'][:3, 3+j*2:3+(j+1)*2] = P_update[:3, 3:]
            # ekf_state['P'][3+j*2:3+(j+1)*2, :3] = P_update[3:, :3]
            # ekf_state['P'][3+j*2:3+(j+1)*2, 3+j*2:3+(j+1)*2] = P_update[3:, 3:]
            ekf_state['P'] = np.dot((np.eye(P.shape[0]) - np.dot(K, H)), P)
    ekf_state['P'] = fix_P(ekf_state['P'])

    return ekf_state


def run_ekf_slam(events, ekf_state_0, vehicle_params, filter_params, sigmas):
    last_odom_t = -1
    ekf_state = {
            'x': ekf_state_0['x'].copy(),
            'P': ekf_state_0['P'].copy(),
            'num_landmarks': ekf_state_0['num_landmarks']
            }

    state_history = {
            't': [0],
            'x': ekf_state['x'],
            'P': np.diag(ekf_state['P'])
            }

    if filter_params["do_plot"]:
        plot = slam_utils.init_plot()

    for i, event in enumerate(events):
        t = event[1][0]
        if i % 1000 == 0:
            print("t = {}".format(t))

        if event[0] == 'gps':
            gps_msmt = event[1][1:]
            ekf_state = gps_update(gps_msmt, ekf_state, sigmas)

        elif event[0] == 'odo':
            if last_odom_t < 0:
                last_odom_t = t
                continue
            u = event[1][1:]
            dt = t - last_odom_t
            ekf_state = odom_predict(u, dt, ekf_state, vehicle_params, sigmas)
            last_odom_t = t

        else:
            # Laser
            scan = event[1][1:]
            trees = tree_extraction.extract_trees(scan, filter_params)
            assoc = compute_data_association(ekf_state, trees, sigmas, filter_params)
            ekf_state = laser_update(trees, assoc, ekf_state, sigmas, filter_params)
            if filter_params["do_plot"]:
                slam_utils.do_plot(state_history['x'], ekf_state, trees, scan, assoc, plot, filter_params)

        state_history['x'] = np.vstack((state_history['x'], ekf_state['x'][0:3]))
        state_history['P'] = np.vstack((state_history['P'], np.diag(ekf_state['P'][:3, :3])))
        state_history['t'].append(t)

    return state_history


def main():
    odo = slam_utils.read_data_file("data/DRS.txt")
    gps = slam_utils.read_data_file("data/GPS.txt")
    laser = slam_utils.read_data_file("data/LASER.txt")

    # collect all events and sort by time
    events = [('gps', x) for x in gps]
    events.extend([('laser', x) for x in laser])
    events.extend([('odo', x) for x in odo])

    events = sorted(events, key=lambda event: event[1][0])

    vehicle_params = {
            "a": 3.78,
            "b": 0.50,
            "L": 2.83,
            "H": 0.76
            }

    filter_params = {
            # measurement params
            "max_laser_range": 75,  # meters

            # general...
            "do_plot": True,
            "plot_raw_laser": True,
            "plot_map_covariances": False

            # Add other parameters here if you need to...
            }

    # Noise values
    sigmas = {
            # Motion model noise
            "xy": 0.05,
            "phi": 0.5 * np.pi / 180,

            # Measurement noise
            "gps": 3,
            "range": 0.5,
            "bearing": 5 * np.pi / 180
            }

    # Initial filter state
    ekf_state = {
            "x": np.array([gps[0, 1], gps[0, 2], 36 * np.pi / 180]),
            "P": np.diag([.1, .1, 1]),
            "num_landmarks": 0
            }

    run_ekf_slam(events, ekf_state, vehicle_params, filter_params, sigmas)


if __name__ == '__main__':
    main()
