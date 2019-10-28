import numpy as np
import nest
from itertools import accumulate
import matplotlib.pyplot as plt

from world_populations import create_planner, create_cortex, \
                              create_sensory_io, create_motor_io
import trajectories

nest.Install("extracerebmodule")


trial_len = 300


def run_simulation(n=400, n_trials=1, prism=0.0):
    nest.ResetKernel()
    trajectories.save_file(prism, trial_len)

    planner = create_planner(n, prism)
    cortex = create_cortex(n)

    planner.connect(cortex)

    nest.Simulate(trial_len * n_trials)

    return cortex.get_events()


def integrate_torque(evs, ts, j_id, pop_size, pop_offset):
    j_evs = []
    j_ts = []
    pop_size = pop_size // 4

    for ev, t in zip(evs, ts):
        fiber_id = ev - pop_offset - 1
        joint_id = fiber_id // pop_size

        if joint_id == j_id:
            j_evs.append(fiber_id % pop_size)
            j_ts.append(t)

    torques = [2.0*ev / pop_size - 1.0 for ev in j_evs]
    vel = np.array(list(accumulate(torques))) / pop_size
    pos = np.array(list(accumulate(vel))) / pop_size

    return j_ts, torques, vel, pos


def cut_trial(evs, ts, trial_i, norm_times=False):
    trial_events = [
        (ev, t)
        for (ev, t) in zip(evs, ts)
        if trial_len*trial_i <= t < trial_len*(trial_i+1)
    ]
    trial_evs, trial_ts = zip(*trial_events)
    if norm_times:
        return np.array(trial_evs), np.array(trial_ts) - trial_len*trial_i
    else:
        return np.array(trial_evs), np.array(trial_ts)


def compute_trajectories(evs, ts, n, n_trials):
    trjs = []

    for i in range(n_trials):
        trial_evs, trial_ts = cut_trial(evs, ts, i)
        q_ts, qdd, qd, q = integrate_torque(trial_evs, trial_ts, 1, n, n)
        trjs.append([q_ts, q])

    return trjs


def get_final_x(trjs):
    xs = [q[-1] for (q_ts, q) in trjs]
    return np.mean(xs), np.std(xs)


def get_reference(n, n_trials):
    evs, ts = run_simulation(n, n_trials, 0.0)
    trjs = compute_trajectories(evs, ts, n, n_trials)

    ref_mean, ref_std = get_final_x(trjs)
    return ref_mean, ref_std


def get_error(evs, ts, n, ref_mean, n_trials=1):
    trjs = compute_trajectories(evs, ts, n, n_trials)

    mean, std = get_final_x(trjs)

    # ref_mean = 10°
    final_deg = mean * 10.0 / ref_mean
    std_deg = std * 10.0 / ref_mean

    error = final_deg - 10  # error in degrees
    return error, std_deg


def integrate_motor_io(evs, ts, io_plus, io_minus):
    pop_size = len(io_plus) + len(io_minus)
    torques = [
        (1.0 if ev in io_plus else -1.0)
        for ev in evs
    ]
    vel = np.array(list(accumulate(torques))) / pop_size
    pos = np.array(list(accumulate(vel))) / pop_size

    return pos[-1]


def simulate_closed_loop(n=400, prism=0.0, sensory_error=0.0):
    nest.ResetKernel()
    trajectories.save_file(prism, trial_len)

    planner = create_planner(n, prism)
    cortex = create_cortex(n)
    j1 = cortex.slice(n//4, n//2)

    sIO = create_sensory_io(n, sensory_error)
    sIOm = sIO.slice(0, n)
    sIOp = sIO.slice(n)

    mIO = create_motor_io(n, sensory_error)
    mIOm = mIO.slice(0, n)
    mIOp = mIO.slice(n)

    planner.connect(cortex)
    # Closing loop without cerebellum
    sIOp.connect(cortex, w=-1.0)
    sIOm.connect(cortex, w=+1.0)

    nest.Simulate(trial_len)

    s_io_evs, s_io_ts = sIO.get_events()
    print('sIO+ rate:', sIOp.get_rate())
    print('sIO- rate:', sIOm.get_rate())
    sIO.plot_spikes()

    print('mIO+ rate:', mIOp.get_rate())
    print('mIO- rate:', mIOm.get_rate())
    mIO.plot_spikes()

    m_io_evs, m_io_ts = mIO.get_events()
    m_io_pos = integrate_motor_io(m_io_evs, m_io_ts, mIOp.pop, mIOm.pop)
    print("Motor IO contribution to position:", m_io_pos)

    print('j1 rate:', j1.get_rate())
    cortex.plot_spikes()

    return cortex.get_events()


def test_learning():
    n = 400
    prism = 25.0

    ref_mean, ref_std = get_reference(n, 5)

    evs, ts = run_simulation(n, 1, prism)
    error, _ = get_error(evs, ts, n, ref_mean)

    evs, ts = simulate_closed_loop(n, prism, error)
    error_after, _ = get_error(evs, ts, n, ref_mean)

    print()
    print("Error before:", error)
    print("Error after:", error_after)


def plot_integration():
    prism = 0.0
    duration = 300
    q_in = np.array((10.0, -10.0, -90.0, 170.0))
    q_out = np.array((0.0, prism, 0.0,   0.0))

    q, qd, qdd = trajectories.jtraj(q_in, q_out, duration)
    fig, axs = plt.subplots(6, 4)

    for j in range(4):
        axs[0, j].plot(q[:, j])
        axs[1, j].plot(qd[:, j])
        axs[2, j].plot(qdd[:, j])

    n = 400

    evs, ts = run_simulation(n)

    for j in range(4):
        q_ts, qdd, qd, q = integrate_torque(evs, ts, j, n, n)
        axs[3, j].scatter(q_ts, qdd, marker='.')
        axs[4, j].plot(q_ts, qd)
        axs[5, j].plot(q_ts, q)

    plt.show()


def plot_trajectories(n_trials):
    n = 400

    evs, ts = run_simulation(n, n_trials)

    trjs = compute_trajectories(evs, ts, n, n_trials)

    # mean, std = get_final_x(trjs)
    # print("Mean:", mean)
    # print("Std:", std)

    for q_ts, q in trjs:
        plt.plot(q_ts, q)

    plt.show()


def plot_prism(n_trials, prism_values):
    n = 400

    ref_mean, ref_std = get_reference(n, n_trials)
    errors = []
    stds = []

    for prism in prism_values:
        evs, ts = run_simulation(n, n_trials, prism)
        error, std_deg = get_error(evs, ts, n, ref_mean, n_trials)

        errors.append(error)
        stds.append(std_deg)

    plt.errorbar(prism_values, errors, stds)
    plt.show()


def main():
    # plot_integration()
    # plot_trajectories(10)
    # plot_prism(4, range(-25, 30, 5))
    test_learning()


if __name__ == '__main__':
    main()
