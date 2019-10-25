import numpy as np
import nest
from itertools import accumulate
import matplotlib.pyplot as plt

import trajectories

nest.Install("extracerebmodule")


trial_len = 300


def new_planner(n, prism=0.0):
    planner = nest.Create(
        "planner_neuron",
        n=n,
        params={
            "trial_length": trial_len,
            "target": 0.0,
            "prism_deviation": float(prism),
            "baseline_rate": 30.0,
            "gain_rate": 1.0,
            }
        )
    return planner


def new_cortex(n):
    cortex = nest.Create(
        "cortex_neuron",
        n=n,
        params={
            "trial_length": trial_len,
            "fibers_per_joint": n//4,
            "rbf_sdev": 15.0,
            "baseline_rate": 10.0,
            }
        )

    for i, neuron in enumerate(cortex):
        nest.SetStatus([neuron], {"joint_id": i // (n//4),
                                  "fiber_id": i % (n//4)})
    return cortex


def new_spike_detector(pop):
    spike_detector = nest.Create("spike_detector")
    nest.Connect(pop, spike_detector)
    return spike_detector


def get_spike_events(spike_detector):
    dSD = nest.GetStatus(spike_detector, keys="events")[0]
    evs = dSD["senders"]
    ts = dSD["times"]

    return evs, ts


def plot_spikes(evs, ts, pop=None):
    plt.scatter(ts, evs, marker='.')
    if pop:
        plt.ylim(min(pop), max(pop))
    plt.show()


def get_rate(spike_detector, pop):
    rate = nest.GetStatus(spike_detector, keys="n_events")[0] * 1e3 / trial_len
    rate /= len(pop)
    return rate


def run_simulation(n=400, n_trials=1, prism=0.0):
    nest.ResetKernel()
    trajectories.save_file(prism, trial_len)

    planner = new_planner(n, prism)
    cortex = new_cortex(n)

    nest.Connect(planner, cortex, 'one_to_one')

    spike_detector = new_spike_detector(cortex)

    nest.Simulate(trial_len * n_trials)

    evs, ts = get_spike_events(spike_detector)
    return evs, ts


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


def test_integration():
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


def test_trajectories(n_trials):
    n = 400

    evs, ts = run_simulation(n, n_trials)

    trjs = compute_trajectories(evs, ts, n, n_trials)

    # mean, std = get_final_x(trjs)
    # print("Mean:", mean)
    # print("Std:", std)

    for q_ts, q in trjs:
        plt.plot(q_ts, q)

    plt.show()


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


def test_prism(n_trials, prism_values):
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


def create_sensory_io(n, sensory_error):
    sensory_io = nest.Create(
        'poisson_generator',
        n=n*2,
        params={"rate": 0.0}
    )
    s_io_minus = sensory_io[:n]
    s_io_plus = sensory_io[n:]

    # TODO: tune scale factor
    s_io_rate = 1.0 * abs(sensory_error)

    if sensory_error > 0:
        nest.SetStatus(s_io_plus, {"rate": s_io_rate})
    else:
        nest.SetStatus(s_io_minus, {"rate": s_io_rate})

    return sensory_io, s_io_minus, s_io_plus


def create_motor_io(n, sensory_error):
    def make_template(upside=False):
        q_in = np.array((10.0, -10.0, -90.0, 170.0))
        q_out = np.array((0.0, 0.0, 0.0, 0.0))

        q, qd, qdd = trajectories.jtraj(q_in, q_out, trial_len)
        template = qdd[:, 1]
        template /= max(abs(template))

        if upside:
            template = -template

        template = np.clip(template, 0.0, np.max(template))
        return template

    def gen_spikes(template):
        # TODO: tune scale factor
        m_io_freqs = 0.001 * template * abs(sensory_error)

        m_io_ts = []
        for t, f in enumerate(m_io_freqs):
            n_spikes = np.random.poisson(f)
            if n_spikes:
                m_io_ts.append(float(t+1))

        return m_io_ts

    motor_io = nest.Create('spike_generator', n=n*2)
    m_io_minus = motor_io[:n]
    m_io_plus = motor_io[n:]

    template_p = make_template()
    template_m = make_template(upside=True)

    if sensory_error > 0:
        for cell in m_io_plus:
            nest.SetStatus([cell], {'spike_times': gen_spikes(template_p)})
        for cell in m_io_minus:
            nest.SetStatus([cell], {'spike_times': gen_spikes(template_m)})
    else:
        for cell in m_io_plus:
            nest.SetStatus([cell], {'spike_times': gen_spikes(template_m)})
        for cell in m_io_minus:
            nest.SetStatus([cell], {'spike_times': gen_spikes(template_p)})

    return motor_io, m_io_minus, m_io_plus


def simulate_closed_loop(n=400, prism=0.0, sensory_error=0.0):
    nest.ResetKernel()
    trajectories.save_file(prism, trial_len)

    planner = new_planner(n, prism)
    cortex = new_cortex(n)
    ctx_j1 = cortex[n//4:n//2]

    sensory_io, s_io_minus, s_io_plus = create_sensory_io(n, sensory_error)
    motor_io, m_io_minus, m_io_plus = create_motor_io(n, sensory_error)

    nest.Connect(planner, cortex, 'one_to_one')
    # Closing loop without cerebellum
    nest.Connect(s_io_plus, cortex, 'one_to_one', syn_spec={'weight': -1.0})
    nest.Connect(s_io_minus, cortex, 'one_to_one', syn_spec={'weight': 1.0})

    ctx_detector = new_spike_detector(cortex)
    j1_detector = new_spike_detector(ctx_j1)

    s_io_detector = new_spike_detector(sensory_io)
    s_io_p_detector = new_spike_detector(s_io_plus)
    s_io_m_detector = new_spike_detector(s_io_minus)

    m_io_detector = new_spike_detector(motor_io)
    m_io_p_detector = new_spike_detector(m_io_plus)
    m_io_m_detector = new_spike_detector(m_io_minus)

    nest.Simulate(trial_len)

    s_io_evs, s_io_ts = get_spike_events(s_io_detector)
    print('sIO+ rate:', get_rate(s_io_p_detector, s_io_plus))
    print('sIO- rate:', get_rate(s_io_m_detector, s_io_minus))
    plot_spikes(s_io_evs, s_io_ts, sensory_io)

    m_io_evs, m_io_ts = get_spike_events(m_io_detector)
    print('mIO+ rate:', get_rate(m_io_p_detector, m_io_plus))
    print('mIO- rate:', get_rate(m_io_m_detector, m_io_minus))
    plot_spikes(m_io_evs, m_io_ts, motor_io)

    ctx_evs, ctx_ts = get_spike_events(ctx_detector)
    print('j1 rate:', get_rate(j1_detector, ctx_j1))
    plot_spikes(ctx_evs, ctx_ts, cortex)

    return ctx_evs, ctx_ts


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


def main():
    # test_integration()
    # test_trajectories(10)
    test_prism(4, range(-25, 30, 5))
    # test_learning()


if __name__ == '__main__':
    main()
