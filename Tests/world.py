import numpy as np
import nest
import trajectories

trial_len = 300


def select_trial_events(evs, ts, trial_i, norm_times=False):
    evts = zip(evs, ts)
    trial_evts = [
        (ev, t) for (ev, t) in evts
        if trial_len * trial_i <= t < trial_len*(trial_i+1)
    ]
    if len(trial_evts) == 0:
        return [], []

    trial_evs, trial_ts = zip(*trial_evts)

    trial_evs = np.array(trial_evs)
    trial_ts = np.array(trial_ts)

    if norm_times:
        trial_ts -= trial_len * trial_i

    return trial_evs, trial_ts


def run_open_loop(n, prism, n_trials=1):
    # Ugly workaround to avoid circular imports
    from world_populations import Planner, Cortex

    nest.ResetKernel()
    trajectories.save_file(prism, trial_len)

    planner = Planner(n, prism)
    cortex = Cortex(n)

    planner.connect(cortex)

    xs = []

    for i in range(n_trials):
        nest.Simulate(trial_len)

        x = cortex.integrate(trial_i=i)
        xs.append(x)

    mean = np.mean(xs)
    std = np.std(xs)
    return mean, std


def get_reference(n, n_trials=1):
    prism = 0.0
    return run_open_loop(n, prism, n_trials)


def get_error(ref_mean, mean, std=0.0):
    raise Exception("Deprecated, use get_error_function instead")
    # ref_mean = 10°
    final_deg = mean * 10.0 / ref_mean
    std_deg = std * 10.0 / ref_mean

    error = final_deg - 10  # error in degrees
    return error, std_deg


def get_error_function(x_0, x_10):
    delta_10 = x_10 - x_0

    def error_function(x):
        return 10.0 * (x - x_0) / delta_10

    return error_function


def to_degrees(ref_mean, x):
    # ref_mean = 10°
    x_deg = x * 10.0 / ref_mean

    return x_deg


def calibrate(planner, cortex):
    # Get reference x
    xs = []

    nest.Simulate(trial_len)
    trial_offset = 1

    planner.set_prism(0.0)

    print("Comupting x_0")
    for i in range(10):
        nest.Simulate(trial_len)
        x = cortex.integrate(trial_i=i+trial_offset)
        xs.append(x)

    trial_offset += 10

    x_0 = np.mean(xs)
    print("x_0", x_0)

    planner.set_prism(10.0)
    nest.Simulate(trial_len)
    trial_offset += 1

    print("Comupting x_10")
    xs = []

    for i in range(10):
        nest.Simulate(trial_len)
        x = cortex.integrate(trial_i=i+trial_offset)
        xs.append(x)

    trial_offset += 10

    x_10 = np.mean(xs)
    print("x_10", x_10)

    get_error = get_error_function(x_0, x_10)

    # Get open loop error
    xs = []

    planner.set_prism(25.0)
    nest.Simulate(trial_len)
    trial_offset += 1

    for i in range(10):
        nest.Simulate(trial_len)

        x = cortex.integrate(trial_i=i+trial_offset)
        if i >= 1:
            xs.append(x)

    trial_offset += 10

    open_loop_error = get_error(np.mean(xs))

    return x_0, x_10, open_loop_error
