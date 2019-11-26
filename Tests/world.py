import numpy as np
import nest
import trajectories

trial_len = 300


def select_trial_events(evs, ts, trial_i, norm_times=True):
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

    nest.Simulate(trial_len * n_trials)

    cortex.integrate(n_trials)
    mean, std = cortex.get_final_x()
    return mean, std


def get_reference(n, n_trials=1):
    prism = 0.0
    return run_open_loop(n, prism, n_trials)


def get_error(ref_mean, mean, std=0.0):
    # ref_mean = 10°
    final_deg = mean * 10.0 / ref_mean
    std_deg = std * 10.0 / ref_mean

    error = final_deg - 10  # error in degrees
    return error, std_deg
