import numpy as np
import nest
import matplotlib.pyplot as plt

from world_populations import Planner, Cortex, SensoryIO, MotorIO
from world import get_reference, get_error
import trajectories

nest.Install("extracerebmodule")


trial_len = 300


def run_simulation(n=400, n_trials=1, prism=0.0):
    nest.ResetKernel()
    trajectories.save_file(prism, trial_len)

    planner = Planner(n, prism)
    cortex = Cortex(n)

    planner.connect(cortex)

    nest.Simulate(trial_len * n_trials)

    cortex.integrate(n_trials)
    return cortex


def simulate_closed_loop(n=400, prism=0.0, sensory_error=0.0):
    nest.ResetKernel()
    trajectories.save_file(prism, trial_len)

    planner = Planner(n, prism)
    cortex = Cortex(n)
    j1 = cortex.slice(n//4, n//2)

    sIO = SensoryIO(n, sensory_error)
    sIOm = sIO.minus
    sIOp = sIO.plus

    mIO = MotorIO(n, sensory_error)
    mIOm = mIO.minus
    mIOp = mIO.plus

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

    mIO.integrate()
    mIO_pos, _ = mIO.get_final_x()
    print("Motor IO contribution to position:", mIO_pos)

    print('j1 rate:', j1.get_rate())
    cortex.plot_spikes()

    cortex.integrate()
    mean, std = cortex.get_final_x()

    return mean, std


def test_learning():
    n = 400
    prism = 25.0

    ref_mean, ref_std = get_reference(n, 5)

    cortex = run_simulation(n, 1, prism)

    mean, std = cortex.get_final_x()
    error, _ = get_error(ref_mean, mean, std)

    mean, std = simulate_closed_loop(n, prism, error)
    error_after, _ = get_error(ref_mean, mean, std)

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

    cortex = run_simulation(n)

    for j in range(4):
        q_ts, qdd, qd, q = cortex.joints[j].states[0]
        axs[3, j].scatter(q_ts, qdd, marker='.')
        axs[4, j].plot(q_ts, qd)
        axs[5, j].plot(q_ts, q)

    plt.show()


def plot_integrate_cortex():
    n = 400
    prism = 0.0
    n_trials = 5

    nest.ResetKernel()
    trajectories.save_file(prism, trial_len)

    planner = Planner(n, prism)
    cortex = Cortex(n)
    planner.connect(cortex)

    nest.Simulate(trial_len * n_trials)
    cortex.integrate(n_trials, plot=True)


def plot_trajectories(n_trials):
    n = 400

    cortex = run_simulation(n, n_trials)

    for i in range(n_trials):
        q_ts, _, _, q = cortex.joints[1].states[i]
        plt.plot(q_ts, q)

    plt.show()


def plot_prism(n_trials, prism_values):
    n = 300

    ref_mean, ref_std = get_reference(n, n_trials)
    errors = []
    stds = []

    for prism in prism_values:
        cortex = run_simulation(n, n_trials, prism)
        mean, std = cortex.get_final_x()

        error, std_deg = get_error(ref_mean, mean, std)
        errors.append(error)
        stds.append(std_deg)

    plt.errorbar(prism_values, errors, stds)
    plt.show()


def plot_mIO_correction(n_trials, prism_values):
    n = 400

    ref_mean, ref_std = get_reference(n, n_trials)

    corrections = []
    stds = []

    for prism in prism_values:
        cortex = run_simulation(n, n_trials, prism)

        mean, std = cortex.get_final_x()
        error, _ = get_error(ref_mean, mean, std)
        print("Error:", error)

        nest.ResetKernel()
        trajectories.save_file(prism, trial_len)

        planner = Planner(n, prism)
        cortex = Cortex(n)

        mIO = MotorIO(n, error)
        mIOm = mIO.minus
        mIOp = mIO.plus

        planner.connect(cortex)

        nest.Simulate(trial_len * n_trials)

        print('mIO+ rate:', mIOp.get_rate())
        print('mIO- rate:', mIOm.get_rate())

        mIO.integrate(n_trials)
        mIO_pos, mIO_std = mIO.get_final_x()
        print("Motor IO contribution to position:", mIO_pos)

        corrections.append(mIO_pos)
        stds.append(mIO_std)

    plt.errorbar(prism_values, corrections, stds)
    plt.show()


def main():
    # plot_integration()
    # plot_integrate_cortex()
    # plot_trajectories(10)
    # plot_prism(4, range(-25, 30, 5))
    test_learning()
    # plot_mIO_correction(5, range(-10, 25, 5))


if __name__ == '__main__':
    main()
