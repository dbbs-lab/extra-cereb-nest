from time import time
from datetime import datetime
from contextlib import contextmanager
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import nest
import world
from world_populations import Planner, Cortex

from cerebellum import MF_number, define_models, \
        create_forward_cerebellum, create_inverse_cerebellum
import trajectories


Brain = namedtuple("Brain", "planner cortex forward inverse")

nest.Install("cerebmodule")
nest.Install("extracerebmodule")


csv_path = './csv/'


trial_len = 300


def save_csv(name, data):
    now = datetime.now()
    path = csv_path + now.strftime("%d-%m_%H:%M_") + name
    np.savetxt(path, data, delimiter=",")


@contextmanager
def timing(label=""):
    t0 = time()
    yield None
    dt = time() - t0
    print(label + " %.2fs" % dt)


def create_brain(prism):
    trajectories.save_file(prism, trial_len)

    define_models()
    cereb_inv = create_inverse_cerebellum()
    cereb_for = create_forward_cerebellum()
    # cereb_foo = create_forward_cerebellum()

    planner = Planner(MF_number, prism)
    cortex = Cortex(MF_number)

    planner.connect(cortex)

    # Forward model:
    # - motor input from the cortex (efference copy)
    # - sensory output to the cortex
    # - sensory error signal
    cortex.connect(cereb_for.mf)  # Efference copy

    fDCN = cereb_for.dcn
    conn_dict = {"rule": "fixed_indegree", "indegree": 1}
    nest.Connect(fDCN.plus.pop, cortex.pop, conn_dict, {'weight': 1.0})
    nest.Connect(fDCN.minus.pop, cortex.pop, conn_dict, {'weight': -1.0})

    # Inverse model;
    # - sensory input from planner
    # - motor output to world
    # - motor error signal
    planner.connect(cereb_inv.mf)  # Sensory input

    return cortex, cereb_for, cereb_inv


def get_weights(pop1, pop2):
    conns = nest.GetConnections(pop1[::50], pop2[::50])
    weights = nest.GetStatus(conns, "weight")
    return weights


def test_learning():
    FORWARD = True
    INVERSE = True
    prism = 20.0
    # prism = 0.0
    n_trials = 4

    error_history = []

    # Get reference x
    nest.ResetKernel()
    cortex, _, _ = create_brain(0.0)
    xs = []

    for i in range(10):
        nest.Simulate(trial_len)

        x = cortex.integrate(trial_i=i)
        if i >= 5:
            xs.append(x)

    ref_x = np.mean(xs)
    #

    # Get open loop error
    mean, std = world.run_open_loop(MF_number, prism)
    sensory_error, std_deg = world.get_error(ref_x, mean, std)

    print("Open loop error:", sensory_error)
    #

    weights_for = []
    weights_inv = []

    nest.ResetKernel()
    cortex, cereb_for, cereb_inv = create_brain(prism)

    for i in range(n_trials):
        if FORWARD:
            cereb_for.io.set_rate(sensory_error)
        if INVERSE:
            cereb_inv.io.set_rate(sensory_error, trial_i=i)

        print("Simulating")
        with timing("Trial time"):
            nest.Simulate(trial_len)
        print()
        print("Trial ", i+1)
        print()

        x_cortex = cortex.integrate(trial_i=i)

        if INVERSE:
            cereb_inv.dcn.plus.integrate(trial_i=i)
            cereb_inv.dcn.minus.integrate(trial_i=i)

            x_dcnp, _ = cereb_inv.dcn.plus.get_final_x()
            x_dcnn, _ = cereb_inv.dcn.minus.get_final_x()

            print("Contributions from inverse DCN:")
            print("Positive:", x_dcnp)
            print("Negative:", x_dcnn)

            x_sum = x_cortex + x_dcnp - x_dcnn
        else:
            x_sum = x_cortex

        sensory_error, std_deg = world.get_error(ref_x, x_sum, 0.0)
        error_history.append(sensory_error)
        print("Closed loop error %d:" % i, sensory_error)

        if FORWARD:
            print()
            print("Forward IO: %.1f" % cereb_for.io.get_per_trial_rate())
            print("Forward MF: %.1f" % cereb_for.mf.get_per_trial_rate())
            print("Forward GR: %.1f" % cereb_for.gr.get_per_trial_rate())
            print("Forward PC: %.1f" % cereb_for.pc.get_per_trial_rate())
            print("Forward DCN: %.1f" % cereb_for.dcn.get_per_trial_rate())

            weights = get_weights(cereb_for.gr.pop, cereb_for.pc.pop)
            weights_for.append(weights)
            print("Forward PFPC weights:", min(weights), "to", max(weights))

        if INVERSE:
            print()
            print("Inverse IO: %.1f" % cereb_inv.io.get_per_trial_rate())
            print("Inverse MF: %.1f" % cereb_inv.mf.get_per_trial_rate())
            print("Inverse GR: %.1f" % cereb_inv.gr.get_per_trial_rate())
            print("Inverse PC: %.1f" % cereb_inv.pc.get_per_trial_rate())
            print("Inverse DCN: %.1f" % cereb_inv.dcn.get_per_trial_rate())

            weights = get_weights(cereb_inv.gr.pop, cereb_inv.pc.pop)
            weights_inv.append(weights)
            print("Inverse PFPC weights:", min(weights), "to", max(weights))

    save_csv("weights_for.csv", np.transpose(weights_for))
    save_csv("weights_inv.csv", np.transpose(weights_inv))

    fig, axs = plt.subplots(1, 2)
    axs[0].matshow(np.transpose(weights_for), aspect='auto')
    axs[1].matshow(np.transpose(weights_inv), aspect='auto')
    plt.show()

    fig, axs = plt.subplots(5)
    if FORWARD:
        #   cereb_for.mf.plot_spikes('f MF', axs[0])
        #   cereb_for.io.plot_spikes('f IO', axs[1])
        #   cereb_for.pc.plot_spikes('f PC', axs[2])
        #   cereb_for.dcn.plot_spikes('f DCN', axs[3])

        cereb_for.mf.plot_per_trial_rates('MF', axs[0])
        cereb_for.io.plot_per_trial_rates('IO', axs[1])
        cereb_for.pc.plot_per_trial_rates('PC', axs[2])
        cereb_for.dcn.plot_per_trial_rates('DCN', axs[3])

        # conns = nest.GetConnections(cereb_for.gr.pop, cereb_for.pc.pop)
        # weights = nest.GetStatus(conns, "weight")
        # print("Minimum weight at PFPC:", min(weights))
        # axs[4].plot(weights)

        print()
        print("Forward MF rate:", cereb_for.mf.get_rate(n_trials))
        print("Forward GR rate:", cereb_for.gr.get_rate(n_trials))
        print("Forward PC rate:", cereb_for.pc.get_rate(n_trials))
        print("Forward DCN rate:", cereb_for.dcn.get_rate(n_trials))

    if INVERSE:
        #   cereb_inv.mf.plot_spikes('i MF', axs[0])
        #   cereb_inv.io.plot_spikes('i IO', axs[1])
        #   cereb_inv.pc.plot_spikes('i PC', axs[2])
        #   cereb_inv.dcn.plot_spikes('i DCN', axs[3])

        cereb_inv.mf.plot_per_trial_rates('MF', axs[0])
        cereb_inv.io.plot_per_trial_rates('IO', axs[1])
        cereb_inv.pc.plot_per_trial_rates('PC', axs[2])
        cereb_inv.dcn.plot_per_trial_rates('DCN', axs[3])

        # conns = nest.GetConnections(cereb_inv.gr.pop, cereb_inv.pc.pop)
        # weights = nest.GetStatus(conns, "weight")
        # print("Minimum weight at PFPC:", min(weights))
        # axs[4].plot(weights)

        print()
        print("Inverse MF rate:", cereb_inv.mf.get_rate(n_trials))
        print("Inverse GR rate:", cereb_inv.gr.get_rate(n_trials))
        print("Inverse PC rate:", cereb_inv.pc.get_rate(n_trials))
        print("Inverse DCN rate:", cereb_inv.dcn.get_rate(n_trials))

    axs[4].set_ylabel('Error')
    axs[4].plot(error_history)
    plt.show()


def test_initial_rates():
    prism = 25.0

    nest.ResetKernel()
    cortex, cereb_for, cereb_inv = create_brain(prism)

    nest.Simulate(trial_len)

    print()
    print("Forward MF rate:", cereb_for.mf.get_rate())
    print("Inverse MF rate:", cereb_inv.mf.get_rate())

    print()
    print("Forward GR rate:", cereb_for.gr.get_rate())
    print("Inverse GR rate:", cereb_inv.gr.get_rate())

    print()
    print("Forward PC rate:", cereb_for.pc.get_rate())
    print("Inverse PC rate:", cereb_inv.pc.get_rate())

    print()
    print("Forward DCN rate:", cereb_for.dcn.get_rate())
    print("Inverse DCN rate:", cereb_inv.dcn.get_rate())

    # fig, axs = plt.subplots(4)
    # cereb_for.mf.plot_spikes('Forward MF', axs[0])
    # cereb_for.gr.plot_spikes('Forward GR', axs[1])
    # cereb_for.pc.plot_spikes('Forward PC', axs[2])
    # cereb_for.dcn.plot_spikes('Forward DCN', axs[3])
    # plt.show()

    # fig, axs = plt.subplots(4)
    # cereb_inv.mf.plot_spikes('Inverse MF', axs[0])
    # cereb_inv.gr.plot_spikes('Inverse GR', axs[1])
    # cereb_inv.pc.plot_spikes('Inverse PC', axs[2])
    # cereb_inv.dcn.plot_spikes('Inverse DCN', axs[3])
    # plt.show()


def test_error():
    prism = 0.0
    n_trials = 10

    error_history = []

    # Get reference x
    nest.ResetKernel()
    cortex, _, _ = create_brain(0.0)
    xs = []

    for i in range(10):
        nest.Simulate(trial_len)

        x = cortex.integrate(trial_i=i)
        if i >= 5:
            xs.append(x)

    ref_x = np.mean(xs)
    ref_std = np.std(xs)
    #

    # Get open loop error
    mean, std = world.run_open_loop(MF_number, prism)
    print("Reference mean:", ref_x, "std:", ref_std)

    sensory_error, std_deg = world.get_error(ref_x, mean, std)
    print("Open loop error:", sensory_error)
    #

    nest.ResetKernel()
    cortex, _, _ = create_brain(prism)

    for i in range(n_trials):
        print("Simulating")
        with timing("Trial time"):
            nest.Simulate(trial_len)

        print()
        print("Trial ", i+1)

        x_cortex = cortex.integrate(trial_i=i)

        sensory_error, std_deg = world.get_error(ref_x, x_cortex, 0.0)
        error_history.append(sensory_error)
        print("X from cortex:", x_cortex)
        print("Error:", sensory_error)

    plt.plot(error_history)
    plt.show()


def test_creation():
    define_models()

    with timing():
        create_forward_cerebellum()


def main():
    # test_creation()
    # test_initial_rates()
    test_learning()
    # test_error()


if __name__ == '__main__':
    main()
