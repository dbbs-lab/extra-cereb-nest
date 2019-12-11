from time import time
from contextlib import contextmanager
from collections import namedtuple
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


trial_len = 300


@contextmanager
def timing():
    t0 = time()
    yield None
    dt = time() - t0
    print("%.2fs" % dt)


def create_brain(prism):
    trajectories.save_file(prism, trial_len)

    planner = Planner(MF_number, prism)
    cortex = Cortex(MF_number)

    define_models()

    planner.connect(cortex)

    # Forward model:
    # - motor input from the cortex (efference copy)
    # - sensory output to the cortex
    # - sensory error signal
    cereb_for = create_forward_cerebellum()
    cortex.connect(cereb_for.mf)  # Efference copy

    fDCN = cereb_for.dcn
    conn_dict = {"rule": "fixed_indegree", "indegree": 1}
    nest.Connect(fDCN.plus.pop, cortex.pop, conn_dict, {'weight': 1.0})
    nest.Connect(fDCN.minus.pop, cortex.pop, conn_dict, {'weight': -1.0})

    # Inverse model;
    # - sensory input from planner
    # - motor output to world
    # - motor error signal
    cereb_inv = create_inverse_cerebellum()
    planner.connect(cereb_inv.mf)  # Sensory input

    return cortex, cereb_for, cereb_inv


def test_learning():
    prism = 25.0

    # Get open loop error
    ref_mean, ref_std = world.get_reference(MF_number)

    mean, std = world.run_open_loop(MF_number, prism)
    sensory_error, std_deg = world.get_error(ref_mean, mean, std)

    print("Open loop error:", sensory_error)
    #

    nest.ResetKernel()
    cortex, cereb_for, cereb_inv = create_brain(prism)

    for i in range(1):
        cereb_for.io.set_rate(sensory_error)
        cereb_inv.io.set_rate(sensory_error)

        nest.Simulate(trial_len)

        cortex.integrate(trial_i=i)
        x_cortex, std = cortex.get_final_x()

        cereb_inv.dcn.plus.integrate(trial_i=i)
        cereb_inv.dcn.minus.integrate(trial_i=i)

        x_dcnp, _ = cereb_inv.dcn.plus.get_final_x()
        x_dcnn, _ = cereb_inv.dcn.minus.get_final_x()

        x_sum = x_cortex + x_dcnp - x_dcnn

        sensory_error, std_deg = world.get_error(ref_mean, x_sum, std)
        print("Closed loop error %d:" % i, sensory_error)

        print("Contributions from inverse DCN:")
        print("Positive:", x_dcnp)
        print("Negative:", x_dcnn)

    print('Forward DCN rate:', cereb_for.dcn.get_rate())

    fig, axs = plt.subplots(4)
    cereb_for.io.plot_spikes('Forward IO', axs[0])
    cereb_for.dcn.plot_spikes('Forward DCN', axs[1])

    cereb_inv.io.plot_spikes('Inverse IO', axs[2])
    cereb_inv.dcn.plot_spikes('Inverse DCN', axs[3])

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


def test_creation():
    define_models()

    with timing():
        create_forward_cerebellum()


def main():
    # test_creation()
    # test_learning()
    test_initial_rates()


if __name__ == '__main__':
    main()
