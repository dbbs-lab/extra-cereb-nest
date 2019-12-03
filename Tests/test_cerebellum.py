from time import time
from contextlib import contextmanager
from collections import namedtuple
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

    for i in range(4):
        cereb_for.io.set_rate(sensory_error)
        cereb_inv.io.set_rate(sensory_error)

        nest.Simulate(trial_len)

        cortex.integrate(trial_i=i)
        mean, std = cortex.get_final_x()
        sensory_error, std_deg = world.get_error(ref_mean, mean, std)
        print("Closed loop error %d:" % i, sensory_error)

        cereb_inv.dcn.plus.integrate(trial_i=i)
        cereb_inv.dcn.minus.integrate(trial_i=i)

        x_dcnp, _ = cereb_inv.dcn.plus.get_final_x()
        x_dcnn, _ = cereb_inv.dcn.minus.get_final_x()

        print("Contributions from inverse DCN:")
        print("Positive:", x_dcnp)
        print("Negative:", x_dcnn)

    print('Forward DCN rate:', cereb_for.dcn.get_rate())
    cereb_for.dcn.plot_spikes()
    cereb_inv.dcn.plot_spikes()


def test_creation():
    define_models()

    with timing():
        create_forward_cerebellum()


def main():
    # test_creation()
    test_learning()


if __name__ == '__main__':
    main()
