from time import time
from contextlib import contextmanager
from collections import namedtuple
import nest
import world
from world_populations import Planner, Cortex, SensoryIO, MotorIO

from cerebellum import MF_number, IO_number, DCN_number, \
        define_models, create_cerebellum
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


def test_learning():
    n = MF_number
    prism = 25.0

    # Get open loop error
    ref_mean, ref_std = world.get_reference(n)

    mean, std = world.run_open_loop(n, prism)
    sensory_error, std_deg = world.get_error(ref_mean, mean, std)

    print("Open loop error:", sensory_error)
    #

    nest.ResetKernel()
    trajectories.save_file(prism, trial_len)

    planner = Planner(n, prism)
    cortex = Cortex(n)

    sIO = SensoryIO(IO_number // 2, sensory_error)
    mIO = MotorIO(IO_number // 2, sensory_error)

    planner.connect(cortex)

    # Closing loop without cerebellum
    # sIOp.connect(cortex, w=-1.0)
    # sIOm.connect(cortex, w=+1.0)

    define_models()

    # Forward model:
    # - motor input from the cortex (efference copy)
    # - sensory output to the cortex
    # - sensory error signal
    cereb_for = create_cerebellum(sIO)
    cortex.connect(cereb_for.mf)  # Efference copy

    fDCNp = cereb_for.dcn.pop[:DCN_number//2]
    fDCNn = cereb_for.dcn.pop[DCN_number//2:]
    conn_dict = {"rule": "fixed_indegree", "indegree": 1}
    nest.Connect(fDCNp, cortex.pop, conn_dict, {'weight': 1.0})
    nest.Connect(fDCNn, cortex.pop, conn_dict, {'weight': -1.0})

    # Inverse model;
    # - sensory input from planner
    # - motor output to world
    # - motor error signal
    cereb_inv = create_cerebellum(mIO)
    planner.connect(cereb_inv.mf)  # Sensory input

    for i in range(4):
        nest.Simulate(trial_len)

        cortex.integrate()
        mean, std = cortex.get_final_x()
        sensory_error, std_deg = world.get_error(ref_mean, mean, std)
        print("Closed loop error %d:" % i, sensory_error)

        sIO.set_rate(sensory_error)
        mIO.set_rate(sensory_error)

    # print('Forward DCN rate:', cereb_for.dcn.get_rate())
    # cereb_for.dcn.plot_spikes()
    # cereb_inv.dcn.plot_spikes()


def main():
    test_learning()


if __name__ == '__main__':
    main()
