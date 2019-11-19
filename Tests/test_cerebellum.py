from time import time
from contextlib import contextmanager
from collections import namedtuple
import nest

from world_functions import get_reference, run_open_loop, get_error
from world_populations import Planner, Cortex, SensoryIO, MotorIO

from cerebellum import MF_number, IO_number, define_models, create_cerebellum
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


def create_brain(sensory_error):
    prism = 10.0
    n = MF_number

    trajectories.save_file(prism, trial_len)

    planner = Planner(n, prism)
    cortex = Cortex(n)
    # j1 = cortex.slice(n//4, n//2)

    sIO = SensoryIO(IO_number // 2, sensory_error)
    mIO = MotorIO(IO_number // 2, sensory_error)

    planner.connect(cortex)

    # Direct model
    cereb_dir = create_cerebellum(sIO)
    planner.connect(cereb_dir.mf)  # Sensory input

    # Inverse model
    cereb_inv = create_cerebellum(mIO)
    cortex.connect(cereb_inv.mf)  # Efference copy

    return Brain(planner, cortex, cereb_dir, cereb_inv), sIO, mIO


def test_learning():
    n = MF_number
    prism = 25.0

    # Get open loop error
    ref_mean, ref_std = get_reference(n)

    mean, std = run_open_loop(n, prism)
    sensory_error, std_deg = get_error(ref_mean, mean, std)

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

    print("sIO len:", len(sIO.pop))
    # Direct model
    cereb_dir = create_cerebellum(sIO)
    planner.connect(cereb_dir.mf)  # Sensory input

    # Inverse model
    cereb_inv = create_cerebellum(mIO)
    cortex.connect(cereb_inv.mf)  # Efference copy

    nest.Simulate(trial_len)

    print('sIO+ rate:', sIO.plus.get_rate())
    print('sIO- rate:', sIO.minus.get_rate())
    sIO.plot_spikes()

    print('mIO+ rate:', mIO.plus.get_rate())
    print('mIO- rate:', mIO.minus.get_rate())
    mIO.plot_spikes()


def main():
    test_learning()


if __name__ == '__main__':
    main()
