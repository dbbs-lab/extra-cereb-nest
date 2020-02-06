from collections import namedtuple
import nest
from scaffold.core import from_hdf5
from world_populations import Planner, Cortex, \
        SensoryIO, MotorIO, DirectDCN, InverseDCN

Cerebellum = namedtuple("Cerebellum", "mf gr pc io dcn")

trial_len = 300

nest.Install("extracerebmodule")


def create_brain(prism):
    hdf5_file = 'scaffold_network_VOR.hdf5'
    scaffold_model = from_hdf5(hdf5_file)
    # scaffold_model.configuration.verbosity = 3

    S_MF = scaffold_model.get_entities_by_type("mossy_fibers")
    S_IO = scaffold_model.get_cells_by_type("io_cell")[:, 0]
    S_DCN = scaffold_model.get_cells_by_type("dcn_cell")[:, 0]

    adapter_forward = scaffold_model.create_adapter("FCN_2019")
    adapter_inverse = scaffold_model.create_adapter("FCN_2019")

    adapter_forward.enable_multi("forward")
    adapter_inverse.enable_multi("reverse")

    adapter_forward.prepare()
    adapter_inverse.prepare()

    f_MF = adapter_forward.get_nest_ids(S_MF)
    f_IO = adapter_forward.get_nest_ids(S_IO)
    f_DCN = adapter_forward.get_nest_ids(S_DCN)

    i_MF = adapter_inverse.get_nest_ids(S_MF)
    i_IO = adapter_inverse.get_nest_ids(S_IO)
    i_DCN = adapter_inverse.get_nest_ids(S_DCN)

    def connect(pop_1, pop_2, w=1.0):
        conn_dict = {"rule": "fixed_indegree", "indegree": 1}
        nest.Connect(pop_1, pop_2, conn_dict, {'weight': w})

    MF_number = len(f_MF)
    IO_number = len(f_IO)

    planner = Planner(MF_number, prism)
    cortex = Cortex(MF_number)

    planner.connect(cortex)

    # Forward model:
    # - motor input from the cortex (efference copy)
    # - sensory output to the cortex
    # - sensory error signal
    connect(cortex.pop, f_MF)  # Efference copy

    # TODO: use labels to select positive and negatives
    f_IO_view = SensoryIO(IO_number)
    connect(f_IO_view.pop, f_IO, 10.0)

    # TODO: use labels to select positive and negatives
    f_DCN_view = DirectDCN(f_DCN)
    connect(f_DCN_view.plus.pop, cortex.pop, 1.0)
    connect(f_DCN_view.minus.pop, cortex.pop, -1.0)

    # Inverse model;
    # - sensory input from planner
    # - motor output to world
    # - motor error signal
    connect(planner.pop, i_MF)  # Sensory input

    i_IO_view = MotorIO(IO_number)
    connect(i_IO_view.pop, i_IO, 10.0)


def main():
    prism = 25.0

    create_brain(prism)
    nest.Simulate(trial_len)


if __name__ == '__main__':
    main()
