from collections import namedtuple
import numpy as np
import nest
from scaffold.core import from_hdf5
from population_view import PopView
from world_populations import Planner, Cortex, \
        SensoryIO, MotorIO, DirectDCN, InverseDCN

Cerebellum = namedtuple("Cerebellum", "mf gr pc io dcn")

trial_len = 300

nest.Install("extracerebmodule")


def create_brain(prism):
    hdf5_file = 'scaffold_network_VOR.hdf5'
    scaffold_model = from_hdf5(hdf5_file)
    # scaffold_model.configuration.verbosity = 3

    # Get scaffold model populations
    S_MF = scaffold_model.get_entities_by_type("mossy_fibers")
    S_IO = scaffold_model.get_cells_by_type("io_cell")[:, 0]
    S_DCN = scaffold_model.get_cells_by_type("dcn_cell")[:, 0]

    uz_pos = scaffold_model.labels["microzone-positive"]
    uz_neg = scaffold_model.labels["microzone-negative"]

    S_DCNp = np.intersect1d(S_DCN, uz_pos)
    S_DCNn = np.intersect1d(S_DCN, uz_neg)
    S_IOp = np.intersect1d(S_IO, uz_pos)
    S_IOn = np.intersect1d(S_IO, uz_neg)
    #

    # Prepare adapters
    adapter_forward = scaffold_model.create_adapter("FCN_2019")
    adapter_inverse = scaffold_model.create_adapter("FCN_2019")

    adapter_forward.enable_multi("forward")
    adapter_inverse.enable_multi("reverse")

    adapter_forward.prepare()
    adapter_inverse.prepare()
    #

    # Get NEST populations
    f_IOp = adapter_forward.get_nest_ids(S_IOp)
    f_IOn = adapter_forward.get_nest_ids(S_IOn)

    i_IOp = adapter_inverse.get_nest_ids(S_IOp)
    i_IOn = adapter_inverse.get_nest_ids(S_IOn)

    f_DCNp = adapter_forward.get_nest_ids(S_DCNp)
    f_DCNn = adapter_forward.get_nest_ids(S_DCNn)

    i_DCNp = adapter_forward.get_nest_ids(S_DCNp)
    i_DCNn = adapter_forward.get_nest_ids(S_DCNn)

    f_MF = adapter_forward.get_nest_ids(S_MF)
    i_MF = adapter_inverse.get_nest_ids(S_MF)
    #

    # Define population views
    MF_number = len(f_MF)
    IO_number = len(f_IOp) + len(f_IOn)

    planner_pv = Planner(MF_number, prism)
    cortex_pv = Cortex(MF_number)

    f_IO_pv = SensoryIO(IO_number)  # External from the scaffold,
    i_IO_pv = MotorIO(IO_number)    # to be connected after

    f_DCN_pv = DirectDCN(f_DCNp, f_DCNn)
    i_DCN_pv = InverseDCN(i_DCNp, i_DCNn)

    f_MF_pv = PopView(f_MF)
    i_MF_pv = PopView(i_MF)
    #

    # Connect populations
    def connect(pop_1, pop_2, w=1.0):
        conn_dict = {"rule": "one_to_one"}
        nest.Connect(pop_1, pop_2, conn_dict, {'weight': w})

    planner_pv.connect(cortex_pv)

    cortex_pv.connect(f_MF_pv)  # Efference copy
    planner_pv.connect(i_MF_pv)  # Sensory input

    connect(f_IO_pv.plus.pop, f_IOp, 10.0)
    connect(f_IO_pv.minus.pop, f_IOn, 10.0)
    connect(i_IO_pv.plus.pop, i_IOp, 10.0)
    connect(i_IO_pv.minus.pop, i_IOn, 10.0)

    conn_dict = {"rule": "fixed_indegree", "indegree": 1}
    nest.Connect(f_DCNp, cortex_pv.pop, conn_dict, {'weight': 1.0})
    nest.Connect(f_DCNn, cortex_pv.pop, conn_dict, {'weight': -1.0})

    # TODO: altre connesisoni?


def main():
    prism = 25.0

    create_brain(prism)
    nest.Simulate(trial_len)


if __name__ == '__main__':
    main()
