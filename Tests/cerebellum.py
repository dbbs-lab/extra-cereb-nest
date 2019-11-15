import nest
from contextlib import contextmanager
from time import time
import matplotlib.pyplot as plt
from collections import namedtuple

from world_functions import get_reference, run_open_loop, get_error
from world_populations import Planner, Cortex, SensoryIO, MotorIO
from population_view import PopView
import trajectories


nest.Install("cerebmodule")
nest.Install("extracerebmodule")


Cerebellum = namedtuple("Cerebellum", "mf gr pc io dcn")
Brain = namedtuple("Brain", "planner cortex forward inverse")


trial_len = 300

MF_number = 300
GR_number = MF_number*100
PC_number = 72
IO_number = PC_number
DCN_number = PC_number//2


def define_models():
    # Neuron models definitions
    nest.CopyModel('iaf_cond_exp', 'granular_neuron')
    nest.CopyModel('iaf_cond_exp', 'purkinje_neuron')
    nest.CopyModel('iaf_cond_exp', 'olivary_neuron')
    nest.CopyModel('iaf_cond_exp', 'nuclear_neuron')

    nest.SetDefaults('granular_neuron', {'t_ref': 1.0,
                                         'C_m': 2.0,
                                         'V_th': -40.0,
                                         'V_reset': -70.0,
                                         'g_L': 0.2,
                                         'tau_syn_ex': 0.5,
                                         'tau_syn_in': 10.0})

    nest.SetDefaults('purkinje_neuron', {'t_ref': 2.0,
                                         'C_m': 400.0,
                                         'V_th': -52.0,
                                         'V_reset': -70.0,
                                         'g_L': 16.0,
                                         'tau_syn_ex': 0.5,
                                         'tau_syn_in': 1.6})

    nest.SetDefaults('olivary_neuron', {'t_ref': 1.0,
                                        'C_m': 2.0,
                                        'V_th': -40.0,
                                        'V_reset': -70.0,
                                        'g_L': 0.2,
                                        'tau_syn_ex': 0.5,
                                        'tau_syn_in': 10.0})

    nest.SetDefaults('nuclear_neuron', {'t_ref': 1.0,
                                        'C_m': 2.0,
                                        'V_th': -40.0,
                                        'V_reset': -70.0,
                                        'g_L': 0.2,
                                        'tau_syn_ex': 0.5,
                                        'tau_syn_in': 10.0})


@contextmanager
def timing():
    t0 = time()
    yield None
    dt = time() - t0
    print("%.2fs" % dt)


def create_cerebellum(inferior_olive):
    PLAST1 = True  # PF-PC ex
    PLAST2 = True  # MF-DCN ex
    PLAST3 = True  # PC-DCN

    LTP1 = 0.1
    LTD1 = -1.0
    LTP2 = 1e-5
    LTD2 = -1e-6
    LTP3 = 1e-7
    LTD3 = 1e-6

    Init_PFPC = 1.0
    Init_MFDCN = 0.4
    Init_PCDCN = -1.0

    MF = nest.Create("parrot_neuron", MF_number)
    GR = nest.Create("granular_neuron", GR_number)
    PC = nest.Create("purkinje_neuron", PC_number)
    DCN = nest.Create("nuclear_neuron", DCN_number)
    if inferior_olive:
        IO = inferior_olive.pop
    else:
        IO = nest.Create("olivary_neuron", IO_number)

    # Weights recorder
    # rec_params = {
    #     "to_memory": False,
    #     "to_file":    True,
    #     "label":     "PFPC",
    #     "senders":    GR,
    #     "targets":    PC,
    #     "precision":  8
    # }
    # w_PFPC = nest.Create('weight_recorder', params=rec_params)

    # MFGR
    MFGR_conn_dict = {'rule': 'fixed_indegree',
                      'indegree': 4,
                      "multapses": False}

    MFGR_syn_dict = {"model": "static_synapse",
                     "weight": {'distribution': 'uniform',
                                # -> 0.75 GR fire at 7 Hz
                                'low': 1.0, 'high': 2.0},
                     "delay": 1.0}

    nest.Connect(MF, GR, MFGR_conn_dict, MFGR_syn_dict)
    #

    if PLAST1:
        # Volume transmitter
        vt1 = nest.Create("volume_transmitter_alberto", PC_number)
        for i, vt_i in enumerate(vt1):
            nest.SetStatus([vt_i], {"vt_num": i})

        vt1_syn_dict = {"model": "static_synapse", "weight": 1.0, "delay": 1.0}
        print("IO_number:", IO_number)
        print("Actual len of IO:", len(IO))
        nest.Connect(IO, vt1, 'one_to_one', vt1_syn_dict)
        #

        nest.SetDefaults('stdp_synapse_sinexp',
                         {"A_minus":   LTD1,
                          "A_plus":    LTP1,
                          "Wmin":      0.0,
                          "Wmax":      4.0,
                          "vt":        vt1[0],
                          # "weight_recorder": w_PFPC[0]
                          })

        PFPC_conn_dict = {'rule': 'fixed_indegree',
                          'indegree': int(0.8*GR_number),
                          "multapses": False}

        PFPC_syn_dict = {"model":  'stdp_synapse_sinexp',
                         "weight": Init_PFPC, "delay":  1.0}

        for i, PCi in enumerate(PC):
            nest.Connect(GR, [PCi], PFPC_conn_dict, PFPC_syn_dict)
            A = nest.GetConnections(GR, [PCi])
            nest.SetStatus(A, {'vt_num': i})
    else:
        PFPC_conn_dict = {'rule': 'fixed_indegree',
                          'indegree': int(0.8*GR_number),
                          "multapses": False}

        PFPC_syn_dict = {"model":  "static_synapse",
                         "weight": Init_PFPC, "delay":  1.0}

        nest.Connect(GR, PC, PFPC_conn_dict, PFPC_syn_dict)

    if PLAST2:
        vt2 = nest.Create("volume_transmitter_alberto", DCN_number)
        for i, vt_i in enumerate(vt2):
            nest.SetStatus([vt_i], {"vt_num": i})

        # MF-DCN excitatory plastic connections
        # every MF is connected with every DCN
        nest.SetDefaults('stdp_synapse_cosexp',
                         {"A_minus":   LTD2,
                          "A_plus":    LTP2,
                          "Wmin":      0.0,
                          "Wmax":      0.25,
                          "vt":        vt2[0]})

        MFDCN_syn_dict = {"model": 'stdp_synapse_cosexp',
                          "weight": Init_MFDCN, "delay": 1.0}

        for i, DCNi in enumerate(DCN):
            nest.Connect(MF, [DCNi], 'all_to_all', MFDCN_syn_dict)
            A = nest.GetConnections(MF, [DCNi])
            nest.SetStatus(A, {'vt_num': i})

        # PC-DCN inhibitory plastic connections
        # each DCN receives 2 connections from 2 contiguous PC
        vt2_syn_dict = {"model": "static_synapse", "weight": 1.0, "delay": 1.0}

        for P in range(PC_number):
            count_DCN = P // 2
            nest.Connect([PC[P]], [vt2[count_DCN]], 'one_to_one', vt2_syn_dict)

    else:
        MFDCN_syn_dict = {"model":  "static_synapse",
                          "weight": Init_MFDCN, "delay":  10.0}

        nest.Connect(MF, DCN, 'all_to_all', MFDCN_syn_dict)

    if PLAST3:
        nest.SetDefaults('stdp_synapse', {"tau_plus": 30.0,
                                          "lambda": LTP3,
                                          "alpha": LTD3/LTP3,
                                          "mu_plus": 0.0,   # Additive STDP
                                          "mu_minus": 0.0,  # Additive STDP
                                          "Wmax": -0.5,
                                          "weight": Init_PCDCN,
                                          "delay": 1.0})

        PCDCN_syn_dict = {"model": "stdp_synapse"}
    else:
        PCDCN_syn_dict = {"model": "static_synapse",
                          "weight": Init_PCDCN, "delay": 1.0}

    # PC-DCN inhibitory plastic connections
    # each DCN receives 2 connections from 2 contiguous PC
    for P in range(PC_number):
        count_DCN = P // 2
        nest.Connect([PC[P]], [DCN[count_DCN]], 'one_to_one', PCDCN_syn_dict)

    pop_views = [PopView(pop) for pop in (MF, GR, PC, IO, DCN)]

    if inferior_olive:
        pop_views[3] = inferior_olive

    cereb = Cerebellum(*pop_views)
    return cereb


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
