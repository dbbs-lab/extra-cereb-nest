import nest
from contextlib import contextmanager
from time import time

from world_populations import Planner
from population_view import PopView


nest.Install("cerebmodule")
nest.Install("extracerebmodule")


MF_number = 100
GR_number = MF_number*100
PC_number = 72
IO_number = PC_number
DCN_number = PC_number//2


def create_cerebellum():
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

    MF = nest.Create("parrot_neuron", MF_number)
    GR = nest.Create("granular_neuron", GR_number)
    PC = nest.Create("purkinje_neuron", PC_number)
    IO = nest.Create("olivary_neuron", IO_number)
    DCN = nest.Create("nuclear_neuron", DCN_number)

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
                          "weight": Init_MFDCN, "delay": 10.0}

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

    return MF, GR, PC, IO, DCN


@contextmanager
def timing():
    t0 = time()
    yield None
    dt = time() - t0
    print("%.2fs" % dt)


def main():
    trial_len = 300

    print("Creating network")
    with timing():
        MF, GR, PC, IO, DCN = create_cerebellum()

    mf = PopView(MF)
    dcn = PopView(DCN)

    planner = Planner(MF_number, 0.0)
    planner.connect(mf)

    print("Simulating")
    with timing():
        nest.Simulate(trial_len)

    mf.plot_spikes()
    dcn.plot_spikes()


if __name__ == '__main__':
    main()
