import numpy as np
from collections import namedtuple
import nest
from population_view import PopView
from world_populations import SensoryIO, MotorIO, DirectDCN, InverseDCN

Cerebellum = namedtuple("Cerebellum", "mf gr pc io dcn")

cereb_index = 0

MF_number = 360*2
GR_number = MF_number
PC_number = 269*2
IO_number = 10*2
DCN_number = 22*2


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


def create_cerebellum():
    # PLAST1 PF-PC ex
    # PLAST2 MF-DCN ex
    # PLAST3 PC-DCN
    PLAST1 = PLAST2 = PLAST3 = False
    PLAST1 = True

    LTP1 = 0.2
    LTD1 = -5.0
    LTP2 = 1e-5
    LTD2 = -1e-6
    LTP3 = 1e-7
    LTD3 = 1e-6

    Init_PFPC = 16.0
    Init_MFDCN = 0.01
    Init_PCDCN = -0.0005
    # Init_PCDCN = -0.001

    MF = nest.Create("parrot_neuron", MF_number)
    GR = nest.Create("granular_neuron", GR_number)
    PC = nest.Create("purkinje_neuron", PC_number)
    DCN = nest.Create("nuclear_neuron", DCN_number)
    IO = nest.Create("olivary_neuron", IO_number)
    # IO = nest.Create("parrot_neuron", IO_number)

    # Weights recorder
    rec_params = {
        "to_memory": True,
        "to_file":   False,
        "label":     "PFPC",
        "senders":   GR,
        "targets":   PC,
        "precision": 8
    }
    w_PFPC = nest.Create('weight_recorder', params=rec_params)

    # MFGR
    MFGR_conn_dict = {'rule': 'fixed_indegree',
                      'indegree': 4,
                      "multapses": False}

    MFGR_syn_dict = {"model": "static_synapse",
                     "weight": 0.6,
                     "delay": 1.0}

    nest.Connect(MF, GR, MFGR_conn_dict, MFGR_syn_dict)
    #

    if PLAST1:
        # Volume transmitter
        vt1 = nest.Create("volume_transmitter_alberto", PC_number)
        for i, vt_i in enumerate(vt1):
            nest.SetStatus([vt_i], {"vt_num": i})

        vt1_syn_dict = {"model": "static_synapse", "weight": 1.0, "delay": 1.0}

        IOp = IO[:IO_number//2]
        IOn = IO[IO_number//2:]

        IOp_presyn = np.random.randint(np.min(IOp), np.max(IOp),
                                       size=PC_number//2).tolist()
        IOn_presyn = np.random.randint(np.min(IOn), np.max(IOn),
                                       size=PC_number//2).tolist()

        nest.Connect(IOp_presyn, vt1[:PC_number//2],
                     "one_to_one", syn_spec=vt1_syn_dict)
        nest.Connect(IOn_presyn, vt1[PC_number//2:],
                     "one_to_one", syn_spec=vt1_syn_dict)

        global cereb_index
        cereb_index += 1
        syn_name = 'stdp_synapse_sinexp_' + str(cereb_index)
        nest.CopyModel('stdp_synapse_sinexp', syn_name)

        nest.SetDefaults(syn_name,
                         {"A_minus":   LTD1,
                          "A_plus":    LTP1,
                          "Wmin":      0.0,
                          "Wmax":      30.0,
                          "vt":        vt1[0],
                          "weight_recorder": w_PFPC[0]
                          })

        PFPC_conn_dict = {'rule': 'fixed_indegree',
                          'indegree': int(0.8*GR_number),
                          "multapses": False}

        PFPC_syn_dict = {"model":  syn_name,
                         "weight": Init_PFPC, "delay":  1.0}

        for i, PCi in enumerate(PC):
            PFPC_syn_dict['vt_num'] = float(i)
            nest.Connect(GR, [PCi], PFPC_conn_dict, PFPC_syn_dict)
    else:
        PFPC_conn_dict = {'rule': 'fixed_indegree',
                          'indegree': int(0.8*GR_number),
                          "multapses": False}

        PFPC_syn_dict = {"model":  "static_synapse",
                         "weight": Init_PFPC, "delay":  1.0}

        # nest.Connect(GR, PC, PFPC_conn_dict, PFPC_syn_dict)
        for i, PCi in enumerate(PC):
            nest.Connect(GR, [PCi], PFPC_conn_dict, PFPC_syn_dict)

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
            MFDCN_syn_dict["vt_num"] = float(i)
            nest.Connect(MF, [DCNi], 'all_to_all', MFDCN_syn_dict)

        # PC-DCN inhibitory plastic connections
        # each PC sends 5 connections to positive/negative DCN
        vt2_syn_dict = {"model": "static_synapse", "weight": 1.0, "delay": 1.0}

        vt2p = vt2[:DCN_number//2]
        vt2n = vt2[DCN_number//2:]

        vt2p_presyn = np.random.randint(np.min(vt2p), np.max(vt2p),
                                        size=5*PC_number//2).tolist()
        vt2n_presyn = np.random.randint(np.min(vt2n), np.max(vt2n),
                                        size=5*PC_number//2).tolist()
        nest.Connect(PC[:PC_number//2]*5, vt2p_presyn,
                     "one_to_one", syn_spec=vt2_syn_dict)
        nest.Connect(PC[PC_number//2:]*5, vt2n_presyn,
                     "one_to_one", syn_spec=vt2_syn_dict)

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
    # each PC sends 5 connections to positive/negative DCN

    PCp = PC[:PC_number//2]
    PCn = PC[PC_number//2:]
    DCNp = DCN[:DCN_number//2]
    DCNn = DCN[DCN_number//2:]

    nest.Connect(PCp, DCNp,
                 {"rule": "fixed_outdegree", "outdegree": 5}, PCDCN_syn_dict)
    nest.Connect(PCn, DCNn,
                 {"rule": "fixed_outdegree", "outdegree": 5}, PCDCN_syn_dict)

    return MF, GR, PC, IO, DCN


def create_forward_cerebellum():
    MF, GR, PC, IO, DCN = create_cerebellum()
    sIO = SensoryIO(IO[:IO_number], IO[IO_number:])

    return Cerebellum(
        PopView(MF), PopView(GR), PopView(PC),
        sIO, DirectDCN(DCN)
    )


def create_inverse_cerebellum():
    MF, GR, PC, IO, DCN = create_cerebellum()
    mIO = MotorIO(IO[:IO_number], IO[IO_number:])

    return Cerebellum(
        PopView(MF), PopView(GR), PopView(PC),
        mIO, InverseDCN(DCN)
    )
