import nest
# from hbp_nrp_cle.brainsim import simulator as sim
# import numpy as np
import logging
# from hbp_nrp_excontrol.logs import clientLogger
logger = logging.getLogger(__name__)

try:
    nest.Install("cerebmodule")
    print("Albertomodule installed correctly")
except Exception as e:  # DynamicModuleManagementError
    print(e)
    print("Albertomodule already installed")

trial_len = 1000
# CEREBELLUM
PLAST1 = True   # PF-PC ex
LTP1 = 0.0
LTD1 = -1.0

# Init_PFPC = {'distribution': 'normal',
#              'mu': 1.5,
#              'sigma': 1.0}
# Init_PFPC = {'distribution': 'uniform',
#              'low': 1.0, 'high': 3.0}
Init_PFPC = 40.0
Init_MFDCN = 0.1
Init_PCDCN = -5.5
CORES = 1
RECORDING_CELLS = True


def define_models():
    nest.CopyModel('iaf_cond_exp', 'granular_neuron')
    nest.CopyModel('iaf_cond_exp', 'purkinje_neuron')
    nest.CopyModel('iaf_cond_exp', 'olivary_neuron')
    nest.CopyModel('iaf_cond_exp', 'nuclear_neuron')


def create_cerebellum():
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

    # Cell numbers
    MF_num = 10
    GR_num = 100
    PC_num = 4
    IO_num = PC_num
    DCN_num = PC_num//2

    MF = nest.Create("parrot_neuron", MF_num)
    GR = nest.Create("granular_neuron", GR_num)
    PC = nest.Create("purkinje_neuron", PC_num)
    IO = nest.Create("olivary_neuron", IO_num)
    DCN = nest.Create("nuclear_neuron", DCN_num)

    if PLAST1:
        vt = nest.Create("volume_transmitter_alberto", PC_num)
        for n, vti in enumerate(vt):
            nest.SetStatus([vti], {"vt_num": n})

    recdict2 = {"to_memory": True,
                "to_file":   False,
                "label":     "PFPC_",
                "senders":   GR,
                "targets":   PC}

    WeightPFPC = nest.Create('weight_recorder', params=recdict2)

    MFGR_conn_param = {"model": "static_synapse",
                       "weight": {'distribution': 'uniform',
                                  # -> 0.75 GR fire at 7 Hz
                                  'low': 0.07, 'high': 0.15},
                       "delay": 1.0}

    PCDCN_conn_param = {"model": "static_synapse",
                        "weight": Init_PCDCN,
                        "delay": 1.0}

    # MF-GR excitatory fixed connections
    # each GR receives 4 connections from 4 random granule cells
    nest.Connect(MF, GR, {'rule': 'fixed_indegree',
                          'indegree': 4,
                          "multapses": False}, MFGR_conn_param)

    # A_minus - Amplitude of weight change for depression
    # A_plus - Amplitude of weight change for facilitation
    # Wmin - Minimal synaptic weight
    # Wmax - Maximal synaptic weight
    if PLAST1:
        nest.SetDefaults('stdp_synapse_sinexp',
                         {"A_minus":   LTD1,
                          "A_plus":    LTP1,
                          "Wmin":      0.0,
                          "Wmax":      40.0,
                          "vt":        vt[0],
                          "weight_recorder": WeightPFPC[0]})

        PFPC_conn_param = {"model":  'stdp_synapse_sinexp',
                           "weight": Init_PFPC,
                           "delay":  1.0}

        # PF-PC excitatory plastic connections
        # each PC receives the random 80% of the GR

        for i, PCi in enumerate(PC):
            nest.Connect(GR, [PCi],
                         {'rule': 'fixed_indegree',
                          'indegree': int(0.8*GR_num),
                          "multapses": False},
                         PFPC_conn_param)
            A = nest.GetConnections(GR, [PCi])
            nest.SetStatus(A, {'vt_num': float(i)})

    else:
        PFPC_conn_param = {"model":  "static_synapse",
                           "weight": Init_PFPC,
                           "delay":  1.0}
        nest.Connect(GR, PC,
                     {'rule': 'fixed_indegree',
                      'indegree': int(0.8*GR_num),
                      "multapses": False},
                     PFPC_conn_param)

    if PLAST1:
        # IO-PC teaching connections
        # Each IO is one-to-one connected with each PC
        nest.Connect(IO, vt, {'rule': 'one_to_one'},
                     {"model": "static_synapse",
                      "weight": 1.0, "delay": 1.0})
        nest.GetConnections(IO, vt)  # IOPC_conn

    MFDCN_conn_param = {"model":  "static_synapse",
                        "weight": Init_MFDCN,
                        "delay":  10.0}

    nest.Connect(MF, DCN, 'all_to_all', MFDCN_conn_param)
    nest.GetConnections(MF, DCN)  # MFDCN_conn

    # PC-DCN inhibitory plastic connections
    # each DCN receives 2 connections from 2 contiguous PC
    count_DCN = 0
    for P in range(PC_num):
        nest.Connect([PC[P]], [DCN[count_DCN]],
                     'one_to_one', PCDCN_conn_param)
        if P % 2 == 1:
            count_DCN += 1

    nest.GetConnections(PC, DCN)  # PCDCN_conn
    # circuit = MF + PC + IO + DCN
    # return circuit
    return MF, GR, PC, IO, DCN


def get_spike_events(spike_detector):
    dSD = nest.GetStatus(spike_detector, keys="events")[0]
    evs = dSD["senders"]
    ts = dSD["times"]
    return evs, ts


def get_rate(spike_detector, pop):
    rate = nest.GetStatus(spike_detector, keys="n_events")[0] * 1e3 / trial_len
    rate /= len(pop)
    return rate


def connect_noise(cereb):
    MF, GR, PC, IO, DCN = cereb
    noise = nest.Create("poisson_generator", 10, {"rate": 50.0})
    nest.Connect(noise, MF)
    nest.Connect(noise, IO)


define_models()
cereb_1 = create_cerebellum()
cereb_2 = create_cerebellum()

connect_noise(cereb_1)
connect_noise(cereb_2)

nest.Simulate(trial_len*8)

print()

MF, GR, PC, IO, DCN = cereb_1
conns = nest.GetConnections(GR, PC)
weights = nest.GetStatus(conns, "weight")
print("Cereb 1 minimum weight at PFPC:", min(weights))

MF, GR, PC, IO, DCN = cereb_2
conns = nest.GetConnections(GR, PC)
weights = nest.GetStatus(conns, "weight")
print("Cereb 2 minimum weight at PFPC:", min(weights))
