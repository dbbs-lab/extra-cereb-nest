from time import time
from datetime import datetime
from contextlib import contextmanager
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import nest
import world
from world_populations import Planner, Cortex
nest.set_verbosity('M_WARNING')

from cerebellum import MF_number, define_models, \
        create_forward_cerebellum, create_inverse_cerebellum
import trajectories


Brain = namedtuple("Brain", "planner cortex forward inverse")

nest.Install("cerebmodule")
nest.Install("extracerebmodule")


csv_path = './csv/'


trial_len = 300


def save_csv(name, data):
    now = datetime.now()
    path = csv_path + now.strftime("%d-%m_%H:%M_") + name
    np.savetxt(path, data, delimiter=",")


@contextmanager
def timing(label=""):
    t0 = time()
    yield None
    dt = time() - t0
    print(label + " %.2fs" % dt)


def create_brain(prism, FORWARD=False, INVERSE=False):
    trajectories.save_file(prism, trial_len)

    define_models()
    if INVERSE:
        cereb_inv = create_inverse_cerebellum()
    else:
        cereb_inv = None
    if FORWARD:
        cereb_for = create_forward_cerebellum()
    else:
        cereb_for = None

    planner = Planner(MF_number, prism)
    cortex = Cortex(MF_number)

    planner.connect(cortex)

    # Forward model:
    # - motor input from the cortex (efference copy)
    # - sensory output to the cortex
    # - sensory error signal
    if FORWARD:
        cortex.connect(cereb_for.mf)  # Efference copy

        fDCN = cereb_for.dcn
        conn_dict = {"rule": "fixed_indegree", "indegree": 1}
        nest.Connect(fDCN.plus.pop, cortex.pop, conn_dict, {'weight': 1.0})
        nest.Connect(fDCN.minus.pop, cortex.pop, conn_dict, {'weight': -1.0})

    # Inverse model;
    # - sensory input from planner
    # - motor output to world
    # - motor error signal
    if INVERSE:
        planner.connect(cereb_inv.mf)  # Sensory input

    return planner, cortex, cereb_for, cereb_inv


def get_weights(pop1, pop2):
    conns = nest.GetConnections(pop1[::50], pop2[::50])
    weights = nest.GetStatus(conns, "weight")
    return weights



FORWARD = False
INVERSE = True

error_history = []

# Get reference x
nest.ResetKernel()
trial_counter = 0
planner, cortex, _, _ = create_brain(0)
spikedetector = nest.Create("spike_detector")
nest.Connect(planner.pop, spikedetector)
xs = []
all_pos = []
spike_trial_old = 0
for i in range(10):
    nest.Simulate(trial_len)
    x = cortex.integrate(trial_i=trial_counter)
    xs.append(x)
    all_pos.append(cortex.pos)
    dSD = nest.GetStatus(spikedetector, keys="events")[0]
    evs = dSD["senders"]
    ts = dSD["times"]
    spike_trial_n = len(ts)-spike_trial_old
    spike_trial_old = len(ts)
    print(spike_trial_n, len(ts))
    print(x)
    trial_counter += 1
x_0 = np.mean(xs)

xs = []
planner.set_prism(10)
for i in range(10):
    nest.Simulate(trial_len)
    x = cortex.integrate(trial_i=trial_counter)
    xs.append(x)
    all_pos.append(cortex.pos)
    dSD = nest.GetStatus(spikedetector, keys="events")[0]
    evs = dSD["senders"]
    ts = dSD["times"]
    spike_trial_n = len(ts)-spike_trial_old
    spike_trial_old = len(ts)
    print(spike_trial_n, len(ts))
    print(x)
    trial_counter += 1

x_10 = np.mean(xs)

get_error = world.get_error_function(x_0, x_10)

# Get open loop error
for prism in np.arange(-5, 51, 5):
    planner.set_prism(prism)
    xs = []
    for i in range(10):
        nest.Simulate(trial_len)
        x = cortex.integrate(trial_i=trial_counter)
        xs.append(x)
        all_pos.append(cortex.pos)
        trial_counter += 1
        dSD = nest.GetStatus(spikedetector, keys="events")[0]
        evs = dSD["senders"]
        ts = dSD["times"]
        spike_trial_n = len(ts)-spike_trial_old
        spike_trial_old = len(ts)
        print(spike_trial_n, len(ts))
    sensory_error = get_error(xs)


    print("Prism {} - Open loop error: {} (mean {})".format(prism, sensory_error, np.mean(sensory_error[1:])))

altezza = [0.0, 10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
all_pos = np.array(all_pos)
all_pos = all_pos.flatten()
all_pos_correct = get_error(all_pos)
reference = np.zeros(len(all_pos_correct))
for i,a in enumerate(altezza):
    reference[i*3000:(i+1)*3000]=a

plt.plot(all_pos_correct)
plt.plot(reference)
plt.show()
