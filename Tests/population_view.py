import nest
import matplotlib.pyplot as plt

trial_len = 300


def new_spike_detector(pop):
    spike_detector = nest.Create("spike_detector")
    nest.Connect(pop, spike_detector)
    return spike_detector


def get_spike_events(spike_detector):
    dSD = nest.GetStatus(spike_detector, keys="events")[0]
    evs = dSD["senders"]
    ts = dSD["times"]

    return evs, ts


def plot_spikes(evs, ts, pop=None):
    plt.scatter(ts, evs, marker='.')
    if pop:
        plt.ylim(min(pop), max(pop))
    plt.show()


def get_rate(spike_detector, pop):
    rate = nest.GetStatus(spike_detector, keys="n_events")[0] * 1e3 / trial_len
    rate /= len(pop)
    return rate


class PopView:
    def __init__(self, pop):
        self.pop = pop
        self.detector = new_spike_detector(pop)

    def connect(self, other, rule='one_to_one', w=1.0):
        nest.Connect(self.pop, other.pop, rule, syn_spec={'weight': w})

    def slice(self, start, end=None, step=None):
        return PopView(self.pop[start:end:step])

    def get_events(self):
        return get_spike_events(self.detector)

    def get_rate(self):
        return get_rate(self.detector, self.pop)

    def plot_spikes(self):
        evs, ts = self.get_events()
        plot_spikes(evs, ts, self.pop)
