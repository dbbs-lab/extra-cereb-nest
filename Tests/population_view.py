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


def plot_spikes(evs, ts, pop=None, title='', ax=None):
    no_ax = ax is None
    if no_ax:
        # ax = plt
        fig, ax = plt.subplots(1)

    ax.scatter(ts, evs, marker='.')
    ax.set_ylabel(title)
    if pop:
        ax.set_ylim([min(pop), max(pop)])

    if no_ax:
        plt.show()


def get_rate(spike_detector, pop, n_trials=1):
    rate = nest.GetStatus(spike_detector, keys="n_events")[0] * 1e3 / (trial_len*n_trials)
    rate /= len(pop)
    return rate


class PopView:
    def __init__(self, pop):
        self.pop = pop
        self.detector = new_spike_detector(pop)

        self.total_n_events = 0
        self.rates_history = []

    def connect(self, other, rule='one_to_one', w=1.0):
        nest.Connect(self.pop, other.pop, rule, syn_spec={'weight': w})

    def slice(self, start, end=None, step=None):
        return PopView(self.pop[start:end:step])

    def get_events(self):
        return get_spike_events(self.detector)

    def get_rate(self, n_trials=1):
        return get_rate(self.detector, self.pop, n_trials)

    def plot_spikes(self, title='', ax=None):
        evs, ts = self.get_events()
        plot_spikes(evs, ts, self.pop, title, ax)

    def reset_per_trial_rate(self):
        self.total_n_events = 0
        self.rates_history = []

    def get_per_trial_rate(self):
        n_events = nest.GetStatus(self.detector, keys="n_events")[0]

        n_events -= self.total_n_events
        self.total_n_events += n_events

        rate = n_events * 1e3 / trial_len
        rate /= len(self.pop)

        self.rates_history.append(rate)
        return rate

    def plot_per_trial_rates(self, title='', ax=None):
        no_ax = ax is None
        if no_ax:
            fig, ax = plt.subplots(1)

        ax.plot(self.rates_history)
        ax.set_ylabel(title)

        if no_ax:
            plt.show()
