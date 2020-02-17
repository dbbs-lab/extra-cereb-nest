import numpy as np
import matplotlib.pyplot as plt
from itertools import accumulate
import nest
from population_view import PopView
from world import select_trial_events
import trajectories

trial_len = 300


class Event:
    def __init__(self, n_id, t):
        self.n_id = n_id
        self.t = t


class Events(list):
    def __init__(self, *args):
        if len(args) == 1:
            self._from_list(*args)
        if len(args) == 2:
            self._from_ids_ts(*args)

    def _from_list(self, ev_list):
        super().__init__(ev_list)

    def _from_ids_ts(self, n_ids, ts):
        ev_list = [Event(n_id, t) for (n_id, t) in zip(n_ids, ts)]
        self._from_list(ev_list)

    @property
    def n_ids(self):
        return (e.n_id for e in self)

    @property
    def ts(self):
        return (e.t for e in self)


class Planner(PopView):
    def __init__(self, n, prism=0.0):
        self.pop = nest.Create(
            "planner_neuron",
            n=n,
            params={
                "trial_length": trial_len,
                "target": 0.0,
                "prism_deviation": float(prism),
                "baseline_rate": 25.0,
                "gain_rate": 1.0,
                }
            )
        super().__init__(self.pop)

    def change_prism(self, prism):
        nest.SetStatus(self.pop, {"prism_deviation": float(prism)})


class JointFibers(PopView):
    def __init__(self, pop):
        super().__init__(pop)
        self.states = []  # (ts, qdd, qd, q) * n_trials
        self.x_mean = 0.0
        self.x_std = 0.0


class Cortex(PopView):
    def __init__(self, n):
        pop = nest.Create(
            "cortex_neuron",
            n=n,
            params={
                "trial_length": trial_len,
                "fibers_per_joint": n//4,
                "rbf_sdev": n/8,
                "baseline_rate": 7.0,
                "gain_rate": 4.0,
                }
            )

        for i, neuron in enumerate(pop):
            nest.SetStatus([neuron], {"joint_id": i // (n//4),
                                      "fiber_id": i % (n//4)})

        super().__init__(pop)

        self.joints = []
        for i in range(4):
            begin = i * n//4
            end = (i+1) * n//4
            self.joints.append(JointFibers(self.pop[begin:end]))

        self.torques = np.zeros(trial_len)
        self.vel = np.zeros(trial_len)
        self.pos = np.zeros(trial_len)

    def integrate(self, trial_i=0):
        pop_size = len(self.joints[1].pop)
        id_min = min(self.joints[1].pop)

        n_ids, ts = self.joints[1].get_events()
        events = Events(n_ids, ts)

        trial_events = Events(
            Event(e.n_id, e.t) for e in events
            if trial_len*trial_i <= e.t < trial_len*(trial_i+1)
        )

        self.torques = np.zeros(trial_len)
        for e in trial_events:
            t = int(np.floor(e.t)) - trial_len * trial_i
            self.torques[t] += 2.0 * (e.n_id - id_min) / pop_size - 1.0

        # torques = [2.0*n_id / pop_size - 1.0 for n_id in trial_events.n_ids]
        self.vel = np.array(list(accumulate(self.torques))) / pop_size
        self.pos = np.array(list(accumulate(self.vel))) / pop_size

        final_x = self.pos[-1]
        return final_x


class SensoryIO(PopView):
    def __init__(self, n, sensory_error=0.0):
        sensory_io = nest.Create(
            'poisson_generator',
            n=n,
            params={"rate": 0.0}
        )
        super().__init__(sensory_io)

        self.minus = self.slice(0, n//2)
        self.plus = self.slice(n//2)

        self.set_rate(sensory_error)

    def set_rate(self, sensory_error):
        # TODO: tune scale factor
        s_io_rate = 1.0 * abs(sensory_error)

        if sensory_error > 0:
            nest.SetStatus(self.plus.pop, {"rate": s_io_rate})
            nest.SetStatus(self.minus.pop, {"rate": 0.0})
        else:
            nest.SetStatus(self.plus.pop, {"rate": 0.0})
            nest.SetStatus(self.minus.pop, {"rate": s_io_rate})


class MotorIO(PopView):
    def __init__(self, n, sensory_error=0.0):
        motor_io = nest.Create('spike_generator', n=n)
        super().__init__(motor_io)

        self.plus = self.slice(0, n//2)  # [0:n]
        self.minus = self.slice(n//2)    # [n:]

        self.set_rate(sensory_error)

    def set_rate(self, sensory_error, trial_i=0):
        def make_template(upside=False):
            q_in = np.array((10.0, -10.0, -90.0, 170.0))
            q_out = np.array((0.0, 0.0, 0.0, 0.0))

            q, qd, qdd = trajectories.jtraj(q_in, q_out, trial_len)
            template = qdd[:, 1]
            template /= max(abs(template))

            if upside:
                template = -template

            template = np.clip(template, 0.0, np.max(template))
            return template

        def gen_spikes(template):
            # TODO: tune scale factor
            m_io_freqs = 0.002 * template * abs(sensory_error)

            m_io_ts = []
            for t, f in enumerate(m_io_freqs):
                n_spikes = np.random.poisson(f)
                if n_spikes:
                    m_io_ts.append(float(t+1 + trial_i*trial_len))

            return m_io_ts

        template_p = make_template()
        template_m = make_template(upside=True)

        if sensory_error > 0:
            for cell in self.plus.pop:
                nest.SetStatus([cell], {'spike_times': gen_spikes(template_p)})
            for cell in self.minus.pop:
                nest.SetStatus([cell], {'spike_times': gen_spikes(template_m)})
        else:
            for cell in self.plus.pop:
                nest.SetStatus([cell], {'spike_times': gen_spikes(template_m)})
            for cell in self.minus.pop:
                nest.SetStatus([cell], {'spike_times': gen_spikes(template_p)})


class DirectDCN(PopView):
    def __init__(self, pop_1, pop_2=None):
        if pop_2 is None:
            pop = pop_1
            super().__init__(pop)

            n = len(pop)
            self.plus = self.slice(0, n//2)  # [0:n]
            self.minus = self.slice(n//2)    # [n:]
        else:
            pop = list(pop_1) + list(pop_2)
            super().__init__(pop)

            self.plus = PopView(pop_1)
            self.minus = PopView(pop_2)


class InverseDCN(PopView):
    def __init__(self, pop_1, pop_2=None):
        if pop_2 is None:
            pop = pop_1
            super().__init__(pop)

            n = len(pop)
            self.plus = PopView(pop[0:n//2])
            self.minus = PopView(pop[n//2:])
        else:
            pop = list(pop_1) + list(pop_2)
            super().__init__(pop)

            self.plus = PopView(pop_1)
            self.minus = PopView(pop_2)

    def integrate(self, trial_i=0):
        pop_size = len(self.pop)

        n_ids, ts = self.get_events()
        events = Events(n_ids, ts)

        trial_events = Events(
            Event(e.n_id, e.t) for e in events
            if trial_len*trial_i <= e.t < trial_len*(trial_i+1)
        )

        torques = np.zeros(trial_len)
        for e in trial_events:
            t = int(np.floor(e.t)) - trial_len * trial_i
            torques[t] += 1.0 if e.n_id in self.plus.pop else -1.0

        vel = np.array(list(accumulate(torques))) / pop_size
        pos = np.array(list(accumulate(vel))) / pop_size

        final_x = pos[-1]
        return final_x
