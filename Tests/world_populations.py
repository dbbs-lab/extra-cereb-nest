import numpy as np
import matplotlib.pyplot as plt
from itertools import accumulate
import nest
from population_view import PopView
from world import select_trial_events
import trajectories

trial_len = 300


class Planner(PopView):
    def __init__(self, n, prism=0.0):
        pop = nest.Create(
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
        super().__init__(pop)


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

        self.final_x = 0

        self.joints = []
        for i in range(4):
            begin = i * n//4
            end = (i+1) * n//4
            self.joints.append(JointFibers(self.pop[begin:end]))

    def get_final_x(self):
        # legacy
        return self.final_x, 0.0

        mean = self.joints[1].x_mean
        std = self.joints[1].x_std
        return mean, std

    def integrate(self, trial_i=None):
        pop_size = len(self.joints[1].pop)

        evs, ts = self.joints[1].get_events()
        evts = zip(evs, ts)  # [(ev0, t0), (ev1, t1), ...]
        trial_evts = [
            (ev, t) for (ev, t) in evts
            if trial_len*trial_i <= t < trial_len*(trial_i+1)
        ]
        evs, ts = zip(*trial_evts)

        torques = [2.0*ev / pop_size - 1.0 for ev in evs]
        vel = np.array(list(accumulate(torques))) / pop_size
        pos = np.array(list(accumulate(vel))) / pop_size

        self.final_x = pos[-1]
        return pos[-1]


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

        self.final_x = 0.0

    def integrate(self, trial_i=0):
        pop_size = len(self.pop)

        evs, ts = self.get_events()
        evts = zip(evs, ts)  # [(ev0, t0), (ev1, t1), ...]
        trial_evts = [
            (ev, t) for (ev, t) in evts
            if trial_len*trial_i <= t < trial_len*(trial_i+1)
        ]
        if len(trial_evts) > 0:
            evs, ts = zip(*trial_evts)
        else:
            evs, ts = [], []

        torques = [1.0 if ev in self.plus.pop else -1.0
                   for ev in evs]
        vel = np.array(list(accumulate(torques))) / pop_size
        pos = np.array(list(accumulate(vel))) / pop_size

        if len(pos) > 0:
            self.final_x = pos[-1]
        else:
            self.final_x = 0.0

        # print("DCN events:", len(evs))
        # print("DCN integration result:", self.final_x)
        # print()
        return self.final_x
