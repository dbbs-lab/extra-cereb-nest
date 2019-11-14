import numpy as np
import matplotlib.pyplot as plt
from itertools import accumulate
import nest
from population_view import PopView
import trajectories

trial_len = 300


def cut_trial(evs, ts, i, norm_times=True):
    evts = zip(evs, ts)
    trial_evts = [
        (ev, t) for (ev, t) in evts
        if trial_len*i <= t < trial_len*(i+1)
    ]
    if len(trial_evts) == 0:
        return [], []

    trial_evs, trial_ts = zip(*trial_evts)

    trial_evs = np.array(trial_evs)
    trial_ts = np.array(trial_ts)

    if norm_times:
        trial_ts -= trial_len*i

    return trial_evs, trial_ts


class Planner(PopView):
    def __init__(self, n, prism=0.0):
        pop = nest.Create(
            "planner_neuron",
            n=n,
            params={
                "trial_length": trial_len,
                "target": 0.0,
                "prism_deviation": float(prism),
                "baseline_rate": 30.0,
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
                "rbf_sdev": 15.0,
                "baseline_rate": 10.0,
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

    def integrate_joint(self, evs, ts, j_id):
        pop_size = len(self.pop)
        pop_offset = min(self.pop)
        j_evs = []
        j_ts = []
        pop_size = pop_size // 4

        for ev, t in zip(evs, ts):
            fiber_id = ev - pop_offset - 1
            joint_id = fiber_id // pop_size

            if joint_id == j_id:
                j_evs.append(fiber_id % pop_size)
                j_ts.append(t)

        torques = [2.0*ev / pop_size - 1.0 for ev in j_evs]
        vel = np.array(list(accumulate(torques))) / pop_size
        pos = np.array(list(accumulate(vel))) / pop_size

        return j_ts, torques, vel, pos

    def get_final_x(self):
        mean = self.joints[1].x_mean
        std = self.joints[1].x_std
        return mean, std

    def integrate(self, n_trials=1, plot=False):
        for j, joint in enumerate(self.joints):
            evs, ts = joint.get_events()

            joint.states = []  # (ts, qdd, qd, q) * n_trials
            xs = []  # position final value

            for i in range(n_trials):
                trial_evs, trial_ts = cut_trial(evs, ts, i)
                q_ts, qdd, qd, q = self.integrate_joint(trial_evs, trial_ts, j)
                if len(q) > 0:
                    xs.append(q[-1])

                joint.states.append([q_ts, qdd, qd, q])

            joint.x_mean = np.mean(xs)
            joint.x_std = np.std(xs)

        if plot:
            fig, axs = plt.subplots(3, 4)
            for j, joint in enumerate(self.joints):
                for i in range(n_trials):
                    q_ts, qdd, qd, q = joint.states[i]
                    axs[0, j].plot(q_ts, q)
                    axs[1, j].plot(q_ts, qd)
                    axs[2, j].plot(q_ts, qdd)

            plt.show()


class SensoryIO(PopView):
    def __init__(self, n, sensory_error):
        sensory_io = nest.Create(
            'poisson_generator',
            n=n*2,
            params={"rate": 0.0}
        )
        super().__init__(sensory_io)

        self.minus = self.slice(0, n)
        self.plus = self.slice(n)

        self.set_rate(sensory_error)

    def set_rate(self, sensory_error):
        # TODO: tune scale factor
        s_io_rate = 1.0 * abs(sensory_error)

        if sensory_error > 0:
            nest.SetStatus(self.plus.pop, {"rate": s_io_rate})
        else:
            nest.SetStatus(self.minus.pop, {"rate": s_io_rate})


class MotorIO(PopView):
    def __init__(self, n, sensory_error):
        motor_io = nest.Create('spike_generator', n=n*2)
        super().__init__(motor_io)

        self.minus = self.slice(0, n)
        self.plus = self.slice(n)

        self.set_rate(sensory_error)

        self.states = []
        self.x_mean = 0.0
        self.x_std = 0.0

    def set_rate(self, sensory_error):
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
            m_io_freqs = 0.001 * template * abs(sensory_error)

            m_io_ts = []
            for t, f in enumerate(m_io_freqs):
                n_spikes = np.random.poisson(f)
                if n_spikes:
                    m_io_ts.append(float(t+1))

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

    def get_final_x(self):
        mean = self.x_mean
        std = self.x_std
        return mean, std

    def integrate(self, n_trials=1, plot=False):
        evs, ts = self.get_events()

        self.states = []  # (ts, qdd, qd, q) * n_trials
        xs = []  # position final value

        if len(evs) == 0:
            return

        for i in range(n_trials):
            trial_evs, trial_ts = cut_trial(evs, ts, i)

            pop_size = len(self.pop)
            torques = [
                (1.0 if ev in self.plus.pop else -1.0)
                for ev in trial_evs
            ]
            vel = np.array(list(accumulate(torques))) / pop_size
            pos = np.array(list(accumulate(vel))) / pop_size

            q_ts, qdd, qd, q = trial_ts, torques, vel, pos
            if len(q) == 0:
                xs.append(0.0)
            else:
                xs.append(q[-1])

            self.states.append([q_ts, qdd, qd, q])

        self.x_mean = np.mean(xs)
        self.x_std = np.std(xs)
