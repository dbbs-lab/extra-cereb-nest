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
                "baseline_rate": 20.0,
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

    def integrate(self, n_trials=1, trial_i=None, plot=False):
        for j, joint in enumerate(self.joints):
            evs, ts = joint.get_events()

            joint.states = []  # (ts, qdd, qd, q) * n_trials
            xs = []  # position final value

            if trial_i is not None:
                trials = [trial_i]
            else:
                trials = range(n_trials)

            for trial_i in trials:
                trial_evs, trial_ts = select_trial_events(evs, ts, trial_i)
                # print("Trial times:", min(trial_ts), max(trial_ts))

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
    def __init__(self, n, sensory_error=0.0):
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
            nest.SetStatus(self.minus.pop, {"rate": 0.0})
        else:
            nest.SetStatus(self.plus.pop, {"rate": 0.0})
            nest.SetStatus(self.minus.pop, {"rate": s_io_rate})


class MotorIO(PopView):
    def __init__(self, n, sensory_error=0.0):
        motor_io = nest.Create('spike_generator', n=n*2)
        super().__init__(motor_io)

        self.plus = self.slice(0, n)  # [0:n]
        self.minus = self.slice(n)    # [n:]

        self.set_rate(sensory_error)

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


class DirectDCN(PopView):
    def __init__(self, pop):
        super().__init__(pop)

        n = len(pop)
        self.plus = self.slice(0, n//2)  # [0:n]
        self.minus = self.slice(n//2)    # [n:]


class InverseDCN_half(PopView):
    def __init__(self, pop):
        super().__init__(pop)

        self.states = []
        self.x_mean = 0.0
        self.x_std = 0.0

    def get_final_x(self):
        mean = self.x_mean
        std = self.x_std
        return mean, std

    def integrate(self, n_trials=1, trial_i=None, plot=False):
        evs, ts = self.get_events()

        self.states = []  # (ts, qdd, qd, q) * n_trials
        xs = []  # position final value

        if len(evs) == 0:
            return

        if trial_i is not None:
            trials = [trial_i]
        else:
            trials = range(n_trials)

        for i in trials:
            # if n_trials is 1 than integrate only the last trial
            trial_i = n_trials - 1 - i

            trial_evs, trial_ts = select_trial_events(evs, ts, trial_i)

            pop_size = len(self.pop)
            torques = [1.0 for ev in trial_evs if ev in self.pop]
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


class InverseDCN(PopView):
    def __init__(self, pop):
        super().__init__(pop)

        n = len(pop)
        self.plus = InverseDCN_half(pop[0:n//2])
        self.minus = InverseDCN_half(pop[n//2:])
