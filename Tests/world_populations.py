import numpy as np
from itertools import accumulate
import nest
from population_view import PopView
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
                "baseline_rate": 30.0,
                "gain_rate": 1.0,
                }
            )
        super().__init__(pop)


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


class SensoryIO(PopView):
    def __init__(self, n, sensory_error):
        sensory_io = nest.Create(
            'poisson_generator',
            n=n*2,
            params={"rate": 0.0}
        )
        s_io_minus = sensory_io[:n]
        s_io_plus = sensory_io[n:]

        # TODO: tune scale factor
        s_io_rate = 1.0 * abs(sensory_error)

        if sensory_error > 0:
            nest.SetStatus(s_io_plus, {"rate": s_io_rate})
        else:
            nest.SetStatus(s_io_minus, {"rate": s_io_rate})

        super().__init__(sensory_io)

        self.minus = self.slice(0, n)
        self.plus = self.slice(n)


class MotorIO(PopView):
    def __init__(self, n, sensory_error):
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

        motor_io = nest.Create('spike_generator', n=n*2)
        m_io_minus = motor_io[:n]
        m_io_plus = motor_io[n:]

        template_p = make_template()
        template_m = make_template(upside=True)

        if sensory_error > 0:
            for cell in m_io_plus:
                nest.SetStatus([cell], {'spike_times': gen_spikes(template_p)})
            for cell in m_io_minus:
                nest.SetStatus([cell], {'spike_times': gen_spikes(template_m)})
        else:
            for cell in m_io_plus:
                nest.SetStatus([cell], {'spike_times': gen_spikes(template_m)})
            for cell in m_io_minus:
                nest.SetStatus([cell], {'spike_times': gen_spikes(template_p)})

        super().__init__(motor_io)

        self.minus = self.slice(0, n)
        self.plus = self.slice(n)
