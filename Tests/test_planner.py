import nest
from world_populations import Planner


nest.Install("cerebmodule")
nest.Install("extracerebmodule")

trial_len = 300


def test_spikes():
    def transpose(l):
        return list(zip(*l))

    nest.ResetKernel()
    planner = Planner(5, 20.0)

    nest.Simulate(trial_len)
    evts_1 = transpose(planner.get_events())

    nest.Simulate(trial_len)
    evts_2 = transpose(planner.get_events())[len(evts_1):]

    planner.set_prism(15.0)
    nest.Simulate(trial_len)

    planner.set_prism(20.0)
    nest.Simulate(trial_len)
    evts_3 = transpose(planner.get_events())[:len(evts_1)]

    print("Lenghts:", len(evts_1), len(evts_2), len(evts_3))
    # print(evts_1)
    # print(evts_2)
    # print(evts_3)

    for evt_1, evt_2, evt_3 in zip(evts_1, evts_2, evts_3):
        ev_1, t_1 = evt_1
        ev_2, t_2 = evt_2
        ev_3, t_3 = evt_3

        assert ev_1 == ev_2 == ev_3
        assert round(t_1, 2) == round(t_2 % trial_len, 2) == round(t_3 % trial_len, 2)


def test_rates():
    nest.ResetKernel()
    planner = Planner(10, 0.0)

    planner.set_prism(15.0)
    nest.Simulate(trial_len)
    planner.get_per_trial_rate()

    planner.set_prism(20.0)

    nest.Simulate(trial_len)
    planner.get_per_trial_rate()

    print("Rates:", planner.rates_history)


test_spikes()
test_rates()
