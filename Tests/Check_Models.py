import nest
import sys

nest.Install("extra-cerebmodule")


CLOSED = nest.Create("closed_loop_neuron", 1)
PLANNER = nest.Create("planner_neuron", 1)
RBF = nest.Create("radial_basis_function_input", 1)
CORTEX = nest.Create("cortex_neuron", 1)


sys.exit(0) #Everything went fine
