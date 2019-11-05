import nest
import sys

nest.Install("extracerebmodule")


CLOSED = nest.Create("closed_loop_neuron", 1)
RBF = nest.Create("radial_basis_function_input", 1)


sys.exit(0) #Everything went fine
