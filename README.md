This is a package to run quantum circuits using numpy tensor structure.

I followed this reference to learn to build a python package. https://changhsinlee.com/python-package/

# Structure
Circuits/
    |--- Circuits/
        circuit_evolution.py
        entanglement.py
        U_1_entanglement.py

- circuit_evolution.py has basic functions to evolve even and odd layers and perform measurements
- entanglement.py contains functions to calculate Renyi entropy of a pure state
- U_1_entanglement.py calculates symmetry resolved entanglement for U_1 systems.
