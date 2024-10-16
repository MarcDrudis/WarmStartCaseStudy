import pathlib
import sys
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, cpu_count, delayed
from joblib_progress import joblib_progress
from qiskit.quantum_info import Statevector, state_fidelity
from scipy.interpolate import CubicSpline
from scipy.sparse.linalg import expm_multiply

from fidlib.basicfunctions import eigenrange, get_ansatz, lattice_hamiltonian
from fidlib.result_dataclasses import VarResult
from fidlib.variance import VarianceComputer

directory = pathlib.Path(__file__).parent.resolve()
plt.style.use(directory.parent / "plots/plot_style.mplstyle")
# depth = "const"
depth = "linear"
parallel = True
n_jobs = cpu_count() - 2
variance_sample_points = 5000
print("n_jobs", n_jobs)

get_hamiltonian = lambda num_qubits: lattice_hamiltonian(
    num_qubits, [("X", -1), ("ZZ", 0.95)]
)


def qubit_variance(
    num_qubits: int, times: list[float], r: float, depth: str, samples: int
) -> float:
    """
    Computes the variance for a given quantum circuit.
    Args:
        num_qubits(int): number of qubits of the system
        r(float): side of the hypercube to sample from
        depth(str): "linear" or "const" for the number of repetitions
        of the ansatz
    """
    qc = get_ansatz(int(num_qubits), depth)
    hamiltonian = get_hamiltonian(num_qubits)
    vc = VarianceComputer(
        qc=qc,
        initial_parameters=initial_parameters_list[num_qubits],
        times=times,
        H=hamiltonian,
    )

    return vc.direct_compute_variance(samples, r)


r = 0.15
qubits = [int(n) for n in sys.argv[1:]]
times = np.logspace(-3, 1, 50).tolist()

rng_initial_parameters = np.random.default_rng(0)
initial_parameters_list = [
    rng_initial_parameters.uniform(
        -np.pi, np.pi, get_ansatz(int(n), depth).num_parameters
    )
    for n in range(max(qubits) + 1)
]
qubits = [int(n) for n in sys.argv[1:]]
print("qubits", qubits)
rng_initial_parameters = np.random.default_rng(0)
initial_parameters_list = [
    rng_initial_parameters.uniform(
        -np.pi, np.pi, get_ansatz(int(n), depth).num_parameters
    )
    for n in range(max(qubits) + 1)
]

name_variance = f"var_shape_fixedr_{depth}qubits{qubits}.yaml"
if not (directory / name_variance).is_file():
    if parallel:
        with joblib_progress(
            "Simulating Variance", total=len(list(product(times, qubits)))
        ):
            jobs = [
                delayed(qubit_variance)(
                    n,
                    [r],
                    t,
                    depth,
                    variance_sample_points,
                )
                for t, n in product(times, qubits)
            ]
            variances = Parallel(n_jobs=n_jobs)(jobs)
    else:
        pass
    variances = np.array(variances).reshape((len(times), len(qubits))).T
    num_parameters = [get_ansatz(n, depth).num_parameters for n in qubits]
    result_variance = VarResult(
        r=[r],
        qubits=qubits,
        t=times,
        num_parameters=num_parameters,
        variances=variances.tolist(),
        infidelities=None,
        lambdas=[float(eigenrange(get_hamiltonian(n))) for n in qubits],
    )
    result_variance.to_yaml_file(directory / name_variance)
