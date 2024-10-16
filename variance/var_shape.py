import pathlib
import sys
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, cpu_count, delayed
from joblib_progress import joblib_progress
from qiskit.quantum_info import Statevector, state_fidelity
from scipy.interpolate import CubicSpline

from fidlib.basicfunctions import get_ansatz
from fidlib.result_dataclasses import VarResult
from fidlib.variance import VarianceComputer

directory = pathlib.Path(__file__).parent.resolve()
plt.style.use(directory.parent / "plots/plot_style.mplstyle")
depth = "linear"  # Can be changed to "const"
parallel = True  # When True the function evaluations are run in parallel
n_jobs = cpu_count() - 2  # Number of available cpus for the parallel executions


def infi(num_qubits: int, r: float, depth: str, seed: int):
    """Computes the infidelity between states with initial parameters and
    perturbed parameters. The direction of the perturbation is choosen
    at random and its norm is determined by r.
    Args:
        num_qubits(int): number of qubits of the system
        r(float): infinity norm of the perturbation
        depth(str): "linear" or "const" for the number of repetitions
        of the ansatz
    """
    qc = get_ansatz(int(num_qubits), depth)
    initial_parameters = initial_parameters_list[num_qubits]
    direction = np.random.default_rng(seed).uniform(-np.pi, np.pi, qc.num_parameters)
    return state_fidelity(
        Statevector(qc.assign_parameters(initial_parameters)),
        Statevector(
            qc.assign_parameters(
                initial_parameters + direction / np.linalg.norm(direction, np.inf) * r
            )
        ),
    )


def qubit_variance(num_qubits: int, r: float, depth: str, samples: int) -> float:
    """
    Computes the variance for a given quantum circuit.
    Args:
        num_qubits(int): number of qubits of the system
        r(float): side of the hypercube to sample from
        depth(str): "linear" or "const" for the number of repetitions
        of the ansatz
    """
    qc = get_ansatz(int(num_qubits), depth)
    vc = VarianceComputer(
        qc=qc,
        initial_parameters=initial_parameters_list[num_qubits],
        times=None,
        H=None,
    )

    return vc.direct_compute_variance(samples, r)


# Hypercube sizes for the variance and infidelity plot (x axis)
rs = (np.logspace(-1.5, 0, 20) * np.pi).tolist()
# Number of qubits to compute the curves for
qubits = [int(n) for n in sys.argv[1:]]
# The initial parameters of the circuit are chosen at random
rng_initial_parameters = np.random.default_rng(0)
initial_parameters_list = [
    rng_initial_parameters.uniform(
        -np.pi, np.pi, get_ansatz(int(n), depth).num_parameters
    )
    for n in range(max(qubits) + 1)
]

variance_sample_points = 20000
name_variance = f"var_shape_{depth}qubits{qubits}.yaml"
if not (directory / name_variance).is_file():
    if parallel:
        with joblib_progress(
            "Simulating Variance", total=len(list(product(rs, qubits)))
        ):
            jobs = [
                delayed(qubit_variance)(
                    n,
                    r,
                    depth,
                    variance_sample_points,
                )
                for r, n in product(rs, qubits)
            ]
            variances = Parallel(n_jobs=n_jobs)(jobs)
    else:
        variances = [
            qubit_variance(
                n,
                r,
                depth,
                variance_sample_points,
            )
            for r, n in product(rs, qubits)
        ]
    variances = np.array(variances).reshape((len(rs), len(qubits))).T
    num_parameters = [get_ansatz(n, depth).num_parameters for n in qubits]
    result_variance = VarResult(
        r=rs,
        qubits=qubits,
        t=None,
        num_parameters=num_parameters,
        variances=variances.tolist(),
        infidelities=None,
        lambdas=None,
    )
    result_variance.to_yaml_file(directory / name_variance)
else:
    print("Loading Variance results")
    result_variance = VarResult.from_yaml_file(directory / name_variance)


N_directions = 100
if result_variance.infidelities is None:
    print("Simulating Landscape")
    iterables = list(product(rs, qubits, range(N_directions)))
    with joblib_progress("Simulating Landscape", total=len(iterables)):
        jobs = (delayed(infi)(n, r, depth, seed) for r, n, seed in iterables)
        landscape = Parallel(n_jobs=n_jobs)(jobs)
    landscape = np.array(landscape).reshape((len(rs), len(qubits), N_directions))
    mean_landscape = [
        np.mean(landscape[:, i, :], axis=1).T.tolist() for i, _ in enumerate(qubits)
    ]
    result_variance.infidelities = mean_landscape
    result_variance.to_yaml_file(directory / name_variance)
