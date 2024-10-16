import pathlib
import sys

import numpy as np
from joblib import Parallel, delayed
from qiskit.quantum_info import SparsePauliOp, Statevector, state_fidelity
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply
from tqdm import tqdm

from fidlib.basicfunctions import (ansatz_QRTE_Hamiltonian, eigenrange,
                                   lattice_hamiltonian)
from fidlib.result_dataclasses import MovingMinimaResult

directory = pathlib.Path(__file__).parent.resolve()
print(directory)

runnable_args = sys.argv[1:]

num_qubits = int(runnable_args[0])
seed = int(runnable_args[1])

termsA = [["Y", -0.95]]
# termsB = [["XZ", 1]]
termsB = [["XX", 1]]
terms = termsA + termsB
HB = lattice_hamiltonian(num_qubits, termsB)
HB.paulis = HB.paulis[::2] + HB.paulis[1::2]
H = lattice_hamiltonian(num_qubits, termsA) + HB

normalization = eigenrange(H)
# H /= normalization
# print(H, normalization)

qc = ansatz_QRTE_Hamiltonian(H, reps=2)
print(qc)


# def lossfunction(
#     perturbation: np.ndarray, initial_parameters: np.ndarray, H: float | None = None
# ) -> float:
#     state1 = Statevector(qc.assign_parameters(initial_parameters + perturbation))
#     state2 = Statevector(qc.assign_parameters(initial_parameters))
#     if H is not None:
#         state2 = expm_multiply(-1.0j * H.to_matrix(sparse=True), state2.data)
#         state2 = Statevector(state2 / np.linalg.norm(state2))
#
#     return 1 - state_fidelity(state1, state2)


def lossfunction(
    perturbation: np.ndarray,
    initial_parameters: np.ndarray,
    H: SparsePauliOp | None = None,
) -> float:
    state1 = Statevector(qc.assign_parameters(initial_parameters + perturbation))
    state2 = Statevector(qc.assign_parameters(initial_parameters))
    if H is not None:
        state2 = expm_multiply(-1.0j * H.to_matrix(sparse=True), state2.data)
        state2 = Statevector(state2 / np.linalg.norm(state2))

    return 1 - state_fidelity(state1, state2)


initial_parameters = np.random.default_rng(num_qubits + seed).uniform(
    -np.pi, np.pi, qc.num_parameters
)


times = np.linspace(0, 0.5, 20 + 1)
global_inf = [lossfunction(0, initial_parameters)]
global_params = [np.zeros(qc.num_parameters)]

for t in tqdm(times[1:]):
    result_global = minimize(
        lossfunction, global_params[-1], args=(initial_parameters, H * t)
    )
    global_inf.append(float(result_global.fun))
    global_params.append(result_global.x)


def get_cuts(initial_parameters: np.ndarray, H: SparsePauliOp, unit_cut: np.ndarray):
    return (
        delayed(lossfunction)(unit_cut * p, initial_parameters, H) for p in cut_samples
    )


cut_samples = np.linspace(-np.pi, np.pi, 501) * 2
unit_cuts = [p / np.linalg.norm(p, ord=np.inf) for p in global_params[1:]]
unit_cuts = [unit_cuts[0]] + unit_cuts
landscapes = [
    Parallel(n_jobs=11)(get_cuts(initial_parameters, H * t, u))
    for t, u in zip(times, unit_cuts)
]
print(len(landscapes), len(landscapes[0]))


result = MovingMinimaResult(
    initial_parameters.tolist(),
    cut_samples.tolist(),
    global_inf,
    landscapes,
    [g.tolist() for g in global_params],
    terms,
    float(normalization),
    times.tolist(),
    seed,
)
result.to_yaml_file(directory / f"moving_minima_qubits={num_qubits}_seed={seed}.yaml")
