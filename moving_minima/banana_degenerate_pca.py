"""This script generates Fig. 5"""

import pathlib
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import yaml
from joblib import Parallel, delayed
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector, state_fidelity
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply

from fidlib.basicfunctions import (ansatz_QRTE_Hamiltonian, eigenrange,
                                   lattice_hamiltonian)
from fidlib.result_dataclasses import MovingMinimaResult

directory = pathlib.Path(__file__).parent.resolve()
plt.style.use(directory.parent / "plots/plot_style.mplstyle")


def infidelity(
    parameters: np.ndarray,
    initial_state: Statevector,
    qc: QuantumCircuit,
    H: SparsePauliOp | None = None,
) -> float:
    state2 = Statevector(qc.assign_parameters(parameters))
    if H is not None:
        state2 = expm_multiply(-1.0j * H.to_matrix(sparse=True), state2.data)
        state2 = Statevector(state2 / np.linalg.norm(state2))
    return 1 - state_fidelity(initial_state, state2)


# We need to have an adiabatic minima trajectory to do our plots.
num_qubits = 10
data = MovingMinimaResult.from_yaml_file(
    directory.parent / f"moving_minima/XX/moving_minima_qubits={num_qubits}_seed=0.yaml"
)
# terms = [("Y", -0.95), ("XZ", 1)]
HB = lattice_hamiltonian(num_qubits, [data.H[1]])
# HB.paulis = HB.paulis[::2] + HB.paulis[1::2]
H = lattice_hamiltonian(num_qubits, [data.H[0]]) + HB

normalization = eigenrange(H)

qc = ansatz_QRTE_Hamiltonian(H, reps=2)

index = 4
alternative_params = data.perturbation[index]
time_of_cut = data.t[index]
print(f"Time is {time_of_cut}")
print(H.num_qubits, qc.num_qubits)
print(qc.num_parameters)

initial_state = Statevector(qc.assign_parameters(data.initial_parameters))


def cut(
    initial_state: Statevector,
    initial_parameters: np.ndarray,
    direction: np.ndarray,
    qc: QuantumCircuit,
    H: SparsePauliOp | None = None,
):
    """Computes the value for a given cut in our landscape.
    Args:
    initial_state: Initial state that will be time evolved in our loss function.
    initial_parameters: Parameters that lead to the initial_state in our ansatz.
    direction: Unit vector that determines the direction of the cut.
    qc: Our ansatz
    H: The Hamiltonian we study. Note that it has absorbed the timestep.
    """
    jobs = (
        delayed(infidelity)(initial_parameters + direction * p, initial_state, qc, H)
        for p in np.linspace(-0.5, 1.4, 100) * 5
    )
    return Parallel(n_jobs=11)(jobs)


width_document = 510 / 72.27
# Create subplots
fig, axs = plt.subplots(
    1,
    2,
    figsize=(width_document, width_document / 3.2),
)
count = 0

resolution = 60
grid_axis = np.linspace(-0.8, 1.8, resolution)


def cut2D(
    initial_state: Statevector,
    initial_parameters: np.ndarray,
    direction: np.ndarray,
    qc: QuantumCircuit,
    splitter=np.ndarray,
    H: SparsePauliOp | None = None,
):
    directionA = direction * splitter
    directionB = direction * (1 - splitter)

    grid = product(*(2 * [grid_axis]))
    print("Grid is computed")
    jobs = (
        delayed(infidelity)(
            initial_parameters + directionA * a + directionB * b,
            initial_state.copy(),
            qc,
            H,
        )
        for a, b in grid
    )
    return Parallel(n_jobs=9)(jobs)


from matplotlib.colors import LogNorm

## Now the 1D cuts


name = "pca_plot.yaml"


if not (directory.parent / "moving_minima" / "XX" / name).exists():
    # if True:

    cut_samples = [float(c) for c in np.linspace(-np.pi, np.pi, 501, dtype=float) * 2]
    print(type(cut_samples), type(cut_samples[0]))

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

    def get_cuts(
        initial_parameters: np.ndarray, H: SparsePauliOp, unit_cut: np.ndarray
    ):
        return (
            delayed(lossfunction)(unit_cut * p, initial_parameters, H)
            for p in cut_samples
        )

    new_parameters_array = data.initial_parameters + data.perturbation
    unit_cuts = [p / np.linalg.norm(p, ord=np.inf) for p in data.perturbation[1:]]
    unit_cuts = [unit_cuts[0]] + unit_cuts
    unit_cuts = [unit_cuts[4]] + unit_cuts[4:]

    landscapes = [
        Parallel(n_jobs=11)(get_cuts(data.initial_parameters, t * H, u))
        for t, u in zip(data.t, unit_cuts)
    ]
    trajectory = []
    inf_trajectory = []

    def add_to_trajectory(intermediate_result):
        trajectory.append((intermediate_result.x - data.initial_parameters).tolist())
        inf_trajectory.append(float(intermediate_result.fun))
        print(intermediate_result)

    result = minimize(
        infidelity,
        x0=data.initial_parameters,
        args=(initial_state, qc, H * time_of_cut),
        callback=add_to_trajectory,
        method="BFGS",
    )
    print("trajectory", len(trajectory))

    cuts_data = {
        "Landscapes": list(landscapes),
        "cut_samples": list(cut_samples),
        "times": data.t,
        "perturbation": data.perturbation,
        "trajectory": trajectory,
        "inf_trajectory": inf_trajectory,
    }

    with open(directory.parent / "moving_minima" / "XX" / name, "w") as f:
        yaml.dump(cuts_data, f)
else:
    with open(directory.parent / "moving_minima" / "XX" / name, "r") as f:
        cuts_data = yaml.safe_load(f)


# Plotting with PCA. We will find the most significant cut in the high dimensional
# landscape.


colors = [
    "#4056A1",
    "#F13C20",
    "#D79922",
    "#075C2F",
    "#692411",
] * 10

import seaborn as sns

cmap = sns.color_palette("flare", as_cmap=True)
norm = plt.Normalize(min(cuts_data["times"]), max(cuts_data["times"]))
line_colors = cmap(norm(np.array(cuts_data["times"])))[::-1]


print(cuts_data["times"])
relevant_times = [1, 5, 9]
for l, t, c in zip(cuts_data["Landscapes"], cuts_data["times"], line_colors):
    if t > 9:
        continue
    axs[0].plot(
        np.array(cuts_data["cut_samples"]) / np.pi,
        l,
        color=c,
        linestyle="-" if t not in relevant_times else "-.",
        linewidth=1 if t in relevant_times else 1,
        alpha=1 if t in relevant_times else 0.3,
        label=(
            rf"$\delta t={np.round(t*0.04158516,2)}$" if t in relevant_times else None
        ),
    )
axs[0].set_xlabel(r"Update Size, $\norm{\bm{\theta}}_{\infty}$")
axs[0].tick_params(axis="x", labelsize=11)
axs[0].set_ylabel(r"Infidelity, $\mathcal{L}(\bm{\theta})$")
axs[0].legend(loc="center left")


parameter_update = (
    np.array(cuts_data["trajectory"])[1:] - np.array(cuts_data["trajectory"])[:-1]
)
parameter_update = np.linalg.norm(parameter_update, axis=1)
infidelity_update = np.abs(
    np.array(cuts_data["inf_trajectory"])[1:]
    - np.array(cuts_data["inf_trajectory"])[:-1]
)


from orqviz.pca import (get_pca, perform_2D_pca_scan,
                        plot_optimization_trajectory_on_pca,
                        plot_pca_landscape, plot_scatter_points_on_pca)

pca = get_pca(cuts_data["trajectory"], components_ids=(0, 1))
loss_function = lambda x: infidelity(x + data.initial_parameters, initial_state, qc, H)
# plt.savefig(directory.parent / f"plots/degenerate_cut.svg")
# plt.show()

print(
    loss_function(np.array(cuts_data["trajectory"][0])),
    loss_function(np.array(cuts_data["trajectory"][-1])),
    loss_function(np.array(alternative_params)),
    loss_function(np.array(data.initial_parameters) * 0),
)
print("Initiating scan")
pca_scan_result = perform_2D_pca_scan(pca, loss_function, n_steps_x=70, offset=4)
print("Initiating plotting")

width_document = 246 / 72.27

fig, axs = plt.subplots(
    2,
    1,
    figsize=(width_document, width_document * 1.3),
    layout="constrained",
)
plot_pca_landscape(
    pca_scan_result,
    pca,
    ax=axs[0],
    cmap=cmap,
)
plot_optimization_trajectory_on_pca(
    cuts_data["trajectory"],
    pca,
    ax=axs[0],
    marker=None,
    linewidth=2,
)
plot_scatter_points_on_pca(
    [cuts_data["trajectory"][i] for i in (0, -1)],
    pca,
    color="black",
    # zorder=3,
    linewidth=1,
    ax=axs[0],
    marker="x",
    s=70,
)
axs[0].set_xlabel("")
axs[0].set_ylabel("")
axs[0].set_xticks([])
axs[0].set_yticks([])

ax2 = axs[1].twinx()
axs[1].plot(
    np.cumsum(parameter_update),
    cuts_data["inf_trajectory"][1:],
    label=r"$\mathcal{L}(\theta,t)$",
    color="#4056A1",
)
ax2.plot(
    np.cumsum(parameter_update),
    infidelity_update / parameter_update,
    label=r"$\nabla \mathcal{L}(\theta,t)$",
    color="#D79922",
)
axs[1].set_xlabel("Cummulative Trajectory")
axs[1].set_ylabel("Infidelity")
ax2.set_ylabel("Gradient")
ax2.set_yscale("log")
axs[1].set_yscale("log")
# axs[1].set_ylim(
#     (
#         np.min(cuts_data["inf_trajectory"]),
#         1e-2,
#     )
# )
# axs[1].legend(loc="center right"

axs[1].legend(loc="upper left", bbox_to_anchor=(0.05, 1))
ax2.legend(loc="upper right")
import string

for n, ax in enumerate(axs):
    ax.text(
        -0.1,
        0.95,
        string.ascii_lowercase[n] + ")",
        transform=ax.transAxes,
        weight="bold",
    )


# ax2.legend(loc="upper right")
plt.savefig(directory.parent / f"plots/pcacut.svg")
