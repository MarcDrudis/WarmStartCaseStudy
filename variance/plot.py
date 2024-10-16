import pathlib
import sys
from dataclasses import dataclass
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from dataclass_wizard import YAMLWizard
from joblib import Parallel, cpu_count, delayed
from joblib_progress import joblib_progress
from qiskit.pulse import num_qubits
from qiskit.quantum_info import Statevector, state_fidelity
from scipy.interpolate import CubicSpline, interp1d

from fidlib.basicfunctions import find_maximum, get_ansatz
from fidlib.result_dataclasses import VarResult
from fidlib.variance import VarianceComputer

directory = pathlib.Path(__file__).parent.resolve()
plt.style.use(directory.parent / "plots/plot_style.mplstyle")

from fidlib.result_dataclasses import VarResult

depth = "linear"
result = VarResult.from_yaml_file(directory / f"var_shape_{depth}.yaml")
print(result.qubits)


###########################Plotting
# Set a consistent color palette
colors = [
    "#4056a1",
    "#6b5492",
    "#895283",
    "#a15075",
    "#aa8b28",
    "#7d7c2c",
    "#4e6c2e",
    "#075c2f",
]

# fig, axs = plt.subplots(3, 1, figsize=(5, 12))
# fig.tight_layout(pad=1.0)

# The size of our document
# width_document = 246 / 72.27
width_document = 510 / 72.27

fig = plt.figure(
    layout="constrained",
    figsize=(width_document, width_document / 2),
)
subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[1.5, 1])
axA = subfigs[0].subplots(1, 1)
axB = subfigs[1].subplots(2, 1)

axs = [axA, axB[0], axB[1]]

ax2 = axs[0].twinx()
maximas = list()
maxima_value = list()

included_qubits = [4, 6, 8, 10, 12, 14, 16, 18]
color_counter = 0
for i, n in enumerate(result.qubits):
    linear_interpolation = interp1d(
        np.array(result.r) / np.pi,
        result.variances[i],
    )(
        np.logspace(
            np.log10(result.r[0] / np.pi),
            np.log10(result.r[-1] / np.pi),
            1000,
            base=10,
        )
    )

    maximas_raw = find_maximum(result.r, result.variances[i])
    maximas.append(maximas_raw[0][np.argmax(maximas_raw[1])] / np.pi)
    maxima_value.append(np.max(maximas_raw[1]))

    if n in included_qubits:
        # axs[0].scatter(
        #     x=maximas[-1],
        #     y=maxima_value[-1],
        #     color=colors[color_counter],
        #     marker="*",
        #     s=70,
        # )
        axs[0].scatter(
            x=np.array(result.r) / np.pi,
            y=result.variances[i],
            color=colors[color_counter],
        )
        axs[0].plot(
            # resolution_rs / np.pi,
            # linear_interpolation,
            np.array(result.r) / np.pi,
            result.variances[i],
            label=f"n={n}",
            linestyle="-",
            color=colors[color_counter],
        )
        ax2.plot(
            np.array(result.r) / np.pi,
            1 - np.array(result.infidelities[i]),
            label=f"n={n}",
            color=colors[color_counter],
            marker="x",
            linestyle="--",
            alpha=0.4,
        )
        axs[0].vlines(
            x=maximas[-1],
            ymin=0,
            ymax=maxima_value[-1],
            color=colors[color_counter],
        )
        color_counter += 1

axs[0].set_xlabel(r"Size of Hypercube, $\frac{r}{ \pi}$")
axs[0].set_ylabel(r"Var. in Hypercube, $\mathrm{Var}[ \mathcal{L} ]$")
ax2.set_ylabel(r"Infidelity, $\mathcal{L}(\norm{\mathbf{\theta}}_{\infty}=r)$")
axs[0].set_yscale("log")
axs[0].set_xscale("log")


from matplotlib.legend_handler import HandlerBase


class MarkerHandler(HandlerBase):
    def create_artists(
        self, legend, tup, xdescent, ydescent, width, height, fontsize, trans
    ):
        return [
            plt.Line2D(
                [width / 2],
                [height / 2.0],
                ls="",
                marker=tup[1],
                color=tup[0],
                transform=trans,
            )
        ]


axs[0].legend(
    [(colors[i], "s") for i in range(0, len(included_qubits))]
    + [("black", "."), ("black", "x")],
    [f"n={n}" for n in included_qubits]
    + [r"$\mathrm{Var}[\mathcal{L}]$", r"$\mathcal{L}$"],
    handler_map={tuple: MarkerHandler()},
    bbox_to_anchor=(0, 1.02, 1, 0.2),
    loc="lower left",
    mode="expand",
    handletextpad=0,
    borderaxespad=0,
    ncol=5,
)

########Plot scalings
starsize = 90
coeff, prefactor = np.polyfit(np.log10(result.num_parameters), np.log10(maximas), 1)
axs[1].scatter(
    result.num_parameters,
    maximas,
    label=r"$r_{max}$",
    color=colors,
    marker=".",
    s=starsize,
    zorder=20,
)
axs[1].plot(
    result.num_parameters,
    result.num_parameters**coeff * 10**prefactor,
    label=f"${{{10**prefactor:.2f}}}M^{{{coeff:.2f}}}$",
    color="black",
    zorder=1,
)
axs[1].legend(
    frameon=False,
    loc="upper right",
    bbox_to_anchor=(1.0, 1.05),
)
axs[1].set_yscale("log", base=2)
axs[1].set_xscale("log", base=2)
axs[1].tick_params(labelbottom=False)
axs[1].set_ylabel(r"Argmax of Var., $r_{max}$")

coeff, prefactor = np.polyfit(
    np.log10(result.num_parameters), np.log10(maxima_value), 1
)
axs[2].scatter(
    result.num_parameters,
    maxima_value,
    label=r"Var$[\mathcal{L}]_{max}$",
    color=colors,
    marker=".",
    s=starsize,
    zorder=20,
)
axs[2].plot(
    result.num_parameters,
    result.num_parameters**coeff * 10**prefactor,
    label=f"${{{10**prefactor:.2f}}}M^{{{coeff:.2f}}}$",
    color="black",
    zorder=1,
)
axs[2].legend(
    frameon=False,
    loc="upper right",
    bbox_to_anchor=(1.0, 1.05),
)
axs[2].set_xlabel(r"Number of Parameters, $M$")
axs[2].set_ylabel(r"Max. Var. Value")
axs[2].set_yscale("log", base=2)
axs[2].set_xscale("log", base=2)
axs[2].set_xticks([2**k for k in [5, 6, 7, 8, 9]])
axs[2].set_yticks([2 ** (-k) for k in [6, 7, 8, 9, 10, 11]])
axs[1].set_xticks([2**k for k in [5, 6, 7, 8, 9]])
axs[1].set_yticks([2 ** (-k) for k in [3, 4]])

import string

for n, ax in enumerate(axs):
    ax.text(
        -0.1,
        1.1,
        string.ascii_lowercase[n] + ")",
        transform=ax.transAxes,
        weight="bold",
    )

fig.savefig(directory.parent / f"plots/variance_{depth}.svg")
fig.savefig(directory.parent / f"plots/variance_{depth}.png")
# import os

# os.system("xdg-open " + str(directory.parent / f"plots/variance_{depth}.svg"))
