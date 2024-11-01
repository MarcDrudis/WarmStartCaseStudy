"""
This script creates Fig. 7. 
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase

from fidlib.basicfunctions import find_maximum
from fidlib.result_dataclasses import VarResult

directory = pathlib.Path(__file__).parent.resolve()
plt.style.use(directory.parent / "plots/plot_style.mplstyle")


depth = "linear"
result = VarResult.from_yaml_file(directory / f"var_shape_fixedr_{depth}qubits.yaml")

selected_qubits = range(5, 16 + 1)
result.variances = [result.variances[result.qubits.index(n)] for n in selected_qubits]
result.lambdas = [result.lambdas[result.qubits.index(n)] for n in selected_qubits]
result.num_parameters = [
    result.num_parameters[result.qubits.index(n)] for n in selected_qubits
]

result.qubits = [n for n in selected_qubits]

maximas_raw = [
    find_maximum(result.t, result.variances[i]) for i in range(len(result.variances))
]
maximas_variance = [m[1].max() for m in maximas_raw]
maximas_t = [m[0][np.argmax(m[1])] for m in maximas_raw]


###########################Plotting
# Set a consistent color palette

colors = [
    "#4056a1",
    "#5f5597",
    "#76548d",
    "#895283",
    "#99517a",
    "#a94f70",
    "#b74c66",
    "#b99026",
    "#9b862a",
    "#7d7c2c",
    "#5e722e",
    "#3d672f",
    "#075c2f",
]

width_document = 510 / 72.27
fig = plt.figure(
    layout="constrained",
    figsize=(width_document, width_document / 2),
)
subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[1.5, 1])
axA = subfigs[0].subplots(1, 1)
axB = subfigs[1].subplots(2, 1)

axs = [axA, axB[0], axB[1]]


for var, n, c, i in zip(
    result.variances, result.qubits, colors, range(len(result.qubits))
):
    axs[0].scatter(x=result.t, y=var, label=f"n={n}", color=c)
    axs[0].plot(result.t, var, label=f"n={n}", color=c, linestyle="-")
    axs[0].vlines(x=maximas_t[i], ymin=0, ymax=maximas_variance[i], color=c)

axs[0].set_xlabel(r"Time, $\delta t$")
axs[0].set_ylabel(r"Variance, Var$[\mathcal{L}]" + r"_{r=" + str(result.r[0]) + "}$")
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].set_xlim((1e-2, 3e0))


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


selected_legend_qubits = selected_qubits
axs[0].legend(
    [(colors[result.qubits.index(n)], "s") for n in selected_legend_qubits],
    [f"n={n}" for n in selected_legend_qubits],
    handler_map={tuple: MarkerHandler()},
    bbox_to_anchor=(0, 1.02, 1, 0.2),
    loc="lower left",
    mode="expand",
    handletextpad=0,
    borderaxespad=0,
    ncol=6,
)


########Plot scalings
coeff, prefactor = np.polyfit(np.log10(result.lambdas), np.log10(maximas_t), 1)
axs[1].plot(
    result.lambdas,
    result.lambdas**coeff * 10**prefactor,
    label=rf"${{{10**prefactor:.2f}}}\lambda_" + "{max}" + rf"^{{{coeff:.2f}}}$",
    color="black",
    zorder=1,
)
print(len(result.lambdas), len(maximas_t))
axs[1].scatter(
    result.lambdas,
    maximas_t,
    label=r"$t_{max}$",
    color=colors[: len(maximas_t)],
    zorder=2,
)

axs[1].legend(frameon=False, loc="upper right", bbox_to_anchor=(1.05, 1.05))
axs[1].set_yscale("log", base=2)
axs[1].set_xscale("log", base=2)
axs[1].set_xlabel(r"Max. Eigenval. of H, $\lambda_{max}$")
axs[1].set_ylabel(r"Argmax of Var., $t_{max}$")

coeff, prefactor = np.polyfit(
    np.log10(result.num_parameters), np.log10(maximas_variance), 1
)
axs[2].plot(
    result.num_parameters,
    result.num_parameters**coeff * 10**prefactor,
    label=f"${{{10**prefactor:.2f}}}M^{{{coeff:.2f}}}$",
    color="black",
    zorder=1,
)
axs[2].scatter(
    result.num_parameters,
    maximas_variance,
    label=r"Var$[\mathcal{L}]_{max}$",
    color=colors[: len(maximas_variance)],
    zorder=2,
)
axs[2].legend(frameon=False, loc="upper right", bbox_to_anchor=(1.05, 1.05))
axs[2].set_xlabel(r"Number of Parameters, $M$")
axs[2].set_ylabel(r"Max. Var. Value")
axs[2].set_yscale("log", base=2)
axs[2].set_xscale("log", base=2)

fig.savefig(directory.parent / f"plots/variance_time{depth}.svg")
fig.savefig(directory.parent / f"plots/variance_time{depth}.png")
