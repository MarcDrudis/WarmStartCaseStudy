"""Generates Fig. 3"""

import pathlib
from time import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

from fidlib.result_dataclasses import (MovingMinimaResult,
                                       TildeMovingMinimaCompiledResult)

directory = pathlib.Path(__file__).parents[1].resolve() / "moving_minima"
plt.style.use(directory.parent / "plots/plot_style.mplstyle")

qubits = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# Geta data
result = TildeMovingMinimaCompiledResult.from_yaml_file(
    directory / "tilde_averaged_mov_min.yaml"
)
result_individual = MovingMinimaResult.from_yaml_file(
    directory / "XZ" / "moving_minima_qubits=10_seed=5.yaml"
)

cmap = sns.color_palette("flare", as_cmap=True)
norm = plt.Normalize(min(result_individual.t), max(result_individual.t))
line_colors = cmap(norm(result_individual.t))[::-1]

width_document = 510 / 72.27
# Create subplots
fig = plt.figure(layout="constrained", figsize=(width_document, width_document / 3.2))
subfigs = fig.subfigures(1, 2, wspace=0.03, width_ratios=[1, 1.2])

axsLeft = subfigs[0].subplots(1, 1)
axsRight = subfigs[1].subplots(1, 2, squeeze=True, sharey=True)
axs = [axsLeft, axsRight[0], axsRight[1]]

# Plot Landscape
count = 0
relevant_times = [0, 2, 5, 6, 20]
for l, t_index, c in zip(
    result_individual.landscapes, range(len(result_individual.t)), line_colors
):
    t = np.round(result_individual.t[t_index], 5)
    axs[0].plot(
        np.array(result_individual.cut_samples) / np.pi,
        l,
        color=c,
        linestyle="-" if t_index not in relevant_times else "-.",
        linewidth=1.5 if t_index in relevant_times else 1,
        alpha=1 if t_index in relevant_times else 0.5,
        label=(rf"$\delta t={t}$" if t_index in relevant_times else None),
    )
axs[0].set_xlabel(r"Update Size, $\pi \norm{\bm{\theta}}_{\infty}$")
axs[0].tick_params(axis="x", labelsize=11)
axs[0].set_ylabel(r"Infidelity, $\mathcal{L}(\bm{\theta})$")
axs[0].legend(
    borderpad=0.00001, handlelength=1.3, handletextpad=0.001, borderaxespad=0.4
)

# Plot average distance in moving minima
time_index = 8
print(f"Time is:{result.t[time_index]}")
line_colors = [
    "#4056a1",
    "#645595",
    "#7e5389",
    "#93517e",
    "#a64f72",
    "#b74c66",
    "#b38e27",
    "#8f822b",
    "#6b762d",
    "#44692e",
    "#075c2f",
]
for i, n, c in zip(range(len(qubits)), qubits, line_colors):
    mean = np.mean(result.distances[i], axis=0)
    std = np.std(result.distances[i], axis=0)
    line = axs[1].plot(result.t, mean, c=c, linestyle="-", linewidth=0.5)
    band = axs[1].fill_between(result.t, mean - std, mean + std, alpha=0.1, color=c)
    if i == 0:
        axs[1].legend([line[0], band], ["Mean", "Std."], loc="upper left")


axs[1].vlines(
    x=result.t[time_index],
    ymin=0,
    ymax=5,
    linestyles="--",
    color="black",
)
axs[1].text(
    0.58,
    result.t[time_index],
    rf"$\delta t = {result.t[time_index]}$",
    transform=axs[1].transAxes,
    weight="bold",
    rotation=90,
)
axs[1].set_ylabel(r"Update Size, $\pi \norm{\bm{\theta}}_{\infty}$")
axs[1].tick_params(axis="x", labelsize=11)
axs[1].set_xlabel(r"Time, $\delta t$")
axs[1].set_xscale("log")
axs[1].set_yscale("log")
axs[1].set_ylim((0.3, 5))

distances_t = [
    [r[time_index] for r in result.distances[i]] for i, _ in enumerate(qubits)
]

axs[2].vlines(
    qubits,
    [np.percentile(distances_t[i], [25]) for i, _ in enumerate(qubits)],
    [np.percentile(distances_t[i], [75]) for i, _ in enumerate(qubits)],
    colors=line_colors,
    linestyle="-",
    lw=5,
)
violins = axs[2].violinplot(
    distances_t,
    positions=qubits,
    showmedians=True,
    showextrema=False,
)
for v, c in zip(violins["bodies"], line_colors):
    v.set_facecolor(c)
violins["cmedians"].set_edgecolor("white")

# axs[2].set_ylabel(r"Update Size" + r", $\pi \norm{\bm{\theta}}_{\infty}$")
axs[2].tick_params(axis="x", labelsize=11)
axs[2].set_xlabel(r"Qubits, $n$")
axs[2].set_xticks(qubits)
axs[2].set_xticklabels([q if q in [4, 6, 8, 10, 12, 14] else None for q in qubits])
axs[2].set_title(rf"$\delta t ={result.t[time_index]}$")

import string

axs[0].text(
    -0.25,
    0.92,
    string.ascii_lowercase[0] + ")",
    transform=axs[0].transAxes,
    weight="bold",
)
axs[1].text(
    0.0,
    1.08,
    string.ascii_lowercase[1] + ")",
    transform=axs[1].transAxes,
    weight="bold",
)
axs[2].text(
    0.0,
    1.08,
    string.ascii_lowercase[2] + ")",
    transform=axs[2].transAxes,
    weight="bold",
)


plt.savefig(directory.parent / f"plots/moving_minima.svg")
plt.savefig(directory.parent / f"plots/moving_minima.png")
