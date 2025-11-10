import argparse
from pathlib import Path
import yaml

from typing import Dict, List
import matplotlib.pyplot as plt

from .engine.squeeze_damper import SqueezeDamper


def pars_arg():
    parser = argparse.ArgumentParser(
        prog="lubrication-solver",
        description="Solve a lubrication situation from a yaml filedata",
    )

    parser.add_argument(
        "filename",
        type=str,
        help="Filename of the data computed",
    )

    parser.add_argument("-v", "--version", action="version", version="%(prog)s 0.1.0")

    return parser.parse_args()


def plot_squeeze(datas: Dict) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    x = [datas["theta"], datas["theta"], datas["theta"], datas["u"][0][0]]

    all_y_data = [
        datas["p*"],
        [datas["ha"], datas["hb"]],
        datas["dp_dxs"],
        datas["u"][0][1],
    ]

    labels = ["p*", [r"$h_a$", r"$h_b$"], r"$dp/dx$", f"$u(y)$ - {datas['u'][1]}"]

    label_x = [
        r"angular coordinates $\theta / \pi$",
        r"angular coordinates $\theta / \pi$",
        r"angular coordinates $\theta / \pi$",
        r"$y$",
    ]

    label_y = [r"pressure $p^*$", r"relative coordinates $y/h$", r"$dp/dx$", r"$u(y)$"]

    title = f"$\\tau_0 = {datas["tau_0*"]}$, $\\epsilon = {datas["epsilon"]}$, $q* = {datas["q*"]:.3f}$"

    for i, y in enumerate(all_y_data):
        plt.figure(figsize=(10, 8))

        if isinstance(y, List):
            for j in range(len(y)):
                plt.plot(x[i], y[j], label=labels[i][j])
        else:
            plt.plot(x[i], y, label=labels[i])

        plt.xlabel(label_x[i])
        plt.ylabel(label_y[i])
        plt.title(title)
        plt.grid(True, alpha=0.3)

        plt.legend()
        plt.tight_layout()

    plt.show()


def handle_squeeze(datas: Dict) -> None:
    constantes = datas["constantes"]
    parameters = datas["parameters"]

    squeeze_damper = SqueezeDamper(
        U1=constantes["U1"],
        U2=constantes["U2"],
        mu=constantes["mu"],
        R0=constantes["R0"],
        Ri=constantes["Ri"],
    )

    squeeze_damper.solve(
        tau_zero_star=parameters["tau_0"],
        epsilon=parameters["epsilon"],
        nb_points=parameters["nb_points"],
        verbose=True,
    )

    plot_squeeze(squeeze_damper.post_processing_datas)


def handle_bearing():
    return


def main() -> None:
    args = pars_arg()

    filename = Path(args.filename).resolve()

    datas = None

    with open(filename, "r") as f:
        datas = yaml.safe_load(f)


    match datas["geometry"]:
        case "SQUEEZE":
            handle_squeeze(datas)
        case "BEARING":
            handle_bearing()
        case _:
            print("Situation does not exist")
