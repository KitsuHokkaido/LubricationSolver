import argparse
from pathlib import Path
import yaml

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

def plot_squeeze(datas): 
    x = datas["x"]

    plt.figure(figsize=(10, 8))
    
    p = datas["p"]
    print(p.shape[0])

    plt.plot(x, p, label="p")
    
    plt.xlabel(r"angular coordinates $\theta / \pi$")
    plt.ylabel(r"pressure $p^*$")
    plt.title(f"$\\tau_0 = {datas["tau_0*"]}$, $\\epsilon = {datas["epsilon"]}$, $q* = {datas["q*"]:.3f}$")
    plt.grid(True, alpha=0.3)
    
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 8))
    
    ha = datas["ha"]
    hb = datas["hb"]
    plt.plot(x, ha, label="ha")
    plt.plot(x, hb, label="hb")

    plt.xlabel(r"angular coordinates $\theta / \pi$")
    plt.ylabel(r"relative coordinates $y/h$")
    plt.title(f"$\\tau_0 = {datas["tau_0*"]}$, $\\epsilon = {datas["epsilon"]}$, $q* = {datas["q*"]:.3f}$")
    plt.grid(True, alpha=0.3)

    plt.legend()
    plt.tight_layout()

    plt.show()

def handle_squeeze(constantes, parameters):
    squeeze_damper = SqueezeDamper(
        U1=constantes["U1"], 
        U2=constantes["U2"], 
        mu=constantes["mu"], 
        R0=constantes["R0"], 
        Ri=constantes["Ri"]
    )

    squeeze_damper.solve(
        tau_zero_star=parameters["tau_0"], 
        epsilon=parameters["epsilon"], 
        nb_points=201, 
        verbose=True
    )

    plot_squeeze(squeeze_damper.post_processing_datas)

def main() -> None:
    args = pars_arg()

    filename = Path(args.filename).resolve()

    datas = None

    with open(filename, "r") as f:
        datas = yaml.safe_load(f)

    constantes = datas["constantes"]
    parameters = datas["parameters"]
    
    match datas["geometry"]:
        case "SQUEEZE":
            handle_squeeze(constantes, parameters)
        case _:
            print("Situation does not exist")


