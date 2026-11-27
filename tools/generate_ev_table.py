from pathlib import Path

import numpy as np

from ev_integrals import generate_ev_table


def main():
    repo_root = Path(__file__).resolve().parents[1]
    output = repo_root / "sionna" / "rt" / "data" / "ev_manual_22.npz"

    if output.is_file():
        print(f"EV table already exists: {output}")
        return

    nodes = np.array([
        1e-6, 1e-5, 1e-4, 0.001,
        0.003, 0.007, 0.015, 0.03,
        0.06, 0.10, 0.16, 0.25,
        0.38, 0.55, 0.80, 1.10,
        1.50, 1.95, 2.45, 3.00,
        3.8, 5.0,
    ])

    j = np.linspace(0.0, 1.0, len(nodes))
    q_max = (1 - 0.1) * (1 + 0.1)
    q_min = (1 - 0.999999) * (1 + 0.999999)
    log_q = np.log(q_max) + (np.log(q_min) - np.log(q_max)) * j**2.5
    w_nodes = np.sqrt(-np.expm1(log_q))
    w_nodes[0], w_nodes[-1] = 0.1, 0.999999

    generate_ev_table(
        alpha_nodes=nodes,
        beta_nodes=nodes,
        chi_nodes=nodes,
        w_nodes=w_nodes,
        data_dir=output.parent,
        filename=output.name,
        n_quad=128,
        max_quad=1024,
        rtol=1e-6,
        batch_size=660,
        overwrite=False,
    )


if __name__ == "__main__":
    main()