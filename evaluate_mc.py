import numpy as np


def absolute_error(mc, exact):
    return np.abs(mc - exact)


def relative_error(mc, exact):
    return np.abs(mc - exact) / np.maximum(exact, 1e-300)


def evaluate(mc, exact, name):

    abs_err = absolute_error(mc, exact)
    rel_err = relative_error(mc, exact)

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"Mean absolute error : {abs_err.mean():.12e}")
    print(f"Max absolute error  : {abs_err.max():.12e}")
    print(f"Mean relative error : {rel_err.mean():.12e}")
    print(f"Max relative error  : {rel_err.max():.12e}")
    print(f"L1 error            : {abs_err.sum():.12e}")

    print("\nImportant PageRank positions:")

    # Important pages
    ranking = np.argsort(exact)[::-1]

    for k in [1, 10, 100, 1000]:

        idx = ranking[k - 1]

        print(
            f"Rank {k:4d}: "
            f"Exact = {exact[idx]:.12e}, "
            f"MC = {mc[idx]:.12e}, "
            f"Relative error = {rel_err[idx]:.6e}"
        )


def main():

    exact = np.load(
        "results/power_rank.npy"
    )

    methods = {
        "MC Endpoint - Random Start":
            "results/mc_endpoint_random_rank.npy",

        "MC Endpoint - Cyclic Start":
            "results/mc_endpoint_cyclic_rank.npy",

        "MC Complete Path":
            "results/mc_complete_path_rank.npy",

        "MC Complete Path - Dangling Stop":
            "results/mc_complete_dangling_rank.npy",

        "MC Complete Path - Random Start":
            "results/mc_complete_random_rank.npy",
    }

    print("Ground truth:")
    print("Power Iteration")

    print(
        f"Number of nodes: {len(exact):,}"
    )

    print(
        f"Sum of PageRank: {exact.sum():.12f}"
    )

    for name, filename in methods.items():

        try:
            mc = np.load(filename)
        except FileNotFoundError:
            print(f"\nSkipping {name}: {filename} not found. Run it via main.py first.")
            continue

        if len(mc) != len(exact):
            raise ValueError(
                f"Size mismatch for {name}: "
                f"{len(mc)} vs {len(exact)}"
            )

        evaluate(
            mc,
            exact,
            name
        )


if __name__ == "__main__":
    main()