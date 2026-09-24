import numpy as np


def absolute_error(mc, exact):
    return np.abs(mc - exact)


def relative_error(mc, exact):
    return np.abs(mc - exact) / np.maximum(exact, 1e-300)


def evaluate(mc, exact, name):

    abs_err = absolute_error(mc, exact)
    rel_err = relative_error(mc, exact)

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(f"Mean absolute error : {abs_err.mean():.12e}")
    print(f"Max absolute error  : {abs_err.max():.12e}")
    print(f"Mean relative error : {rel_err.mean():.12e}")
    print(f"Max relative error  : {rel_err.max():.12e}")

    # Important pages
    ranking = np.argsort(exact)[::-1]

    for k in [1, 10, 100, 1000]:

        idx = ranking[k - 1]

        print(
            f"Rank {k:4d}: "
            f"exact={exact[idx]:.12e}, "
            f"MC={mc[idx]:.12e}, "
            f"relative error={rel_err[idx]:.6f}"
        )


def main():

    exact = np.load(
        "results/power_rank.npy"
    )

    mc = np.load(
        "results/mc_rank.npy"
    )

    evaluate(
        mc,
        exact,
        "Monte Carlo vs Power Iteration"
    )


if __name__ == "__main__":
    main()