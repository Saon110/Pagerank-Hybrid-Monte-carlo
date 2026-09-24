from pagerank.exact.power import pagerank_power
from pagerank.monte_carlo.complete_path_dangling import mc_complete_path_dangling


def pagerank_hybrid(graph, runs_per_node=1, damping=0.85, tolerance=1e-12, max_iterations=200):
    """
    Hybrid PageRank: warm-start power iteration with a Monte Carlo estimate.

    Step 1: run MC Complete Path (Dangling Stop) to get a cheap, rough
            estimate of the PageRank vector.
    Step 2: use that estimate as the starting vector for power iteration,
            instead of the usual uniform 1/n start.

    Since the MC estimate is already close to the true answer, power
    iteration should need fewer iterations to converge. To show this, we
    also run power iteration from the usual uniform start and print both
    iteration counts.
    """

    print("Step 1: Monte Carlo warm start (Complete Path - Dangling Stop)...")
    initial_rank = mc_complete_path_dangling(graph, runs_per_node, damping=damping)

    print("\nStep 2: Power iteration warm-started from the MC estimate...")
    warm_rank, warm_iterations = pagerank_power(
        graph,
        damping=damping,
        tolerance=tolerance,
        max_iterations=max_iterations,
        initial_rank=initial_rank,
    )

    print("\nFor comparison: power iteration from the usual uniform (cold) start...")
    _, cold_iterations = pagerank_power(
        graph,
        damping=damping,
        tolerance=tolerance,
        max_iterations=max_iterations,
        verbose=False,
    )

    print(f"\nCold start (uniform 1/n) : {cold_iterations} iterations")
    print(f"Warm start (MC estimate) : {warm_iterations} iterations")

    return warm_rank
