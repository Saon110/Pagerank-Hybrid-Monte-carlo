# Monte Carlo PageRank Approaches — Explained Simply

PageRank is really the stationary distribution of a **random surfer**: someone
who clicks a random outgoing link with probability `damping` (0.85), and
otherwise "teleports" to a random page. Instead of solving this exactly with
linear algebra (LU/QR) or iterating a matrix (power iteration), the Monte
Carlo approach just **simulates the surfer thousands of times** and counts
where they end up or which pages they pass through. More walks = more
accurate estimate, but it's approximate and randomized (hence "Monte Carlo").

All variants build on two walk types defined in [walks.py](pagerank/monte_carlo/walks.py):

- **`random_walk` (the "X_t" process)** — Start at a page. At each step,
  stop with probability `1 - damping`. Otherwise follow a random outgoing
  link. If the page has no outgoing links ("dangling"), teleport to any
  page in the graph uniformly at random. This never gets stuck.
- **`random_walk_stop_dangling` (the "Y_t" process)** — Same idea, but if
  you land on a dangling page, the walk just **stops there** instead of
  teleporting.

Every algorithm below is just "run one of these walks many times, in some
starting pattern, and count something."

---

## Algorithm 1 — MC End-Point, Random Start ([endpoint_random.py](pagerank/monte_carlo/endpoint_random.py))

- Pick a **random** starting page each time.
- Run a full walk (with teleport-on-dangling).
- Only record **where the walk ends** (its last page).
- PageRank estimate for a page = (number of walks that ended there) / (total walks).

Intuition: if a page is important, random surfers tend to end up there more
often, purely because it's easy to reach and hard to leave.

## Algorithm 2 — MC End-Point, Cyclic Start ([endpoint_cyclic.py](pagerank/monte_carlo/endpoint_cyclic.py))

- Same as Algorithm 1, but instead of picking random starting pages, it goes
  through **every page in turn** and starts `runs_per_node` walks from each
  one (a full "cycle" through the graph).
- Still only counts the **end point** of each walk.

Intuition: guarantees every page gets a fair, equal number of walks starting
from it, which reduces sampling noise compared to picking random starts.

## Algorithm 3 — MC Complete Path ([complete_path.py](pagerank/monte_carlo/complete_path.py))

- Cyclic start again (every page, `runs_per_node` walks each).
- But this time it counts **every page visited along the entire walk**, not
  just the endpoint.
- The final counts are scaled by `(1 - damping) / total_walks`.

Intuition: instead of only asking "where did the surfer end up?", it asks
"how much total time did the surfer spend on each page?" — using the whole
path gives you much more information per walk than just the endpoint does,
so it converges faster (needs fewer walks for the same accuracy).

## Algorithm 4 — MC Complete Path, Dangling Stop ([complete_path_dangling.py](pagerank/monte_carlo/complete_path_dangling.py))

- Same as Algorithm 3 (cyclic start, count every visited page along the
  path), but uses the **other walk type** — walks stop as soon as they hit
  a dangling page instead of teleporting.
- Normalizes by dividing by the total number of page-visits recorded
  (`total_visits`) rather than a fixed formula.

Intuition: avoids "wasting" randomness on the teleport step when a dangling
page is hit — the visit counts alone already encode the right proportions.
This is the version used as the "warm start" estimate in the hybrid method.

## Algorithm 5 — MC Complete Path, Random Start ([complete_path_random.py](pagerank/monte_carlo/complete_path_random.py))

- Like Algorithm 4 (dangling-stop walk, count every page on the path), but
  starts from a **random** page each time instead of cycling through every
  page.

Intuition: same idea as Algorithm 4, just with a simpler/cheaper sampling
scheme for choosing where each walk begins — useful when you want to control
the total number of walks directly (`num_walks`) rather than "N walks per
page."

---

## The Hybrid Method ([mc_warm_start.py](pagerank/hybrid/mc_warm_start.py))

This combines Monte Carlo with the exact **power iteration** method:

1. Run **Algorithm 4** (MC Complete Path, Dangling Stop) to get a cheap,
   rough PageRank estimate very quickly.
2. Instead of starting power iteration from the usual uniform guess
   (`1/n` for every page), start it from this MC estimate.
3. Because the MC estimate is already roughly correct, power iteration needs
   **fewer iterations** to fully converge than starting from scratch.
4. The code runs both a "cold start" (uniform) and "warm start" (MC-seeded)
   power iteration and prints how many iterations each took, to show the
   speed-up.

**In short:** Monte Carlo alone gives a fast-but-noisy approximate answer;
power iteration alone gives an exact-but-slower answer; the hybrid uses MC to
give power iteration a head start, getting exact answers faster.

---

## Quick comparison

| Algorithm | Start pattern | What's counted | Dangling handling |
|---|---|---|---|
| 1. End-point random | Random | Endpoint only | Teleport |
| 2. End-point cyclic | Every page | Endpoint only | Teleport |
| 3. Complete path | Every page | Every visited page | Teleport |
| 4. Complete path (dangling stop) | Every page | Every visited page | Stop |
| 5. Complete path random | Random | Every visited page | Stop |
| Hybrid | Uses Alg. 4 as warm start, then power iteration | — | — |
