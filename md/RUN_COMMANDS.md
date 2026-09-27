# Commands to Run Each PageRank Approach

Run these from the project root. Each prints runtime, the top-k PageRank
pages, and saves the resulting vector to `results/<name>_rank.npy`.

By default they use `--dataset data/web-Stanford.txt`, `--damping 0.85`, and
`--seed 42`. Add `--param N` to control the number of walks (see notes below
each command).

## Exact methods (for reference / ground truth)

```bash
python main.py power
```

```bash
python main.py lu
```

```bash
python main.py qr
```

## Monte Carlo approaches

### Algorithm 1 — End-Point, Random Start
`--param` = number of random walks (default = number of nodes `n`).

```bash
python main.py mc-endpoint-random
```

### Algorithm 2 — End-Point, Cyclic Start
`--param` = walks per node (default = 1).

```bash
python main.py mc-endpoint-cyclic
```

### Algorithm 3 — Complete Path
`--param` = walks per node (default = 1).

```bash
python main.py mc-complete-path
```

### Algorithm 4 — Complete Path, Dangling Stop
`--param` = walks per node (default = 1).

```bash
python main.py mc-complete-dangling
```

### Algorithm 5 — Complete Path, Random Start
`--param` = number of random walks (default = number of nodes `n`).

```bash
python main.py mc-complete-random
```

## Hybrid (Monte Carlo warm start + Power Iteration)
`--param` = walks per node for the MC warm-start step (default = 1).

```bash
python main.py hybrid-power-mc
```

---

## Getting more accuracy

Monte Carlo estimates get better (less noisy) with more walks. Bump up
`--param`, e.g.:

```bash
python main.py mc-complete-dangling --param 10
```

## Using a different dataset / damping / top-k

```bash
python main.py mc-complete-dangling --dataset data/web-Stanford.txt --damping 0.85 --top-k 10
```
