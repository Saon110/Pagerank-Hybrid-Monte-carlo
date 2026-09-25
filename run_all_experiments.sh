#!/bin/bash

# Runs all PageRank methods and evaluates the Monte Carlo variants
# against the power-iteration ground truth.
#
# QR is not included: it must factorize the graph's 150,532-node strongly
# connected core and does not finish here. Run it on its own with
# `python3 -u main.py qr` if you want to reproduce that.

set -e

cd "$(dirname "$0")"

echo "========================================================================"
echo "Running All PageRank Experiments"
echo "========================================================================"
echo ""

echo "Step 1/9: Power Iteration (ground truth)..."
python3 -u main.py power | tee results/power.txt
echo ""

echo "Step 2/9: Sparse LU Decomposition..."
python3 -u main.py lu | tee results/lu.txt
echo ""

echo "Step 3/9: MC Endpoint - Random Start..."
python3 -u main.py mc-endpoint-random | tee results/mc_endpoint_random.txt
echo ""

echo "Step 4/9: MC Endpoint - Cyclic Start..."
python3 -u main.py mc-endpoint-cyclic | tee results/mc_endpoint_cyclic.txt
echo ""

echo "Step 5/9: MC Complete Path..."
python3 -u main.py mc-complete-path | tee results/mc_complete_path.txt
echo ""

echo "Step 6/9: MC Complete Path - Dangling Stop..."
python3 -u main.py mc-complete-dangling | tee results/mc_complete_dangling.txt
echo ""

echo "Step 7/9: MC Complete Path - Random Start..."
python3 -u main.py mc-complete-random | tee results/mc_complete_random.txt
echo ""

echo "Step 8/9: Hybrid (MC warm start + Power Iteration)..."
python3 -u main.py hybrid-power-mc | tee results/hybrid_power_mc.txt
echo ""

echo "Step 9/9: Generating comparison figures + full error-metrics table..."
python3 -u make_all_figures.py | tee results/all_figures.txt
echo ""

echo "========================================================================"
echo "All experiments completed. Results saved under results/."
echo "========================================================================"
