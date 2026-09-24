#!/bin/bash

# Runs all PageRank methods and evaluates the Monte Carlo variants
# against the power-iteration ground truth.

set -e

cd "$(dirname "$0")"

echo "========================================================================"
echo "Running All PageRank Experiments"
echo "========================================================================"
echo ""

echo "Step 1/7: Power Iteration (ground truth)..."
python3 -u main.py power | tee results/power.txt
echo ""

echo "Step 2/7: MC Endpoint - Random Start..."
python3 -u main.py mc-endpoint-random | tee results/mc_endpoint_random.txt
echo ""

echo "Step 3/7: MC Endpoint - Cyclic Start..."
python3 -u main.py mc-endpoint-cyclic | tee results/mc_endpoint_cyclic.txt
echo ""

echo "Step 4/7: MC Complete Path..."
python3 -u main.py mc-complete-path | tee results/mc_complete_path.txt
echo ""

echo "Step 5/7: MC Complete Path - Dangling Stop..."
python3 -u main.py mc-complete-dangling | tee results/mc_complete_dangling.txt
echo ""

echo "Step 6/7: MC Complete Path - Random Start..."
python3 -u main.py mc-complete-random | tee results/mc_complete_random.txt
echo ""

echo "Step 7/7: Evaluating Monte Carlo methods against ground truth..."
python3 evaluate_mc.py | tee results/evaluation_m1.txt
echo ""

echo "========================================================================"
echo "All experiments completed. Results saved under results/."
echo "========================================================================"
