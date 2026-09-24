#!/bin/bash

# Script to run all Monte Carlo experiments and evaluation

echo "========================================================================"
echo "Running All Monte Carlo PageRank Experiments"
echo "========================================================================"
echo ""

cd /home/tahmidul-islam-omi/Pagerank-Hybrid-Monte-carlo

# Step 1: Generate ground truth with Power Iteration
echo "Step 1/5: Running Power Iteration (ground truth)..."
python3 -u pagerank_power.py | tee results/power.txt
echo ""

# Step 2: Run Algorithm 2 - MC Endpoint Cyclic
echo "Step 2/5: Running Algorithm 2 (MC Endpoint Cyclic)..."
python3 -u mc_endpoint_cyclic.py | tee results/mc_endpoint_cyclic.txt
echo ""

# Step 3: Run Algorithm 4 - MC Complete Path with Dangling Stop
echo "Step 3/5: Running Algorithm 4 (MC Complete Path - Dangling Stop)..."
python3 -u mc_complete_path_dangling.py | tee results/mc_complete_dangling.txt
echo ""

# Step 4: Run Algorithm 5 - MC Complete Path with Random Start
echo "Step 4/5: Running Algorithm 5 (MC Complete Path - Random Start)..."
python3 -u mc_complete_path_random.py | tee results/mc_complete_random.txt
echo ""

# Step 5: Evaluate all methods
echo "Step 5/5: Evaluating Monte Carlo methods against ground truth..."
python3 evaluate_mc.py | tee results/evaluation_m1.txt
echo ""

# Verify all files were created
echo "========================================================================"
echo "Verification:"
echo "========================================================================"
ls -lh results/*rank.npy 2>/dev/null && echo "✓ All rank files created" || echo "✗ Some rank files missing"
echo ""
echo "Results saved to:"
echo "  - results/power.txt"
echo "  - results/mc_endpoint_cyclic.txt"
echo "  - results/mc_complete_dangling.txt"
echo "  - results/mc_complete_random.txt"
echo "  - results/evaluation_m1.txt"
echo ""
echo "All experiments completed!"
