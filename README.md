# PageRank — Setup & Run

## 1. Install Python dependencies

```bash
pip install numpy scipy
```

For Sparse QR:

```bash
sudo apt update
sudo apt install libsuitesparse-dev
pip install sparseqr
```

Verify Sparse QR:

```bash
python3 -c "import sparseqr; print('sparseqr OK')"
```

---

## 2. Run Power Iteration

```bash
python3 -u pagerank.py | tee results/power.txt
```

---

## 3. Run Sparse LU

```bash
python3 -u pagerank_lu.py | tee results/lu.txt
```

---

## 4. Run Sparse QR

```bash
python3 -u pagerank_qr.py | tee results/qr.txt
```

---

## 5. Run All Methods Sequentially

```bash
python3 -u pagerank.py | tee results/power.txt && \
python3 -u pagerank_lu.py | tee results/lu.txt && \
python3 -u pagerank_qr.py | tee results/qr.txt
```

---

## 6. View Results

```bash
cat results/power.txt
cat results/lu.txt
cat results/qr.txt
```

Or:

```bash
less results/power.txt
less results/lu.txt
less results/qr.txt
```

---

## 7. Monitor Memory / CPU

In another terminal:

```bash
htop
```

---

## 8. Stop a Running Program

```text
Ctrl + C
```