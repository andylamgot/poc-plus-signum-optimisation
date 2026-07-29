# Optimising Proof of Capacity Plus (PoC+) for Energy-Efficient Blockchain Consensus in Signum

MSc Computer Science final thesis reproducibility package  
**Author:** @andylamgot
**Institution:** Liverpool John Moores University / upGrad  
**Date:** July 2026

## Locked results
- Optimal parameters: plot size **6.28 MB**, deadline **200 s**, commitment ratio **0.3375**
- Mean energy reduction: **16.55 %** (n = 10 000 per arm; p < 0.001; Cohen's d ≈ 1.59)
- Stratified live sample: **5 421** Signum blocks (Jul 2025–Jul 2026, 258 miners, seed = 42)

## Hybrid data design
- **Primary experiments:** seeded synthetic workload (LP, Monte Carlo, sensitivity, adversarial tests)
- **Calibration only:** stratified real Signum blocks from public explorer/API

## Repository contents
| Path | Description |
|------|-------------|
| `src/` | Simulation and scraper scripts |
| `data/` | Stratified sample, summary CSVs, optimal vector JSON |
| `figures/` | Publication figures used in the thesis |
| `requirements.txt` | Python dependencies |

## How to reproduce
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/poc_optimization_simulation.py