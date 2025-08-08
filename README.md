# Portfolio Optimization Report

A clean, reproducible pipeline to compare portfolio weighting strategies (EW, IV, ERC, GMV, MSR) with multiple covariance estimators (sample, constant correlation, shrinkage). It fetches data via `yfinance`, generates plots/tables, and compiles a LaTeX report.

## Project Structure

```
portfolio_opt/           # Python package
  ├─ __init__.py
  ├─ data_loader.py      # Download prices and compute returns
  ├─ portfolio.py        # Portfolio class and plotting
  └─ kit.py              # Utilities (risk, covariances, stats)

scripts/
  └─ generate_report.py  # Entrypoint to build results and compile LaTeX

data/
  └─ FF/                 # Fama-French CSVs (e.g., FF_Monthly_Data.csv)

notebooks/
  └─ FF_Data.ipynb

main.tex                 # LaTeX report (inputs from results/ and config/)
requirements.txt
```

Generated folders (ignored by Git):
- `results/`: LaTeX tables
- `config/`: figures (PDF)

## Setup

1) Python 3.9+

2) Create a virtualenv and install deps:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data
- Ensure `data/FF/FF_Monthly_Data.csv` exists (Fama-French monthly factors). Adjust path in `portfolio_opt/portfolio.py` if you use a different filename.

## Run
From the project root:
```
python scripts/generate_report.py
```
This will:
- Download market data via `yfinance`
- Generate figures into `config/` and tables into `results/`
- Try to compile `main.tex` using `pdflatex` (fallback path included). If it fails, run `pdflatex main.tex` manually.

## Notes
- Outputs are ignored by Git via `.gitignore`.
- The script enforces running from project root so relative paths resolve correctly.

## License
MIT
