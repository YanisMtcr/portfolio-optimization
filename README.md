# Portfolio Optimization Report

A clean, reproducible pipeline to compare portfolio weighting strategies (EW, IV, ERC, GMV, MSR) with multiple covariance estimators (sample, constant correlation, shrinkage). It fetches data via `yfinance`, generates plots/tables, and compiles a LaTeX report.

## Features

- **Multiple Portfolio Strategies**: EW (Equal Weight), IV (Inverse Volatility), ERC (Equal Risk Contribution), GMV (Global Minimum Variance), MSR (Maximum Sharpe Ratio)
- **Covariance Estimation Methods**: Sample, Constant Correlation, Shrinkage, and Lower Bound estimators
- **Automated Report Generation**: Fetches market data, generates tables and plots, compiles LaTeX PDF
- **Backtesting & Analysis**: Wealth evolution, drawdown analysis, risk contributions, CAPM regression
- **Asset Management**: Easy cleanup of generated files

## Project Structure

```
portfolio_opt/           # Python package
  ├─ __init__.py
  ├─ data_loader.py      # Download prices and compute returns via yfinance
  ├─ portfolio.py       # Portfolio class and plotting functions
  ├─ kit.py              # Utilities (risk metrics, covariances, stats)
  └─ table_analyse.py   # Table analysis utilities

scripts/
  ├─ generate_report.py  # Entrypoint to build results and compile LaTeX
  └─ clean.py            # Cleanup script to remove generated files

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

## Requirements

- Python 3.9+
- LaTeX distribution (for PDF compilation): `pdflatex`

### Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical visualizations
- **yfinance**: Yahoo Finance API for market data
- **statsmodels**: Statistical modeling
- **scipy**: Scientific computing

## Data

- **Fama-French Data**: Ensure `data/FF/FF_Monthly_Data.csv` exists (Fama-French monthly factors). Adjust path in `portfolio_opt/portfolio.py` if you use a different filename.
- **Market Data**: Downloaded automatically via `yfinance` API

## Run

### Generate Report
From the project root:
```bash
python scripts/generate_report.py
```
This will:
- Download market data via `yfinance`
- Generate figures into `config/` and tables into `results/`
- Try to compile `main.tex` using `pdflatex` (fallback path included). If it fails, run `pdflatex main.tex` manually.

### Clean Generated Files
To remove all generated files (tables, plots, LaTeX auxiliaries):
```bash
python scripts/clean.py
```
This removes:
- All files in `results/` directory
- All files in `config/` directory  
- LaTeX auxiliary files (`.aux`, `.log`, `.out`)
- Compiled PDF (`main.pdf`)

## Portfolio Strategies Explained

The project implements several portfolio optimization strategies:

1. **EW (Equal Weight)**: Equal allocation across all assets
2. **IV (Inverse Volatility)**: Weight inversely proportional to volatility
3. **ERC (Equal Risk Contribution)**: Optimal diversification by equalizing risk contributions
4. **GMV (Global Minimum Variance)**: Minimizes portfolio variance
5. **MSR (Maximum Sharpe Ratio)**: Maximizes risk-adjusted returns

Each strategy is tested with different covariance estimators:
- **Sample**: Standard historical covariance
- **Constant Correlation**: Assumes constant pairwise correlations
- **Shrinkage**: Ledoit-Wolf shrinkage estimator
- **Lower Bound**: Constrained optimization variant

## Notes
- Outputs are ignored by Git via `.gitignore`.
- The script enforces running from project root so relative paths resolve correctly.
- Default tickers: SPY, QQQ, EFA, EEM, TLT, IEF, LQD, GLD (can be modified in `generate_report.py`)

## License
MIT
