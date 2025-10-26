import pandas as pd
import os
import sys
from pathlib import Path
import subprocess
import yfinance as yf


# Add the project root to Python path so we can import portfolio_opt
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from portfolio_opt.portfolio import Portfolio
from portfolio_opt.data_loader import TickerData


def save_df_to_latex(df: pd.DataFrame, filename: str, transpose=False, **kwargs):
    """
    Saves a pandas DataFrame to a LaTeX file with proper formatting.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The name of the output .tex file.
    """
    if transpose:
        df = df.T
    column_format = kwargs.get("column_format", "l" + "c" * len(df.columns))
    df.to_latex(
        filename,
        index=kwargs.get("index", True),
        float_format=kwargs.get("float_format", "%.4f"),
        escape=kwargs.get("escape", True),
        column_format=column_format,
        header=kwargs.get("header", True),
    )


def get_company_names_and_create_tex(tickers, output_dir="results"):
    """
    Fetches company names, industry, and asset type for a list of tickers 
    and creates a LaTeX table.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    asset_data = []
    print("Fetching asset data for the report...")
    for ticker_str in tickers:
        ticker = yf.Ticker(ticker_str)
        try:
            info = ticker.info
            long_name = info.get('longName', ticker_str)
            industry = info.get('industry', 'N/A')
            quote_type = info.get('quoteType', 'N/A')
            
            if long_name == ticker_str:
                 print(f"Warning: Could not find long name for {ticker_str}.")
        except Exception:
            long_name = ticker_str
            industry = 'N/A'
            quote_type = 'N/A'
            print(f"Warning: Could not fetch info for {ticker_str}.")

        asset_data.append((ticker_str, long_name, industry, quote_type))

    df = pd.DataFrame(asset_data, columns=['Ticker', "Company Name", "Industry", "Asset Type"])
    
    filename = os.path.join(output_dir, "ticker_names.tex")
    
    save_df_to_latex(
        df, 
        filename,
        index=False,
        column_format="llll"
    )
    print(f"Asset information table saved to {filename}")


def generator_file(portfolio_instance, output_dir="results"):
    """
    Generate the files for the portfolio instance.
    """
    from portfolio_opt.kit import summary_stats

    returns_data = portfolio_instance.returns_data()
    rets_erc = returns_data[["ERC(Sample Cov)", "ERC(cc Cov)", "ERC(Shrinkage Cov)"]]
    rets_gmv = returns_data[["GMV(Sample Cov)", "GMV(cc Cov)", "GMV(Shrinkage Cov)"]]
    rets_gmv_lower_bound = returns_data[["GMV(Lower Bound)", "GMV(cc Cov)(Lower Bound)", "GMV(Shrinkage Cov)(Lower Bound)"]]
    rets_gmv_vs = returns_data[["GMV(Shrinkage Cov)", "GMV(Shrinkage Cov)(Lower Bound)"]]
    rets_msr = returns_data[["MSR(Sample Cov)", "MSR(cc Cov)", "MSR(Shrinkage Cov)"]]   
    rets_msr_vs = returns_data[["MSR(Shrinkage Cov)", "MSR(Shrinkage Cov)(Lower Bound)"]]
    rets_msr_lower_bound = returns_data[["MSR(Lower Bound)", "MSR(cc Cov)(Lower Bound)", "MSR(Shrinkage Cov)(Lower Bound)"]]
    rets_global = returns_data[["EW","ERC(Sample Cov)","GMV(Sample Cov)","GMV(Lower Bound)","MSR(Sample Cov)","MSR(Lower Bound)","IV"]]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    dataframes_to_save = {
        "rets_erc": rets_erc,
        "rets_gmv": rets_gmv,
        "rets_gmv_lower_bound": rets_gmv_lower_bound,
        "rets_gmv_vs": rets_gmv_vs,
        "rets_msr": rets_msr,
        "rets_msr_vs": rets_msr_vs,
        "rets_msr_lower_bound": rets_msr_lower_bound,
        "rets_global": rets_global
    }
    
    for name, df in dataframes_to_save.items():
        portfolio_instance.plot_wealth(df, save=True, filename=f"wealth_plot_{name}.pdf")
        
    portfolio_instance.plot_drawdown(rets_global, save=True, filename="drawdown_plot.pdf")
    portfolio_instance.plot_all_weight_evolutions()
    portfolio_instance.plot_all_risk_contributions()
    save_df_to_latex(portfolio_instance.capm(), os.path.join(output_dir, "capm.tex"))
    
    for name, df in dataframes_to_save.items():
        df_stats = summary_stats(df)
        df_stats.index.name = "Metric"
        filename = os.path.join(output_dir, f"tableau_resultats_{name}.tex")
        save_df_to_latex(df_stats, filename, transpose=True)



def main(tickers):
    
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    get_company_names_and_create_tex(tickers, "results")
    data = TickerData(tickers=tickers, start_date="2000-05-01")
    Portfolio_1 = Portfolio(data.returns_series, estimation_window=12, start_date=data.start_date, end_date=data.end_date)
    generator_file(Portfolio_1, output_dir="results")


    pdflatex_cmd = "pdflatex"
    try:
        subprocess.run([pdflatex_cmd, "main.tex"], check=True)
    except Exception:
        fallback = "/Library/TeX/texbin/pdflatex"
        try:
            subprocess.run([fallback, "main.tex"], check=True)
        except Exception as e:
            print(f"Warning: Could not run pdflatex automatically. Please compile 'main.tex' manually. Error: {e}")


if __name__ == "__main__":
    tickers = [
        'SPY', 
        'QQQ',  
        'EFA',  
        'EEM',  
        'TLT',  
        'IEF',  
        'LQD',  
        'GLD'   
    ]
    main(tickers)