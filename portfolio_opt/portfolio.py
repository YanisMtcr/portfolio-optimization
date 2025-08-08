import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 


class Portfolio:
    """
    A portfolio optimization class that implements various allocation strategies.
    
    This class provides methods for different portfolio weighting strategies including
    Equal Weight, ERC, GMV, MSR, and Inverse Volatility, along with backtesting capabilities.
    """
    
    def __init__(self, returns: pd.Series, estimation_window: int,start_date: str,end_date: str):
        """
        Initialize the Portfolio object.
        
        Args:
            returns: Historical returns data
            estimation_window: Window size for parameter estimation
            start_date: Start date for analysis
            end_date: End date for analysis
        """
        self.returns = returns
        self.cov_matrix = self.cov_sample()
        self.estimation_window = estimation_window
        self.start_date = start_date
        self.end_date = end_date

    def cov_sample(self,r=None):
        """Calculate sample covariance matrix."""
        if r is None:
            return self.returns.cov()
        else:
            return r.cov()
    
    def cov_shrinkage(self):
        """Calculate shrinkage covariance matrix."""
        from portfolio_opt.kit import shrinkage_cov
        return shrinkage_cov(self.returns)
    
    def cov_cc(self):
        """Calculate constant correlation covariance matrix."""
        from portfolio_opt.kit import cc_cov
        return cc_cov(self.returns)

    def weight_ew(self, r=None):
        """Calculate equal weight allocation."""
        if r is None:
            r = self.returns
        n = len(r.columns)
        ew = pd.Series(1/n, index=r.columns)
        return ew/ew.sum()


    def weight_erc(self, r=None):
        """Calculate Equal Risk Contribution weights."""
        from portfolio_opt.kit import equal_risk_contributions
        if r is None:
            est_cov = self.cov_matrix
        else:
            est_cov = r.cov()
        return equal_risk_contributions(est_cov)
    
    def weight_erc_cc(self, r=None):
        """Calculate ERC weights using constant correlation covariance."""
        from portfolio_opt.kit import equal_risk_contributions, cc_cov
        if r is None:
            est_cov = cc_cov(self.returns)
        else:
            est_cov = cc_cov(r)
        return equal_risk_contributions(est_cov)
    
    def weight_erc_shrinkage(self, r=None):
        """Calculate ERC weights using shrinkage covariance."""
        from portfolio_opt.kit import equal_risk_contributions, shrinkage_cov
        if r is None:
            est_cov = shrinkage_cov(self.returns)
        else:
            est_cov = shrinkage_cov(r)
        return equal_risk_contributions(est_cov)

    def weight_gmv(self, r=None):
        """Calculate Global Minimum Variance weights."""
        from portfolio_opt.kit import gmv
        if r is None:
            cov = self.cov_matrix
        else:
            cov = r.cov()
        return gmv(cov)
    
    def gmv_cc(self, r=None):
        """Calculate GMV weights using constant correlation covariance."""
        from portfolio_opt.kit import gmv, cc_cov
        if r is None:
            cov = cc_cov(self.returns)
        else:
            cov = cc_cov(r)
        return gmv(cov)
    
    def gmv_shrinkage(self, r=None):
        """Calculate GMV weights using shrinkage covariance."""
        from portfolio_opt.kit import gmv, shrinkage_cov
        cov = shrinkage_cov(r)
        return gmv(cov)
    
    def weight_msr(self, r=None):
        """Calculate Maximum Sharpe Ratio weights."""
        if r is None:
            er = self.returns.mean()
            cov = self.cov_matrix
        else:
            er = r.mean()
            cov = r.cov()
        from portfolio_opt.kit import msr
        return msr(0.03, er, cov)
    
    def msr_cc(self, r=None):
        """Calculate MSR weights using constant correlation covariance."""
        from portfolio_opt.kit import msr, cc_cov
        if r is None:
            er = self.returns.mean()
            cov = cc_cov(self.returns)
        else:
            er = r.mean()
            cov = cc_cov(r)
        return msr(0.03,er,cov)
    
    def msr_shrinkage(self, r=None):
        """Calculate MSR weights using shrinkage covariance."""
        from portfolio_opt.kit import msr, shrinkage_cov 
        if r is None:
            er = self.returns.mean()
            cov = shrinkage_cov(self.returns)
        else:
            er = r.mean()
            cov = shrinkage_cov(r)
        return msr(0.03,er,cov)
    
    def weight_inv_vol(self, r=None):
        """Calculate Inverse Volatility weights."""
        if r is None:
            r = self.returns
        vol = r.std()
        inv_vol = 1 / vol
        weights = inv_vol / inv_vol.sum()
        return weights
    
    def weight_msr_lower_bound(self, r=None):
        """Calculate MSR weights with lower bound constraints."""
        if r is None:
            er = self.returns.mean()
            cov = self.cov_matrix
        else:
            er = r.mean()
            cov = r.cov()
        from portfolio_opt.kit import msr
        weights = msr(0.03,er,cov,min_weight_factor=0.5)
        return weights
    
    def weight_msr_lower_bound_cc(self, r=None):
        """Calculate MSR weights with lower bounds using constant correlation covariance."""
        from portfolio_opt.kit import msr, cc_cov
        if r is None:
            er = self.returns.mean()
            cov = cc_cov(self.returns)
        else:
            er = r.mean()
            cov = cc_cov(r)
        return msr(0.03,er,cov,min_weight_factor=0.5)
    
    def weight_msr_lower_bound_shrinkage(self, r=None):
        """Calculate MSR weights with lower bounds using shrinkage covariance."""
        from portfolio_opt.kit import msr, shrinkage_cov
        if r is None:
            er = self.returns.mean()
            cov = shrinkage_cov(self.returns)
        else:
            er = r.mean()
            cov = shrinkage_cov(r)
        return msr(0.03,er,cov,min_weight_factor=0.5)

    def weight_gmv_lower_bound(self, r=None):
        """Calculate GMV weights with lower bound constraints."""
        from portfolio_opt.kit import gmv
        if r is None:
            cov = self.cov_matrix
        else:
            cov = r.cov()
        return gmv(cov,min_weight_factor=0.5)

    def gmv_lower_bound_cc(self, r=None):
        """Calculate GMV weights with lower bounds using constant correlation covariance."""
        from portfolio_opt.kit import gmv, cc_cov
        if r is None:
            cov = cc_cov(self.returns)
        else:
            cov = cc_cov(r)
        return gmv(cov,min_weight_factor=0.5)

    def gmv_lower_bound_shrinkage(self, r=None):
        """Calculate GMV weights with lower bounds using shrinkage covariance."""
        from portfolio_opt.kit import gmv, shrinkage_cov
        if r is None:
            cov = shrinkage_cov(self.returns)
        else:
            cov = shrinkage_cov(r)
        return gmv(cov,min_weight_factor=0.5)

    def backtest_ws(self,weighting, window=None):
        """Backtest a weighting strategy with rolling windows."""
        if window is None:
            n_periods = self.returns.shape[0]
            windows = [(start, start+self.estimation_window) for start in range(n_periods-self.estimation_window)]
            weights = [weighting(self.returns.iloc[win[0]:win[1]]) for win in windows]
            weights = pd.DataFrame(weights, index=self.returns.iloc[self.estimation_window:].index, columns=self.returns.columns)
            returns = (weights.shift(1) * self.returns).sum(axis="columns",  min_count=1)
            return returns
        else:
            n_periods = self.returns.shape[0]
            windows = [(start, start+window) for start in range(n_periods-window)]
            weights = [weighting(self.returns.iloc[win[0]:win[1]]) for win in windows]
            weights = pd.DataFrame(weights, index=self.returns.iloc[window:].index, columns=self.returns.columns)
            returns = (weights.shift(1) * self.returns).sum(axis="columns",  min_count=1)
            return returns
    
    def backtest_weights(self,weighting):
        """Get weight evolution for a weighting strategy."""
        n_periods = self.returns.shape[0]
        windows = [(start, start+self.estimation_window) for start in range(n_periods-self.estimation_window)]
        weights = [weighting(self.returns.iloc[win[0]:win[1]]) for win in windows]
        weights = pd.DataFrame(weights, index=self.returns.iloc[self.estimation_window:].index, columns=self.returns.columns)
        return weights
    
    def returns_data(self,r:pd.Series=None):
        """Generate returns data for all portfolio strategies."""
        df =pd.DataFrame({
            "EW": self.backtest_ws(self.weight_ew),
            "ERC(Sample Cov)": self.backtest_ws(self.weight_erc),
            "ERC(cc Cov)": self.backtest_ws(self.weight_erc_cc),
            "ERC(Shrinkage Cov)": self.backtest_ws(self.weight_erc_shrinkage),
            "GMV(Sample Cov)": self.backtest_ws(self.weight_gmv),
            "GMV(cc Cov)": self.backtest_ws(self.gmv_cc),
            "GMV(Shrinkage Cov)": self.backtest_ws(self.gmv_shrinkage),
            "GMV(Lower Bound)": self.backtest_ws(self.weight_gmv_lower_bound),
            "GMV(cc Cov)(Lower Bound)": self.backtest_ws(self.gmv_lower_bound_cc),
            "GMV(Shrinkage Cov)(Lower Bound)": self.backtest_ws(self.gmv_lower_bound_shrinkage),
            "MSR(Sample Cov)": self.backtest_ws(self.weight_msr),
            "MSR(cc Cov)": self.backtest_ws(self.msr_cc),
            "MSR(Shrinkage Cov)": self.backtest_ws(self.msr_shrinkage),
            "IV": self.backtest_ws(self.weight_inv_vol),
            "MSR(Lower Bound)": self.backtest_ws(self.weight_msr_lower_bound),
            "MSR(cc Cov)(Lower Bound)": self.backtest_ws(self.weight_msr_lower_bound_cc),
            "MSR(Shrinkage Cov)(Lower Bound)": self.backtest_ws(self.weight_msr_lower_bound_shrinkage),
        })
        return df.dropna()

    def plot_risk_contribution(self, weighting_method, method_name,filename):
        """Plot risk contribution evolution for a weighting method."""
        from portfolio_opt.kit import risk_contribution
        weights_df = self.backtest_weights(weighting_method)
        contributions = []

        for i, date in enumerate(weights_df.index):
            w = weights_df.iloc[i].values
            window = self.returns.loc[:date].iloc[-self.estimation_window:]
            cov = self.cov_sample(window)
            rc = risk_contribution(w, cov)
            total_rc = np.sum(rc)
            if total_rc > 0:
                rc = rc / total_rc
            else:
                rc = np.ones_like(rc) / len(rc)
                
            contributions.append(rc)

        rc_df = pd.DataFrame(contributions, index=weights_df.index, columns=self.returns.columns)
        rc_df = rc_df.dropna(axis=1, how='any')
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(14, 7))
        rc_df.plot(ax=ax, lw=2, alpha=0.9)

        ax.set_title(f"Risk Contribution Evolution: {method_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Risk Contribution")
        ax.axhline(0, color='black', linestyle='--', linewidth=0.7)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
        plt.tight_layout()
        output_dir = "config"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, format="pdf", bbox_inches="tight")
        plt.close(fig)
        
    def plot_all_weight_evolutions(self):
        """
        Generates and saves weight evolution charts 
        for different portfolio allocation strategies.
        """
        plotting_configs = [
            (self.weight_ew, "Equally Weighted", "weights_plot_ew.pdf"),
            (self.weight_inv_vol, "Inverse Volatility", "weights_plot_inv_vol.pdf"),

            (self.weight_erc, "ERC (Sample)", "weights_plot_rets_erc.pdf"),
            (self.weight_erc_cc, "ERC (Constant Correlation)", "weights_plot_erc_cc.pdf"),
            (self.weight_erc_shrinkage, "ERC (Shrinkage)", "weights_plot_erc_shrinkage.pdf"),

            (self.weight_gmv, "GMV (Sample)", "weights_plot_rets_gmv.pdf"),
            (self.gmv_cc, "GMV (Constant Correlation)", "weights_plot_gmv_cc.pdf"),
            (self.gmv_shrinkage, "GMV (Shrinkage)", "weights_plot_gmv_shrinkage.pdf"),

            (self.weight_gmv_lower_bound, "GMV (Sample, Lower Bound)", "weights_plot_gmv_lb.pdf"),
            (self.gmv_lower_bound_cc, "GMV (CC, Lower Bound)", "weights_plot_gmv_cc_lb.pdf"),
            (self.gmv_lower_bound_shrinkage, "GMV (Shrinkage, Lower Bound)", "weights_plot_gmv_shrinkage_lb.pdf"),

            (self.weight_msr, "MSR (Sample)", "weights_plot_rets_msr.pdf"),
            (self.msr_cc, "MSR (Constant Correlation)", "weights_plot_msr_cc.pdf"),
            (self.msr_shrinkage, "MSR (Shrinkage)", "weights_plot_msr_shrinkage.pdf"),

            (self.weight_msr_lower_bound, "MSR (Sample, Lower Bound)", "weights_plot_msr_lb.pdf"),
            (self.weight_msr_lower_bound_cc, "MSR (CC, Lower Bound)", "weights_plot_msr_cc_lb.pdf"),
            (self.weight_msr_lower_bound_shrinkage, "MSR (Shrinkage, Lower Bound)", "weights_plot_msr_shrinkage_lb.pdf"),
        ]

        for weighting_method, method_name, filename in plotting_configs:
            weights = self.backtest_weights(weighting_method)
            self.plot_stack_weights(weights, method_name, save=True, filename=filename)


    def plot_all_risk_contributions(self):
        """Plot risk contribution evolution for all portfolio strategies."""
        plotting_configs = [
            (self.weight_ew, "Equally Weighted", "risk_contribution_plot_ew.pdf"),
            (self.weight_inv_vol, "Inverse Volatility", "risk_contribution_plot_inv_vol.pdf"),

            (self.weight_erc, "ERC (Sample)", "risk_contribution_plot_erc.pdf"),
            (self.weight_erc_cc, "ERC (Constant Correlation)", "risk_contribution_plot_erc_cc.pdf"),
            (self.weight_erc_shrinkage, "ERC (Shrinkage)", "risk_contribution_plot_erc_shrinkage.pdf"),

            (self.weight_gmv, "GMV (Sample)", "risk_contribution_plot_gmv.pdf"),
            (self.gmv_cc, "GMV (Constant Correlation)", "risk_contribution_plot_gmv_cc.pdf"),
            (self.gmv_shrinkage, "GMV (Shrinkage)", "risk_contribution_plot_gmv_shrinkage.pdf"),

            (self.weight_msr, "MSR (Sample)", "risk_contribution_plot_msr.pdf"),
            (self.msr_cc, "MSR (Constant Correlation)", "risk_contribution_plot_msr_cc.pdf"),
            (self.msr_shrinkage, "MSR (Shrinkage)", "risk_contribution_plot_msr_shrinkage.pdf"),

            (self.weight_gmv_lower_bound, "GMV (Sample, Lower Bound)", "risk_contribution_plot_gmv_lb.pdf"),
            (self.gmv_lower_bound_cc, "GMV (CC, Lower Bound)", "risk_contribution_plot_gmv_cc_lb.pdf"),
            (self.gmv_lower_bound_shrinkage, "GMV (Shrinkage, Lower Bound)", "risk_contribution_plot_gmv_shrinkage_lb.pdf"),

            (self.weight_msr_lower_bound, "MSR (Sample, Lower Bound)", "risk_contribution_plot_msr_lb.pdf"),
            (self.weight_msr_lower_bound_cc, "MSR (CC, Lower Bound)", "risk_contribution_plot_msr_cc_lb.pdf"),
            (self.weight_msr_lower_bound_shrinkage, "MSR (Shrinkage, Lower Bound)", "risk_contribution_plot_msr_shrinkage_lb.pdf"),
        ]
        
        for weighting_method, method_name, filename in plotting_configs:
            self.plot_risk_contribution(weighting_method, method_name, filename)
        

    @staticmethod
    def plot_stack_weights(weights, method, save=False, filename="stack_weights.pdf"):
        """Plot weight evolution as a stacked area chart."""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 14,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 12,
            'text.usetex': False
        })
        weights = weights.apply(pd.to_numeric, errors='coerce')
        weights = weights.dropna(axis=0, how='any')
        weights = weights.dropna(axis=1, how='any')

        plt.figure(figsize=(14, 7))
        plt.stackplot(weights.index, weights.T.values, labels=weights.columns, alpha=0.9)
        plt.title(f"Portfolio Weights Evolution: {method}")
        plt.xlabel("Date")
        plt.ylabel("Weight")
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(rotation=45)
        plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
        plt.tight_layout()
        if save:
            output_dir = "config"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight')
        else:
            plt.show()


    def plot_wealth(self, r=None, x=100, save=False, filename="wealth_plot.pdf"):
        """Plot portfolio wealth evolution."""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 11,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 12,
            'text.usetex': False
        })

        if r is None:
            wealth = x * (1 + self.returns_data()).cumprod()
        else:
            wealth = x * (1 + r).cumprod()

        fig, ax = plt.subplots(figsize=(10, 5))
        wealth.plot(ax=ax, lw=2)

        ax.set_title("Portfolio Wealth Evolution")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Wealth")


        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)

        plt.tight_layout(rect=[0, 0, 0.85, 1])  

        if save:
            output_dir = "config"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight')
        else:
            plt.show()


    def plot_drawdown(self, r=None,save=False,filename="drawdown_plot.pdf"):
        """Plot portfolio drawdown evolution."""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 11,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 12,
            'text.usetex': False
        })
        if r is None:
            wealth_index = 100 * (1 + self.returns_data()).cumprod()
        else:
            wealth_index = 100 * (1 + r).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        drawdowns.plot(figsize=(12, 6))
        plt.title("Portfolio Drawdown")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.tight_layout()
        if save:
            output_dir = "config"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight')
        else:
            plt.show()



    def capm(self):
        """Calculate CAPM and Fama-French three-factor model results."""
        import statsmodels.api as sm
        try:

            ff = pd.read_csv("data/FF/FF_Monthly_Data.csv", index_col=0, parse_dates=True)/100
 
            common_index = self.returns_data().index.intersection(ff.index)
            if common_index.empty:
                print("Error: No common dates found between returns and Fama-French data.")
                return None
            
            r_aligned = self.returns_data().loc[common_index]
            ff_aligned = ff.loc[common_index]

 
            ex_factors = ff_aligned[["Mkt-RF", "SMB", "HML"]]
            exp_var = sm.add_constant(ex_factors) 

            results = {}

            if isinstance(r_aligned, pd.DataFrame):
                for col in r_aligned.columns:
                    ex_rets = r_aligned[col] - ff_aligned["RF"]
                    

                    all_data = pd.concat([ex_rets.rename('ex_rets'), exp_var], axis=1).dropna()
                    
                    if all_data.empty:
                        print(f"Warning: No valid data for portfolio '{col}' after cleaning.")
                        continue


                    regression_vars = ['const', 'Mkt-RF', 'SMB', 'HML']
                    lm = sm.OLS(all_data['ex_rets'], all_data[regression_vars]).fit()

                    results[col] = {
                        'Alpha': lm.params['const'],
                        'Beta_Mkt': lm.params['Mkt-RF'],
                        'Beta_SMB': lm.params['SMB'],
                        'Beta_HML': lm.params['HML'],
                        'R-squared': lm.rsquared
                    }
                return pd.DataFrame(results).T
            else:
                ex_rets = r_aligned - ff_aligned["RF"]
                all_data = pd.concat([ex_rets.rename('ex_rets'), exp_var], axis=1).dropna()
                
                if all_data.empty:
                    print(f"Warning: No valid data for portfolio after cleaning.")
                    return None
                    
                regression_vars = ['const', 'Mkt-RF', 'SMB', 'HML']
                lm = sm.OLS(all_data['ex_rets'], all_data[regression_vars]).fit()
                
                return pd.Series({
                    'Alpha': lm.params['const'],
                    'Beta_Mkt': lm.params['Mkt-RF'],
                    'Beta_SMB': lm.params['SMB'],
                    'Beta_HML': lm.params['HML'],
                    'R-squared': lm.rsquared
                })

        except Exception as e:
            print(f"An error occurred in the factor model calculation: {e}")
            import traceback
            traceback.print_exc()
            return None

    def summary(self):
        """Generate summary statistics for every portfolio strategy."""
        from portfolio_opt.kit import summary_stats
        return summary_stats(self.returns_data())