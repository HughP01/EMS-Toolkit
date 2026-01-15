#some stats functions thanks to core libraries such as numpy, scipy.stats, pandas, statsmodels
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from warnings import filterwarnings
filterwarnings('ignore')  #cleaner output


def ttest(data, formula=None, paired=False, var_equal=False, alternative='two-sided', mu=0):
    """
    Does a t test on pandas DataFrame data and prints the results
    Parameters:
    -----------
    data : pandas DataFrame
        DataFrame containing the data
    formula : str, optional
        - One-sample: "column_name" (tests if mean equals mu)
        - Two-sample: "column_name ~ group_column" 
        - Paired: "column1 ~ column2" (when paired=True)
    paired : bool, default=False
        If True, perform paired t-test
    var_equal : bool, default=False
        If True, assume equal variances (standard two-sample t-test)
        If False, use Welch's t-test (unequal variances)
    alternative : str, default='two-sided'
        Alternative hypothesis: 'two-sided', 'less', or 'greater'
    mu : float, default=0
        Theoretical mean for one-sample test
    
    Returns:
    --------
    tuple: (t_statistic, p_value, degrees_of_freedom)
    """
    
    if formula is None:
        raise ValueError("Formula must be specified. Use 'column' for one-sample or 'column ~ group' for two-sample.")
    
    # Parse the formula
    formula_parts = formula.split('~')
    
    if len(formula_parts) == 1:
        #1 sample t-test
        column = formula_parts[0].strip()
        sample_data = data[column].dropna()
        
        t_stat, p_val = stats.ttest_1samp(sample_data, mu, alternative=alternative)
        df = len(sample_data) - 1
        test_type = "One Sample t-test"
        
        mean_val = np.mean(sample_data)
        std_val = np.std(sample_data, ddof=1)
        n = len(sample_data)
        
        print(f"\n\t{test_type}")
        print(f"\nData: {column}")
        print(f"t = {t_stat:.4f}, df = {df}, p-value = {p_val:.4f}")
        
        #Alt hypothesis wording
        alt_hyp = f"true mean is not equal to {mu}"
        if alternative == 'greater':
            alt_hyp = f"true mean is greater than {mu}"
        elif alternative == 'less':
            alt_hyp = f"true mean is less than {mu}"
        
        print(f"Alternative hypothesis: {alt_hyp}")
        print(f"95 percent confidence interval:")
        ci = stats.t.interval(0.95, df, loc=mean_val, scale=std_val/np.sqrt(n))
        print(f" {ci[0]:.4f} {ci[1]:.4f}")
        print(f"Sample estimates:")
        print(f"mean of x")
        print(f"  {mean_val:.4f}")
        
    elif len(formula_parts) == 2: # 2 parts to formula
        left = formula_parts[0].strip()
        right = formula_parts[1].strip()
        
        if paired:
            #Paired t-test: column1 ~ column2
            data1 = data[left].dropna()
            data2 = data[right].dropna()
            
            common_idx = data1.index.intersection(data2.index)
            data1 = data1.loc[common_idx]
            data2 = data2.loc[common_idx]
            
            if len(data1) != len(data2):
                raise ValueError("For paired t-test, columns must have the same number of non-missing values")
            
            t_stat, p_val = stats.ttest_rel(data1, data2, alternative=alternative)
            df = len(data1) - 1
            test_type = "Paired t-test"
            
            mean_diff = np.mean(data1 - data2)
            std_diff = np.std(data1 - data2, ddof=1)
            
            print(f"\n\t{test_type}")
            print(f"\nData: {left} and {right}")
            print(f"t = {t_stat:.4f}, df = {df}, p-value = {p_val:.4f}")
            
            # Alternative hypothesis wording
            alt_hyp = "true mean difference is not equal to 0"
            if alternative == 'greater':
                alt_hyp = "true mean difference is greater than 0"
            elif alternative == 'less':
                alt_hyp = "true mean difference is less than 0"
            
            print(f"Alternative hypothesis: {alt_hyp}")
            print(f"95 percent confidence interval:")
            ci = stats.t.interval(0.95, df, loc=mean_diff, scale=std_diff/np.sqrt(len(data1)))
            print(f" {ci[0]:.4f} {ci[1]:.4f}")
            print(f"Sample estimates:")
            print(f"mean of the differences")
            print(f"  {mean_diff:.4f}")
            
        else:
            # Two-sample t-test: column ~ group_column
            # Check if right side is a column in the dataframe
            if right in data.columns:
                # It's a group column
                group_column = right
                value_column = left
                
                # Get unique groups
                groups = data[group_column].dropna().unique()
                if len(groups) != 2:
                    raise ValueError(f"Group column must have exactly 2 unique values, found {len(groups)}")
                
                group1, group2 = groups
                data1 = data[data[group_column] == group1][value_column].dropna()
                data2 = data[data[group_column] == group2][value_column].dropna()
                
                test_name = "Two Sample t-test"
                
                if var_equal:
                    # Standard two-sample t-test (equal variances)
                    t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=True, alternative=alternative)
                    df = len(data1) + len(data2) - 2
                else:
                    # Welch's t-test (unequal variances)
                    t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False, alternative=alternative)
                    
                    # Calculate degrees of freedom for Welch's test
                    n1, n2 = len(data1), len(data2)
                    s1, s2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
                    df = ((s1**2/n1 + s2**2/n2)**2) / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
                    test_name = "Welch Two Sample t-test"
                
                mean1, mean2 = np.mean(data1), np.mean(data2)
                std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
                n1, n2 = len(data1), len(data2)
                
                print(f"\n\t{test_name}")
                print(f"\nData: {value_column} by {group_column}")
                print(f"t = {t_stat:.4f}, df = {df:.2f}, p-value = {p_val:.4f}")
                
                #alt hypothesis wording
                alt_hyp = "true difference in means is not equal to 0"
                if alternative == 'greater':
                    alt_hyp = f"true difference in means of group {group1} and group {group2} is greater than 0"
                elif alternative == 'less':
                    alt_hyp = f"true difference in means of group {group1} and group {group2} is less than 0"
                else:
                    alt_hyp = f"true difference in means between group {group1} and group {group2} is not equal to 0"
                
                print(f"Alternative hypothesis: {alt_hyp}")
                print(f"95 percent confidence interval:")
                
                #confidence interval
                se = np.sqrt(std1**2/n1 + std2**2/n2)
                ci = stats.t.interval(0.95, df, loc=mean1-mean2, scale=se)
                print(f" {ci[0]:.4f} {ci[1]:.4f}")
                
                print(f"Sample estimates:")
                print(f"mean in group {group1} mean in group {group2}")
                print(f"           {mean1:.4f}            {mean2:.4f}")
                
            else:
                data1 = data[left].dropna()
                data2 = data[right].dropna()
                
                test_name = "Two Sample t-test"
                
                if var_equal:
                    t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=True, alternative=alternative)
                    df = len(data1) + len(data2) - 2
                else:
                    t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False, alternative=alternative)
                    n1, n2 = len(data1), len(data2)
                    s1, s2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
                    df = ((s1**2/n1 + s2**2/n2)**2) / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
                    test_name = "Welch Two Sample t-test"
                
                mean1, mean2 = np.mean(data1), np.mean(data2)
                
                print(f"\n\t{test_name}")
                print(f"\nData: {left} and {right}")
                print(f"t = {t_stat:.4f}, df = {df:.2f}, p-value = {p_val:.4f}")
                
                # alt hypothesis wording
                alt_hyp = "true difference in means is not equal to 0"
                if alternative == 'greater':
                    alt_hyp = "true difference in means is greater than 0"
                elif alternative == 'less':
                    alt_hyp = "true difference in means is less than 0"
                
                print(f"Alternative hypothesis: {alt_hyp}")
                print(f"95 percent confidence interval:")
                
                #find confidence interval
                std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
                n1, n2 = len(data1), len(data2)
                se = np.sqrt(std1**2/n1 + std2**2/n2)
                ci = stats.t.interval(0.95, df, loc=mean1-mean2, scale=se)
                print(f" {ci[0]:.4f} {ci[1]:.4f}")
                
                print(f"Sample estimates:")
                print(f"mean of {left} mean of {right}")
                print(f"    {mean1:.4f}     {mean2:.4f}")
    
    else:
        raise ValueError("Invalid formula format. Use 'column' or 'column ~ group' or 'column1 ~ column2'")
    
    #Add a separator line similar to R output
    print("\n" + "="*50)
    
    return t_stat, p_val, df

class DistributionIdentifier:
    """
    Comprehensive distribution identification system.
    Tests for multiple common distributions and determines the best fit.
    """
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        
        # Define distributions with their fitting functions
        self.distributions = {
            'normal': {
                'fit_func': self._fit_normal,
                'n_params': 2,
                'scipy_dist': stats.norm
            },
            'exponential': {
                'fit_func': self._fit_exponential,
                'n_params': 1,
                'scipy_dist': stats.expon
            },
            'lognormal': {
                'fit_func': self._fit_lognormal,
                'n_params': 2,
                'scipy_dist': stats.lognorm
            },
            'gamma': {
                'fit_func': self._fit_gamma,
                'n_params': 2,
                'scipy_dist': stats.gamma
            },
            'beta': {
                'fit_func': self._fit_beta,
                'n_params': 2,
                'scipy_dist': stats.beta
            },
            'uniform': {
                'fit_func': self._fit_uniform,
                'n_params': 2,
                'scipy_dist': stats.uniform
            },
            'weibull': {
                'fit_func': self._fit_weibull,
                'n_params': 2,
                'scipy_dist': stats.weibull_min
            },
            'laplace': {
                'fit_func': self._fit_laplace,
                'n_params': 2,
                'scipy_dist': stats.laplace
            },
            'rayleigh': {
                'fit_func': self._fit_rayleigh,
                'n_params': 1,
                'scipy_dist': stats.rayleigh
            },
            'pareto': {
                'fit_func': self._fit_pareto,
                'n_params': 2,
                'scipy_dist': stats.pareto
            }
        }
    
    def _fit_normal(self, data):
        """Fit normal distribution."""
        # Direct parameter calculation for stability
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        params = (mean, std)
        fitted_dist = stats.norm(loc=mean, scale=std)
        return params, fitted_dist
    
    def _fit_exponential(self, data):
        """Fit exponential distribution."""
        # Shift data to start from near 0 for better fitting
        min_val = np.min(data)
        if min_val > 0:
            shifted_data = data - min_val * 0.99
        else:
            shifted_data = data - min_val + 0.01
        
        scale = np.mean(shifted_data)
        params = (0, scale)  # loc=0, scale=mean
        fitted_dist = stats.expon(loc=0, scale=scale)
        return params, fitted_dist
    
    def _fit_lognormal(self, data):
        """Fit lognormal distribution."""
        # Ensure all data > 0
        if np.any(data <= 0):
            # Shift data to be positive
            data_positive = data - np.min(data) + 0.01 * np.std(data)
        else:
            data_positive = data
            
        params = stats.lognorm.fit(data_positive, floc=0)  # Fix location at 0
        fitted_dist = stats.lognorm(*params)
        return params, fitted_dist
    
    def _fit_gamma(self, data):
        """Fit gamma distribution."""
        params = stats.gamma.fit(data, floc=0)  # Fix location at 0
        fitted_dist = stats.gamma(*params)
        return params, fitted_dist
    
    def _fit_beta(self, data):
        """Fit beta distribution - requires data in (0,1)."""
        # Scale data to (0,1) for beta distribution
        min_val, max_val = np.min(data), np.max(data)
        scaled_data = (data - min_val) / (max_val - min_val + 1e-10)
        scaled_data = np.clip(scaled_data, 1e-6, 1 - 1e-6)
        
        # Estimate parameters using method of moments
        mean_val = np.mean(scaled_data)
        var_val = np.var(scaled_data, ddof=1)
        
        if var_val > 0:
            # Method of moments for beta distribution
            a = mean_val * (mean_val * (1 - mean_val) / var_val - 1)
            b = (1 - mean_val) * (mean_val * (1 - mean_val) / var_val - 1)
            
            # Ensure parameters are positive
            a = max(a, 0.1)
            b = max(b, 0.1)
            
            params = (a, b, 0, 1)  # a, b, loc=0, scale=1
            fitted_dist = stats.beta(a, b, loc=0, scale=1)
        else:
            # Fallback to uniform if variance is too small
            params = (1, 1, 0, 1)
            fitted_dist = stats.beta(1, 1, loc=0, scale=1)
        
        return params, fitted_dist
    
    def _fit_uniform(self, data):
        """Fit uniform distribution."""
        loc = np.min(data)
        scale = np.max(data) - loc
        params = (loc, scale)
        fitted_dist = stats.uniform(loc=loc, scale=scale)
        return params, fitted_dist
    
    def _fit_weibull(self, data):
        """Fit Weibull distribution."""
        # Ensure all data > 0
        if np.any(data <= 0):
            data_positive = data - np.min(data) + 0.01 * np.std(data)
        else:
            data_positive = data
            
        params = stats.weibull_min.fit(data_positive, floc=0)  # Fix location at 0
        fitted_dist = stats.weibull_min(*params)
        return params, fitted_dist
    
    def _fit_laplace(self, data):
        """Fit Laplace distribution."""
        # Direct parameter calculation
        loc = np.median(data)
        scale = np.mean(np.abs(data - loc))
        params = (loc, scale)
        fitted_dist = stats.laplace(loc=loc, scale=scale)
        return params, fitted_dist
    
    def _fit_rayleigh(self, data):
        """Fit Rayleigh distribution."""
        # Shift data to be positive
        if np.any(data < 0):
            data_positive = data - np.min(data) + 0.01 * np.std(data)
        else:
            data_positive = data
            
        params = stats.rayleigh.fit(data_positive, floc=0)  # Fix location at 0
        fitted_dist = stats.rayleigh(*params)
        return params, fitted_dist
    
    def _fit_pareto(self, data):
        """Fit Pareto distribution."""
        # Ensure all data > 0 and find minimum
        min_val = np.min(data)
        if min_val <= 0:
            data_positive = data - min_val + 0.01 * np.std(data)
        else:
            data_positive = data
            
        # Use a simple estimator for Pareto
        shape = 1.0  # Default shape
        if len(data_positive) > 1:
            log_data = np.log(data_positive)
            shape = 1.0 / (np.mean(log_data) - np.log(np.min(data_positive)))
            shape = max(shape, 0.1)  # Ensure positive
            
        params = (shape, np.min(data_positive), 1)  # shape, loc=min, scale=1
        fitted_dist = stats.pareto(shape, loc=np.min(data_positive), scale=1)
        return params, fitted_dist
    
    def _calculate_goodness_of_fit(self, data, fitted_dist, n_params):
        """Calculate goodness of fit metrics."""
        try:
            # Perform KS test using the fitted distribution
            ks_stat, ks_p = stats.kstest(data, fitted_dist.cdf)
            
            # Calculate log-likelihood safely
            logpdf_vals = fitted_dist.logpdf(data)
            # Replace -inf with a very small number
            logpdf_vals = np.where(np.isneginf(logpdf_vals), -1e10, logpdf_vals)
            log_likelihood = np.sum(logpdf_vals)
            
            # Calculate AIC and BIC
            n = len(data)
            aic = 2 * n_params - 2 * log_likelihood
            bic = n_params * np.log(n) - 2 * log_likelihood
            
            return ks_stat, ks_p, log_likelihood, aic, bic
            
        except Exception as e:
            # Return default values if calculation fails
            return 1.0, 0.0, -np.inf, np.inf, np.inf
    
    def fit_distribution(self, data, dist_name):
        """Fit a specific distribution to data."""
        try:
            dist_info = self.distributions[dist_name]
            fit_func = dist_info['fit_func']
            n_params = dist_info['n_params']
            
            # Fit distribution
            params, fitted_dist = fit_func(data)
            
            # Calculate goodness of fit
            ks_stat, ks_p, log_likelihood, aic, bic = self._calculate_goodness_of_fit(
                data, fitted_dist, n_params
            )
            
            # Create parameter dictionary
            param_names = self._get_param_names(dist_name, params)
            param_dict = {}
            for i, (name, value) in enumerate(zip(param_names, params)):
                param_dict[name] = value
            
            return {
                'name': dist_name,
                'params': params,
                'param_dict': param_dict,
                'fitted_dist': fitted_dist,
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p,
                'good_fit': ks_p > self.alpha,
                'log_likelihood': log_likelihood,
                'aic': aic,
                'bic': bic,
                'n_params': n_params
            }
            
        except Exception as e:
            return {
                'name': dist_name,
                'error': str(e),
                'good_fit': False,
                'ks_p_value': 0,
                'aic': np.inf,
                'bic': np.inf,
                'log_likelihood': -np.inf
            }
    
    def _get_param_names(self, dist_name, params):
        """Get parameter names for a distribution."""
        param_templates = {
            'normal': ['mean', 'std'],
            'exponential': ['loc', 'scale'],
            'lognormal': ['shape', 'loc', 'scale'],
            'gamma': ['shape', 'loc', 'scale'],
            'beta': ['a', 'b', 'loc', 'scale'],
            'uniform': ['loc', 'scale'],
            'weibull': ['shape', 'loc', 'scale'],
            'laplace': ['loc', 'scale'],
            'rayleigh': ['loc', 'scale'],
            'pareto': ['shape', 'loc', 'scale']
        }
        
        template = param_templates.get(dist_name, [f'param_{i}' for i in range(len(params))])
        # Ensure we have the right number of names
        return template[:len(params)]
    
    def test_normality(self, data, column=None, visualize=True):
        """
        Test if data follows a normal distribution.
        """
        sample_data, data_name = self._extract_data(data, column)
        
        if len(sample_data) < 3:
            return {"error": "Insufficient data points"}
        
        results = {
            'data_name': data_name,
            'sample_size': len(sample_data),
            'alpha': self.alpha,
            'tests': {},
            'descriptive': self._calculate_descriptive_stats(sample_data)
        }
        
        # Perform normality tests
        normality_tests = self._perform_normality_tests(sample_data)
        results['tests'] = normality_tests
        
        # Overall conclusion
        normal_tests = [test['is_normal'] for test in normality_tests.values()]
        normal_count = sum(normal_tests)
        total_tests = len(normal_tests)
        
        results['overall'] = {
            'normal_tests_passed': normal_count,
            'total_tests': total_tests,
            'is_normal': normal_count / total_tests >= 0.5,
            'confidence': f"{normal_count}/{total_tests} tests indicate normality"
        }
        
        # Visualize if requested
        if visualize:
            self._visualize_normality(sample_data, results)
        
        return results
    
    def identify_distribution(self, data, column=None, top_n=5, visualize=True):
        """
        Test multiple distributions and identify the best fit.
        """
        sample_data, data_name = self._extract_data(data, column)
        
        if len(sample_data) < 10:
            return {"error": "Insufficient data points for distribution identification"}
        
        results = {
            'data_name': data_name,
            'sample_size': len(sample_data),
            'alpha': self.alpha,
            'tested_distributions': list(self.distributions.keys()),
            'fits': {},
            'descriptive': self._calculate_descriptive_stats(sample_data)
        }
        
        # Fit all distributions
        for dist_name in self.distributions:
            fit_result = self.fit_distribution(sample_data, dist_name)
            results['fits'][dist_name] = fit_result
        
        # Rank distributions by goodness-of-fit
        valid_fits = {k: v for k, v in results['fits'].items() 
                     if 'ks_p_value' in v and not np.isnan(v['ks_p_value']) and v['ks_p_value'] > 0}
        
        if not valid_fits:
            # If all fits failed, try to at least get some results
            for dist_name in results['fits']:
                if 'ks_p_value' in results['fits'][dist_name]:
                    valid_fits[dist_name] = results['fits'][dist_name]
        
        # Sort by KS p-value (descending), then AIC (ascending)
        ranked = sorted(valid_fits.items(), 
                       key=lambda x: (-x[1]['ks_p_value'], x[1]['aic']))
        
        results['ranked_distributions'] = [
            {
                'rank': i + 1,
                'distribution': dist_name,
                'ks_p_value': fit['ks_p_value'],
                'aic': fit['aic'],
                'bic': fit['bic'],
                'params': fit.get('params', []),
                'good_fit': fit.get('good_fit', False)
            }
            for i, (dist_name, fit) in enumerate(ranked[:top_n])
        ]
        
        # Best fit
        if results['ranked_distributions']:
            results['best_fit'] = results['ranked_distributions'][0]
        else:
            results['best_fit'] = None
        
        # Visualize if requested
        if visualize:
            self._visualize_distributions(sample_data, results)
        
        return results
    
    def _extract_data(self, data, column):
        """Extract data from various input types."""
        if isinstance(data, pd.DataFrame) and column:
            sample_data = data[column].dropna().values
            data_name = column
        elif isinstance(data, pd.Series):
            sample_data = data.dropna().values
            data_name = data.name if data.name else 'Series'
        else:
            sample_data = np.array(data).flatten()
            sample_data = sample_data[~np.isnan(sample_data)]
            data_name = 'Array'
        return sample_data, data_name
    
    def _calculate_descriptive_stats(self, data):
        """Calculate descriptive statistics."""
        return {
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data, ddof=1),
            'variance': np.var(data, ddof=1),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'min': np.min(data),
            'max': np.max(data),
            'range': np.ptp(data),
            'q1': np.percentile(data, 25),
            'q3': np.percentile(data, 75),
            'iqr': np.percentile(data, 75) - np.percentile(data, 25),
            'cv': np.std(data, ddof=1) / np.mean(data) if np.mean(data) != 0 else np.nan
        }
    
    def _perform_normality_tests(self, data):
        """Perform multiple normality tests."""
        tests = {}
        
        # Shapiro-Wilk Test
        if len(data) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(data)
            tests['Shapiro-Wilk'] = {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > self.alpha,
                'test': 'Best for n < 5000'
            }
        
        # D'Agostino's K^2 Test
        dagostino_stat, dagostino_p = stats.normaltest(data)
        tests["D'Agostino's K^2"] = {
            'statistic': dagostino_stat,
            'p_value': dagostino_p,
            'is_normal': dagostino_p > self.alpha,
            'test': 'Combines skewness & kurtosis'
        }
        
        # Kolmogorov-Smirnov Test (with fitted normal)
        try:
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            normal_dist = stats.norm(loc=mean, scale=std)
            ks_stat, ks_p = stats.kstest(data, normal_dist.cdf)
            tests['Kolmogorov-Smirnov'] = {
                'statistic': ks_stat,
                'p_value': ks_p,
                'is_normal': ks_p > self.alpha,
                'test': 'Compares to fitted normal'
            }
        except:
            tests['Kolmogorov-Smirnov'] = {
                'statistic': np.nan,
                'p_value': 0,
                'is_normal': False,
                'test': 'Test failed'
            }
        
        # Anderson-Darling Test
        try:
            anderson_result = stats.anderson(data, dist='norm')
            critical_value = anderson_result.critical_values[2]  # 5% significance
            tests['Anderson-Darling'] = {
                'statistic': anderson_result.statistic,
                'critical_value': critical_value,
                'is_normal': anderson_result.statistic < critical_value,
                'test': 'Sensitive to tails'
            }
        except:
            tests['Anderson-Darling'] = {
                'statistic': np.nan,
                'critical_value': np.nan,
                'is_normal': False,
                'test': 'Test failed'
            }
        
        # Jarque-Bera Test
        try:
            jb_stat, jb_p = stats.jarque_bera(data)
            tests['Jarque-Bera'] = {
                'statistic': jb_stat,
                'p_value': jb_p,
                'is_normal': jb_p > self.alpha,
                'test': 'Tests skewness and kurtosis'
            }
        except:
            tests['Jarque-Bera'] = {
                'statistic': np.nan,
                'p_value': 0,
                'is_normal': False,
                'test': 'Test failed'
            }
        
        return tests
    
    def _visualize_normality(self, data, results):
        """Create normality visualization plots."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Normality Analysis: {results["data_name"]}', 
                    fontsize=14, fontweight='bold')
        
        # Histogram with KDE and normal curve
        sns.histplot(data, kde=True, ax=axes[0, 0], color='skyblue', 
                    edgecolor='black', stat='density')
        xmin, xmax = axes[0, 0].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        normal_pdf = stats.norm.pdf(x, np.mean(data), np.std(data))
        axes[0, 0].plot(x, normal_pdf, 'r-', lw=2, label='Normal PDF')
        axes[0, 0].axvline(np.mean(data), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(data):.2f}')
        axes[0, 0].set_title('Histogram with KDE & Normal PDF')
        axes[0, 0].legend()
        
        # Q-Q Plot
        stats.probplot(data, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot')
        axes[0, 1].get_lines()[0].set_markerfacecolor('blue')
        axes[0, 1].get_lines()[0].set_markeredgecolor('blue')
        axes[0, 1].get_lines()[1].set_color('red')
        
        # Box plot
        axes[0, 2].boxplot(data, vert=False)
        axes[0, 2].set_title('Box Plot')
        axes[0, 2].set_xlabel('Values')
        
        # ECDF vs Normal CDF
        sorted_data = np.sort(data)
        ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        normal_cdf = stats.norm.cdf(sorted_data, np.mean(data), np.std(data))
        axes[1, 0].plot(sorted_data, ecdf, 'b-', label='ECDF', lw=2)
        axes[1, 0].plot(sorted_data, normal_cdf, 'r--', label='Normal CDF', lw=2)
        axes[1, 0].set_title('ECDF vs Normal CDF')
        axes[1, 0].set_xlabel('Data')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # P-P Plot
        theoretical_probs = stats.norm.cdf(sorted_data, np.mean(data), np.std(data))
        axes[1, 1].scatter(theoretical_probs, ecdf, alpha=0.6, color='blue')
        axes[1, 1].plot([0, 1], [0, 1], 'r--', lw=2)
        axes[1, 1].set_title('P-P Plot')
        axes[1, 1].set_xlabel('Theoretical Probabilities')
        axes[1, 1].set_ylabel('Sample Probabilities')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Test results summary
        axes[1, 2].axis('off')
        conclusion = "NORMAL" if results['overall']['is_normal'] else "NOT NORMAL"
        color = 'green' if results['overall']['is_normal'] else 'red'
        
        summary_text = f"Sample Size: {results['sample_size']}\n"
        summary_text += f"Significance Level: α = {self.alpha}\n\n"
        summary_text += "Test Results:\n"
        
        for test_name, test_result in results['tests'].items():
            status = "✓ PASS" if test_result['is_normal'] else "✗ FAIL"
            p_val = test_result.get('p_value', 'N/A')
            if isinstance(p_val, float):
                p_val = f"{p_val:.6f}"
            summary_text += f"{test_name:20} {status:10} p={p_val}\n"
        
        summary_text += f"\nOverall: {conclusion}\n"
        summary_text += f"({results['overall']['confidence']})"
        
        axes[1, 2].text(0.1, 0.5, summary_text, fontfamily='monospace',
                       fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
