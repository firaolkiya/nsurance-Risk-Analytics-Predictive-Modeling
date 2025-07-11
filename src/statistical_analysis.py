"""
Statistical Analysis Module for Insurance Risk Hypothesis Testing
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class InsuranceHypothesisTesting:
    """
    Statistical hypothesis testing for insurance risk drivers
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with insurance data
        
        Args:
            df (pd.DataFrame): Insurance dataset
        """
        self.df = df
        self.results = {}
        
    def calculate_metrics(self) -> pd.DataFrame:
        """
        Calculate key metrics for analysis
        
        Returns:
            pd.DataFrame: DataFrame with calculated metrics
        """
        # Calculate claim frequency (proportion of policies with claims)
        self.df['HasClaim'] = (self.df['TotalClaims'] > 0).astype(int)
        
        # Calculate claim severity (average claim amount when claim occurs)
        self.df['ClaimSeverity'] = np.where(
            self.df['TotalClaims'] > 0,
            self.df['TotalClaims'],
            np.nan
        )
        
        # Calculate margin (profit)
        self.df['Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']
        
        # Calculate loss ratio
        self.df['LossRatio'] = self.df['TotalClaims'] / self.df['TotalPremium']
        
        return self.df
    
    def test_province_risk_differences(self) -> Dict:
        """
        Test H₀: There are no risk differences across provinces
        
        Returns:
            Dict: Test results
        """
        print("=" * 60)
        print("HYPOTHESIS TEST: Province Risk Differences")
        print("=" * 60)
        
        # Group by province and calculate metrics
        province_stats = self.df.groupby('Province').agg({
            'HasClaim': ['mean', 'count'],
            'ClaimSeverity': 'mean',
            'LossRatio': 'mean',
            'Margin': 'mean'
        }).round(4)
        
        print("Province Statistics:")
        print(province_stats)
        
        # Chi-square test for claim frequency
        contingency_table = pd.crosstab(self.df['Province'], self.df['HasClaim'])
        chi2, p_value_chi, dof, expected = chi2_contingency(contingency_table)
        
        # ANOVA for claim severity (excluding non-claims)
        claim_data = self.df[self.df['TotalClaims'] > 0]
        provinces = claim_data['Province'].unique()
        severity_groups = [claim_data[claim_data['Province'] == p]['ClaimSeverity'].values 
                          for p in provinces if len(claim_data[claim_data['Province'] == p]) > 0]
        
        f_stat, p_value_anova = stats.f_oneway(*severity_groups)
        
        result = {
            'test_type': 'Province Risk Differences',
            'chi2_statistic': chi2,
            'chi2_p_value': p_value_chi,
            'f_statistic': f_stat,
            'anova_p_value': p_value_anova,
            'reject_null': p_value_chi < 0.05 or p_value_anova < 0.05,
            'conclusion': 'Reject H₀' if (p_value_chi < 0.05 or p_value_anova < 0.05) else 'Fail to reject H₀'
        }
        
        print(f"\nChi-square test for claim frequency: χ² = {chi2:.4f}, p = {p_value_chi:.4f}")
        print(f"ANOVA test for claim severity: F = {f_stat:.4f}, p = {p_value_anova:.4f}")
        print(f"Conclusion: {result['conclusion']}")
        
        self.results['province'] = result
        return result
    
    def test_zipcode_risk_differences(self) -> Dict:
        """
        Test H₀: There are no risk differences between zip codes
        
        Returns:
            Dict: Test results
        """
        print("\n" + "=" * 60)
        print("HYPOTHESIS TEST: Zip Code Risk Differences")
        print("=" * 60)
        
        if 'PostalCode' not in self.df.columns:
            print("PostalCode column not found. Skipping zip code analysis.")
            return {}
        
        # Get top zip codes by policy count
        top_zipcodes = self.df['PostalCode'].value_counts().head(10).index
        
        # Filter data for top zip codes
        zip_data = self.df[self.df['PostalCode'].isin(top_zipcodes)]
        
        # Chi-square test for claim frequency
        contingency_table = pd.crosstab(zip_data['PostalCode'], zip_data['HasClaim'])
        chi2, p_value_chi, dof, expected = chi2_contingency(contingency_table)
        
        # ANOVA for claim severity
        claim_data = zip_data[zip_data['TotalClaims'] > 0]
        zipcodes = claim_data['PostalCode'].unique()
        severity_groups = [claim_data[claim_data['PostalCode'] == z]['ClaimSeverity'].values 
                          for z in zipcodes if len(claim_data[claim_data['PostalCode'] == z]) > 0]
        
        f_stat, p_value_anova = stats.f_oneway(*severity_groups) if len(severity_groups) > 1 else (0, 1)
        
        result = {
            'test_type': 'Zip Code Risk Differences',
            'chi2_statistic': chi2,
            'chi2_p_value': p_value_chi,
            'f_statistic': f_stat,
            'anova_p_value': p_value_anova,
            'reject_null': p_value_chi < 0.05 or p_value_anova < 0.05,
            'conclusion': 'Reject H₀' if (p_value_chi < 0.05 or p_value_anova < 0.05) else 'Fail to reject H₀'
        }
        
        print(f"Chi-square test for claim frequency: χ² = {chi2:.4f}, p = {p_value_chi:.4f}")
        print(f"ANOVA test for claim severity: F = {f_stat:.4f}, p = {p_value_anova:.4f}")
        print(f"Conclusion: {result['conclusion']}")
        
        self.results['zipcode'] = result
        return result
    
    def test_zipcode_margin_differences(self) -> Dict:
        """
        Test H₀: There are no significant margin (profit) differences between zip codes
        
        Returns:
            Dict: Test results
        """
        print("\n" + "=" * 60)
        print("HYPOTHESIS TEST: Zip Code Margin Differences")
        print("=" * 60)
        
        if 'PostalCode' not in self.df.columns:
            print("PostalCode column not found. Skipping zip code margin analysis.")
            return {}
        
        # Get top zip codes by policy count
        top_zipcodes = self.df['PostalCode'].value_counts().head(10).index
        
        # Filter data for top zip codes
        zip_data = self.df[self.df['PostalCode'].isin(top_zipcodes)]
        
        # ANOVA for margin differences
        zipcodes = zip_data['PostalCode'].unique()
        margin_groups = [zip_data[zip_data['PostalCode'] == z]['Margin'].values 
                        for z in zipcodes if len(zip_data[zip_data['PostalCode'] == z]) > 0]
        
        f_stat, p_value_anova = stats.f_oneway(*margin_groups) if len(margin_groups) > 1 else (0, 1)
        
        result = {
            'test_type': 'Zip Code Margin Differences',
            'f_statistic': f_stat,
            'anova_p_value': p_value_anova,
            'reject_null': p_value_anova < 0.05,
            'conclusion': 'Reject H₀' if p_value_anova < 0.05 else 'Fail to reject H₀'
        }
        
        print(f"ANOVA test for margin differences: F = {f_stat:.4f}, p = {p_value_anova:.4f}")
        print(f"Conclusion: {result['conclusion']}")
        
        self.results['zipcode_margin'] = result
        return result
    
    def test_gender_risk_differences(self) -> Dict:
        """
        Test H₀: There are no significant risk differences between Women and Men
        
        Returns:
            Dict: Test results
        """
        print("\n" + "=" * 60)
        print("HYPOTHESIS TEST: Gender Risk Differences")
        print("=" * 60)
        
        if 'Gender' not in self.df.columns:
            print("Gender column not found. Skipping gender analysis.")
            return {}
        
        # Filter for valid gender values
        gender_data = self.df[self.df['Gender'].isin(['M', 'F'])]
        
        # Chi-square test for claim frequency
        contingency_table = pd.crosstab(gender_data['Gender'], gender_data['HasClaim'])
        chi2, p_value_chi, dof, expected = chi2_contingency(contingency_table)
        
        # T-test for claim severity
        male_claims = gender_data[(gender_data['Gender'] == 'M') & (gender_data['TotalClaims'] > 0)]['ClaimSeverity']
        female_claims = gender_data[(gender_data['Gender'] == 'F') & (gender_data['TotalClaims'] > 0)]['ClaimSeverity']
        
        if len(male_claims) > 0 and len(female_claims) > 0:
            t_stat, p_value_t = ttest_ind(male_claims, female_claims, equal_var=False)
        else:
            t_stat, p_value_t = 0, 1
        
        # Mann-Whitney U test for margin differences
        male_margin = gender_data[gender_data['Gender'] == 'M']['Margin']
        female_margin = gender_data[gender_data['Gender'] == 'F']['Margin']
        
        u_stat, p_value_u = mannwhitneyu(male_margin, female_margin, alternative='two-sided')
        
        result = {
            'test_type': 'Gender Risk Differences',
            'chi2_statistic': chi2,
            'chi2_p_value': p_value_chi,
            't_statistic': t_stat,
            't_p_value': p_value_t,
            'u_statistic': u_stat,
            'u_p_value': p_value_u,
            'reject_null': p_value_chi < 0.05 or p_value_t < 0.05 or p_value_u < 0.05,
            'conclusion': 'Reject H₀' if (p_value_chi < 0.05 or p_value_t < 0.05 or p_value_u < 0.05) else 'Fail to reject H₀'
        }
        
        print(f"Chi-square test for claim frequency: χ² = {chi2:.4f}, p = {p_value_chi:.4f}")
        print(f"T-test for claim severity: t = {t_stat:.4f}, p = {p_value_t:.4f}")
        print(f"Mann-Whitney U test for margin: U = {u_stat:.4f}, p = {p_value_u:.4f}")
        print(f"Conclusion: {result['conclusion']}")
        
        self.results['gender'] = result
        return result
    
    def create_visualizations(self) -> None:
        """
        Create visualizations for hypothesis testing results
        """
        print("\n" + "=" * 60)
        print("CREATING HYPOTHESIS TESTING VISUALIZATIONS")
        print("=" * 60)
        
        # Set up plotting style
        plt.rcParams['figure.figsize'] = (15, 10)
        
        # 1. Province risk comparison
        if 'Province' in self.df.columns:
            self._plot_province_comparison()
        
        # 2. Gender comparison
        if 'Gender' in self.df.columns:
            self._plot_gender_comparison()
        
        # 3. Zip code comparison (if available)
        if 'PostalCode' in self.df.columns:
            self._plot_zipcode_comparison()
    
    def _plot_province_comparison(self):
        """Plot province risk comparison"""
        province_stats = self.df.groupby('Province').agg({
            'HasClaim': 'mean',
            'ClaimSeverity': 'mean',
            'LossRatio': 'mean'
        }).round(4)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Claim frequency by province
        province_stats['HasClaim'].sort_values(ascending=False).head(10).plot(
            kind='bar', ax=axes[0], color='skyblue')
        axes[0].set_title('Claim Frequency by Province', fontweight='bold')
        axes[0].set_ylabel('Claim Frequency')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Claim severity by province
        province_stats['ClaimSeverity'].sort_values(ascending=False).head(10).plot(
            kind='bar', ax=axes[1], color='lightcoral')
        axes[1].set_title('Claim Severity by Province', fontweight='bold')
        axes[1].set_ylabel('Average Claim Amount')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Loss ratio by province
        province_stats['LossRatio'].sort_values(ascending=False).head(10).plot(
            kind='bar', ax=axes[2], color='lightgreen')
        axes[2].set_title('Loss Ratio by Province', fontweight='bold')
        axes[2].set_ylabel('Loss Ratio')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('plots/province_hypothesis_testing.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_gender_comparison(self):
        """Plot gender risk comparison"""
        gender_stats = self.df[self.df['Gender'].isin(['M', 'F'])].groupby('Gender').agg({
            'HasClaim': 'mean',
            'ClaimSeverity': 'mean',
            'Margin': 'mean'
        }).round(4)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Claim frequency by gender
        gender_stats['HasClaim'].plot(kind='bar', ax=axes[0], color=['lightblue', 'pink'])
        axes[0].set_title('Claim Frequency by Gender', fontweight='bold')
        axes[0].set_ylabel('Claim Frequency')
        
        # Claim severity by gender
        gender_stats['ClaimSeverity'].plot(kind='bar', ax=axes[1], color=['lightblue', 'pink'])
        axes[1].set_title('Claim Severity by Gender', fontweight='bold')
        axes[1].set_ylabel('Average Claim Amount')
        
        # Margin by gender
        gender_stats['Margin'].plot(kind='bar', ax=axes[2], color=['lightblue', 'pink'])
        axes[2].set_title('Margin by Gender', fontweight='bold')
        axes[2].set_ylabel('Average Margin')
        
        plt.tight_layout()
        plt.savefig('plots/gender_hypothesis_testing.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_zipcode_comparison(self):
        """Plot zip code comparison"""
        top_zipcodes = self.df['PostalCode'].value_counts().head(10).index
        zip_data = self.df[self.df['PostalCode'].isin(top_zipcodes)]
        
        zip_stats = zip_data.groupby('PostalCode').agg({
            'HasClaim': 'mean',
            'Margin': 'mean'
        }).round(4)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Claim frequency by zip code
        zip_stats['HasClaim'].plot(kind='bar', ax=axes[0], color='orange')
        axes[0].set_title('Claim Frequency by Zip Code', fontweight='bold')
        axes[0].set_ylabel('Claim Frequency')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Margin by zip code
        zip_stats['Margin'].plot(kind='bar', ax=axes[1], color='purple')
        axes[1].set_title('Margin by Zip Code', fontweight='bold')
        axes[1].set_ylabel('Average Margin')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('plots/zipcode_hypothesis_testing.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self) -> str:
        """
        Generate comprehensive hypothesis testing report
        
        Returns:
            str: Report file path
        """
        print("\n" + "=" * 60)
        print("GENERATING HYPOTHESIS TESTING REPORT")
        print("=" * 60)
        
        # Create reports directory if it doesn't exist
        import os
        reports_dir = 'reports'
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        
        report_path = f"{reports_dir}/hypothesis_testing_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("INSURANCE ANALYTICS - HYPOTHESIS TESTING REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 30 + "\n")
            
            rejected_hypotheses = sum(1 for result in self.results.values() if result.get('reject_null', False))
            total_tests = len(self.results)
            
            f.write(f"Total hypotheses tested: {total_tests}\n")
            f.write(f"Rejected null hypotheses: {rejected_hypotheses}\n")
            f.write(f"Failed to reject null hypotheses: {total_tests - rejected_hypotheses}\n\n")
            
            f.write("DETAILED RESULTS\n")
            f.write("-" * 30 + "\n")
            
            for test_name, result in self.results.items():
                f.write(f"\n{result['test_type']}:\n")
                f.write(f"  Conclusion: {result['conclusion']}\n")
                
                if 'chi2_p_value' in result:
                    f.write(f"  Chi-square p-value: {result['chi2_p_value']:.4f}\n")
                if 'anova_p_value' in result:
                    f.write(f"  ANOVA p-value: {result['anova_p_value']:.4f}\n")
                if 't_p_value' in result:
                    f.write(f"  T-test p-value: {result['t_p_value']:.4f}\n")
                if 'u_p_value' in result:
                    f.write(f"  Mann-Whitney U p-value: {result['u_p_value']:.4f}\n")
            
            f.write("\nBUSINESS RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            
            for test_name, result in self.results.items():
                if result.get('reject_null', False):
                    f.write(f"• {test_name}: Consider adjusting pricing or risk assessment strategies\n")
                else:
                    f.write(f"• {test_name}: No significant differences detected, current approach may be appropriate\n")
        
        print(f"Report saved to: {report_path}")
        return report_path
    
    def run_all_tests(self) -> Dict:
        """
        Run all hypothesis tests
        
        Returns:
            Dict: All test results
        """
        print("INSURANCE HYPOTHESIS TESTING")
        print("=" * 60)
        
        # Calculate metrics
        self.calculate_metrics()
        
        # Run all tests
        self.test_province_risk_differences()
        self.test_zipcode_risk_differences()
        self.test_zipcode_margin_differences()
        self.test_gender_risk_differences()
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate report
        self.generate_report()
        
        return self.results

def main():
    """Main function to run hypothesis testing"""
    from eda import InsuranceEDA
    
    # Load data
    eda = InsuranceEDA("data/MachineLearningRating_v3.txt")
    df = eda.load_data()
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Run hypothesis testing
    hypothesis_tester = InsuranceHypothesisTesting(df)
    results = hypothesis_tester.run_all_tests()
    
    print("\n" + "=" * 60)
    print("HYPOTHESIS TESTING COMPLETE")
    print("=" * 60)
    print("Check the 'plots' directory for visualizations")
    print("Check the 'reports' directory for detailed report")

if __name__ == "__main__":
    main() 