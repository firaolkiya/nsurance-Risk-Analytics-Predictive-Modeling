"""
Exploratory Data Analysis (EDA) Module for Insurance Analytics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Optional
import os

warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class InsuranceEDA:
    """
    Comprehensive EDA class for insurance data analysis
    """
    
    def __init__(self, data_path: str):
        """
        Initialize EDA with data path
        
        Args:
            data_path (str): Path to the data file
        """
        self.data_path = data_path
        self.df = None
        self.plots_dir = 'plots'
        
        # Create plots directory if it doesn't exist
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
    
    def load_data(self) -> pd.DataFrame:
        """
        Load and preprocess the insurance data
        
        Returns:
            pd.DataFrame: Loaded and preprocessed dataframe
        """
        print("Loading data...")
        
        # Try different separators
        separators = ['\t', ',', ';', '|']
        
        for sep in separators:
            try:
                df = pd.read_csv(self.data_path, sep=sep, low_memory=False)
                if len(df.columns) > 1:
                    print(f"Successfully loaded data with separator: '{sep}'")
                    self.df = df
                    return df
            except Exception as e:
                continue
        
        # If no separator worked, try reading as fixed-width
        try:
            df = pd.read_fwf(self.data_path)
            print("Successfully loaded data as fixed-width format")
            self.df = df
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def basic_info(self) -> Dict:
        """
        Get basic information about the dataset
        
        Returns:
            Dict: Basic dataset information
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return {}
        
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum()
        }
        
        print("=" * 60)
        print("BASIC DATASET INFORMATION")
        print("=" * 60)
        print(f"Dataset Shape: {info['shape']}")
        print(f"Memory Usage: {info['memory_usage'] / 1024**2:.2f} MB")
        print(f"Number of Columns: {len(info['columns'])}")
        
        return info
    
    def data_quality_assessment(self) -> pd.DataFrame:
        """
        Perform comprehensive data quality assessment
        
        Returns:
            pd.DataFrame: Data quality report
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return pd.DataFrame()
        
        print("\n" + "=" * 60)
        print("DATA QUALITY ASSESSMENT")
        print("=" * 60)
        
        # Calculate missing values
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        # Calculate unique values for each column
        unique_counts = self.df.nunique()
        
        # Create quality report
        quality_report = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percentage': missing_percent,
            'Unique_Values': unique_counts,
            'Data_Type': self.df.dtypes
        })
        
        # Add quality flags
        quality_report['Has_Missing'] = quality_report['Missing_Count'] > 0
        quality_report['High_Cardinality'] = quality_report['Unique_Values'] > 50
        
        print("Data Quality Summary:")
        print(f"Total rows: {len(self.df)}")
        print(f"Total columns: {len(self.df.columns)}")
        print(f"Columns with missing values: {quality_report['Has_Missing'].sum()}")
        print(f"High cardinality columns: {quality_report['High_Cardinality'].sum()}")
        
        return quality_report
    
    def descriptive_statistics(self) -> pd.DataFrame:
        """
        Calculate descriptive statistics for numerical columns
        
        Returns:
            pd.DataFrame: Descriptive statistics
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return pd.DataFrame()
        
        print("\n" + "=" * 60)
        print("DESCRIPTIVE STATISTICS")
        print("=" * 60)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            print("No numerical columns found.")
            return pd.DataFrame()
        
        stats = self.df[numerical_cols].describe()
        
        # Add additional statistics
        stats.loc['skewness'] = self.df[numerical_cols].skew()
        stats.loc['kurtosis'] = self.df[numerical_cols].kurtosis()
        stats.loc['missing'] = self.df[numerical_cols].isnull().sum()
        
        print("Numerical Columns Statistics:")
        print(stats)
        
        return stats
    
    def analyze_loss_ratio(self) -> Dict:
        """
        Analyze Loss Ratio (TotalClaims / TotalPremium)
        
        Returns:
            Dict: Loss ratio analysis results
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return {}
        
        print("\n" + "=" * 60)
        print("LOSS RATIO ANALYSIS")
        print("=" * 60)
        
        # Check if required columns exist
        required_cols = ['TotalClaims', 'TotalPremium']
        if not all(col in self.df.columns for col in required_cols):
            print(f"Missing required columns: {required_cols}")
            return {}
        
        # Calculate overall loss ratio
        total_claims = self.df['TotalClaims'].sum()
        total_premium = self.df['TotalPremium'].sum()
        overall_loss_ratio = total_claims / total_premium if total_premium > 0 else 0
        
        print(f"Overall Loss Ratio: {overall_loss_ratio:.4f} ({overall_loss_ratio*100:.2f}%)")
        print(f"Total Claims: {total_claims:,.2f}")
        print(f"Total Premium: {total_premium:,.2f}")
        
        # Analyze by different dimensions
        dimensions = ['Province', 'VehicleType', 'Gender']
        results = {'overall': overall_loss_ratio}
        
        for dim in dimensions:
            if dim in self.df.columns:
                print(f"\nLoss Ratio by {dim}:")
                dim_analysis = self.df.groupby(dim).agg({
                    'TotalClaims': 'sum',
                    'TotalPremium': 'sum'
                }).assign(
                    LossRatio=lambda x: x['TotalClaims'] / x['TotalPremium']
                ).sort_values('LossRatio', ascending=False)
                
                print(dim_analysis.head(10))
                results[dim] = dim_analysis
        
        return results
    
    def temporal_analysis(self) -> Dict:
        """
        Analyze temporal trends in the data
        
        Returns:
            Dict: Temporal analysis results
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return {}
        
        print("\n" + "=" * 60)
        print("TEMPORAL ANALYSIS")
        print("=" * 60)
        
        # Check for date columns
        date_columns = [col for col in self.df.columns if 'date' in col.lower() or 'month' in col.lower()]
        
        if 'TransactionMonth' in self.df.columns:
            print("Analyzing trends by TransactionMonth...")
            
            # Convert to datetime if possible
            try:
                self.df['TransactionMonth'] = pd.to_datetime(self.df['TransactionMonth'])
                
                # Monthly trends
                monthly_stats = self.df.groupby(self.df['TransactionMonth'].dt.to_period('M')).agg({
                    'TotalClaims': 'sum',
                    'TotalPremium': 'sum',
                    'PolicyID': 'count'
                }).assign(
                    LossRatio=lambda x: x['TotalClaims'] / x['TotalPremium'],
                    ClaimFrequency=lambda x: x['TotalClaims'] / x['PolicyID']
                )
                
                print("Monthly Statistics:")
                print(monthly_stats)
                
                return {'monthly_stats': monthly_stats}
                
            except Exception as e:
                print(f"Error processing dates: {e}")
        
        return {}
    
    def vehicle_analysis(self) -> Dict:
        """
        Analyze vehicle-related patterns
        
        Returns:
            Dict: Vehicle analysis results
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return {}
        
        print("\n" + "=" * 60)
        print("VEHICLE ANALYSIS")
        print("=" * 60)
        
        vehicle_cols = [col for col in self.df.columns if any(word in col.lower() 
                    for word in ['make', 'model', 'vehicle', 'car', 'type'])]
        
        print(f"Vehicle-related columns: {vehicle_cols}")
        
        results = {}
        
        # Analyze by vehicle make
        if 'Make' in self.df.columns:
            make_analysis = self.df.groupby('Make').agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum',
                'PolicyID': 'count'
            }).assign(
                LossRatio=lambda x: x['TotalClaims'] / x['TotalPremium'],
                AvgClaimAmount=lambda x: x['TotalClaims'] / x['PolicyID']
            ).sort_values('LossRatio', ascending=False)
            
            print("\nTop 10 Makes by Loss Ratio:")
            print(make_analysis.head(10))
            
            results['make_analysis'] = make_analysis
        
        # Analyze by vehicle type
        if 'VehicleType' in self.df.columns:
            type_analysis = self.df.groupby('VehicleType').agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum',
                'PolicyID': 'count'
            }).assign(
                LossRatio=lambda x: x['TotalClaims'] / x['TotalPremium']
            ).sort_values('LossRatio', ascending=False)
            
            print("\nVehicle Types by Loss Ratio:")
            print(type_analysis)
            
            results['type_analysis'] = type_analysis
        
        return results
    
    def outlier_detection(self) -> Dict:
        """
        Detect outliers in numerical columns using IQR method
        
        Returns:
            Dict: Outlier analysis results
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return {}
        
        print("\n" + "=" * 60)
        print("OUTLIER DETECTION")
        print("=" * 60)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_results = {}
        
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            
            outlier_results[col] = {
                'outlier_count': len(outliers),
                'outlier_percentage': len(outliers) / len(self.df) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outliers': outliers
            }
            
            print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(self.df)*100:.2f}%)")
        
        return outlier_results
    
    def create_visualizations(self) -> None:
        """
        Create comprehensive visualizations for the analysis
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return
        
        print("\n" + "=" * 60)
        print("CREATING VISUALIZATIONS")
        print("=" * 60)
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # 1. Loss Ratio Distribution
        self._plot_loss_ratio_distribution()
        
        # 2. Temporal Trends
        self._plot_temporal_trends()
        
        # 3. Geographic Analysis
        self._plot_geographic_analysis()
        
        # 4. Vehicle Analysis
        self._plot_vehicle_analysis()
        
        # 5. Correlation Matrix
        self._plot_correlation_matrix()
        
        print(f"All plots saved to '{self.plots_dir}' directory")
    
    def _plot_loss_ratio_distribution(self):
        """Plot loss ratio distribution"""
        if 'TotalClaims' in self.df.columns and 'TotalPremium' in self.df.columns:
            # Calculate individual loss ratios
            loss_ratios = self.df['TotalClaims'] / self.df['TotalPremium']
            loss_ratios = loss_ratios[loss_ratios > 0]  # Remove zero/negative values
            
            plt.figure(figsize=(12, 8))
            plt.hist(loss_ratios, bins=50, alpha=0.7, edgecolor='black')
            plt.title('Distribution of Loss Ratios', fontsize=16, fontweight='bold')
            plt.xlabel('Loss Ratio (Claims/Premium)', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.axvline(loss_ratios.mean(), color='red', linestyle='--', 
                       label=f'Mean: {loss_ratios.mean():.3f}')
            plt.axvline(loss_ratios.median(), color='green', linestyle='--', 
                       label=f'Median: {loss_ratios.median():.3f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{self.plots_dir}/loss_ratio_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_temporal_trends(self):
        """Plot temporal trends"""
        if 'TransactionMonth' in self.df.columns:
            try:
                monthly_data = self.df.groupby(pd.to_datetime(self.df['TransactionMonth']).dt.to_period('M')).agg({
                    'TotalClaims': 'sum',
                    'TotalPremium': 'sum'
                }).assign(LossRatio=lambda x: x['TotalClaims'] / x['TotalPremium'])
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
                
                # Premium and Claims over time
                monthly_data[['TotalPremium', 'TotalClaims']].plot(ax=ax1, marker='o')
                ax1.set_title('Monthly Premium and Claims Trends', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Amount', fontsize=12)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Loss ratio over time
                monthly_data['LossRatio'].plot(ax=ax2, marker='o', color='red')
                ax2.set_title('Monthly Loss Ratio Trends', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Loss Ratio', fontsize=12)
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f'{self.plots_dir}/temporal_trends.png', dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"Error plotting temporal trends: {e}")
    
    def _plot_geographic_analysis(self):
        """Plot geographic analysis"""
        if 'Province' in self.df.columns:
            province_analysis = self.df.groupby('Province').agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum'
            }).assign(LossRatio=lambda x: x['TotalClaims'] / x['TotalPremium'])
            
            plt.figure(figsize=(12, 8))
            province_analysis['LossRatio'].sort_values(ascending=False).head(15).plot(kind='bar')
            plt.title('Loss Ratio by Province (Top 15)', fontsize=16, fontweight='bold')
            plt.xlabel('Province', fontsize=12)
            plt.ylabel('Loss Ratio', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{self.plots_dir}/geographic_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_vehicle_analysis(self):
        """Plot vehicle analysis"""
        if 'Make' in self.df.columns:
            make_analysis = self.df.groupby('Make').agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum'
            }).assign(LossRatio=lambda x: x['TotalClaims'] / x['TotalPremium'])
            
            plt.figure(figsize=(12, 8))
            make_analysis['LossRatio'].sort_values(ascending=False).head(15).plot(kind='bar')
            plt.title('Loss Ratio by Vehicle Make (Top 15)', fontsize=16, fontweight='bold')
            plt.xlabel('Vehicle Make', fontsize=12)
            plt.ylabel('Loss Ratio', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{self.plots_dir}/vehicle_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_correlation_matrix(self):
        """Plot correlation matrix for numerical variables"""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 1:
            correlation_matrix = self.df[numerical_cols].corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, fmt='.2f')
            plt.title('Correlation Matrix of Numerical Variables', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{self.plots_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive EDA report
        
        Returns:
            str: Report file path
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return ""
        
        print("\n" + "=" * 60)
        print("GENERATING EDA REPORT")
        print("=" * 60)
        
        # Create reports directory if it doesn't exist
        reports_dir = 'reports'
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        
        report_path = f"{reports_dir}/eda_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("INSURANCE ANALYTICS - EDA REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Basic information
            f.write("1. DATASET OVERVIEW\n")
            f.write("-" * 30 + "\n")
            f.write(f"Dataset Shape: {self.df.shape}\n")
            f.write(f"Total Records: {len(self.df):,}\n")
            f.write(f"Total Columns: {len(self.df.columns)}\n\n")
            
            # Data quality
            quality_report = self.data_quality_assessment()
            f.write("2. DATA QUALITY ASSESSMENT\n")
            f.write("-" * 30 + "\n")
            f.write(f"Columns with missing values: {quality_report['Has_Missing'].sum()}\n")
            f.write(f"High cardinality columns: {quality_report['High_Cardinality'].sum()}\n\n")
            
            # Loss ratio analysis
            loss_ratio_results = self.analyze_loss_ratio()
            if loss_ratio_results:
                f.write("3. LOSS RATIO ANALYSIS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Overall Loss Ratio: {loss_ratio_results.get('overall', 0):.4f}\n\n")
            
            # Key insights
            f.write("4. KEY INSIGHTS\n")
            f.write("-" * 30 + "\n")
            f.write("• Analysis of risk patterns across different dimensions\n")
            f.write("• Identification of high-risk vehicle makes and provinces\n")
            f.write("• Temporal trends in claims and premiums\n")
            f.write("• Outlier detection in financial variables\n\n")
            
            f.write("5. RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            f.write("• Focus on high-loss-ratio provinces for risk management\n")
            f.write("• Investigate vehicle makes with high claim rates\n")
            f.write("• Monitor temporal trends for seasonal patterns\n")
            f.write("• Address data quality issues for better analysis\n")
        
        print(f"Report saved to: {report_path}")
        return report_path

def main():
    """Main function to run the EDA analysis"""
    # Initialize EDA
    eda = InsuranceEDA("data/MachineLearningRating_v3.txt")
    
    # Load data
    df = eda.load_data()
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Perform comprehensive analysis
    eda.basic_info()
    eda.data_quality_assessment()
    eda.descriptive_statistics()
    eda.analyze_loss_ratio()
    eda.temporal_analysis()
    eda.vehicle_analysis()
    eda.outlier_detection()
    
    # Create visualizations
    eda.create_visualizations()
    
    # Generate report
    eda.generate_report()
    
    print("\nEDA Analysis Complete!")

if __name__ == "__main__":
    main() 