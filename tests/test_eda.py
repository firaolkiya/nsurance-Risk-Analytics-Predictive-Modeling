"""
Unit tests for the EDA module
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src directory to path
sys.path.append('../src')
from eda import InsuranceEDA

class TestInsuranceEDA:
    """Test class for InsuranceEDA"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create a sample dataset for testing
        self.sample_data = pd.DataFrame({
            'PolicyID': range(1, 101),
            'TotalPremium': np.random.uniform(1000, 5000, 100),
            'TotalClaims': np.random.uniform(0, 2000, 100),
            'Province': np.random.choice(['Province1', 'Province2', 'Province3'], 100),
            'VehicleType': np.random.choice(['Sedan', 'SUV', 'Truck'], 100),
            'Make': np.random.choice(['Toyota', 'Honda', 'Ford'], 100),
            'Gender': np.random.choice(['M', 'F'], 100),
            'TransactionMonth': pd.date_range('2014-02-01', periods=100, freq='M')
        })
        
        # Save sample data to temporary file
        self.temp_file = 'temp_test_data.csv'
        self.sample_data.to_csv(self.temp_file, index=False)
        
        # Initialize EDA with test data
        self.eda = InsuranceEDA(self.temp_file)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
    
    def test_load_data(self):
        """Test data loading functionality"""
        df = self.eda.load_data()
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert len(df.columns) == 8
    
    def test_basic_info(self):
        """Test basic info generation"""
        self.eda.load_data()
        info = self.eda.basic_info()
        
        assert 'shape' in info
        assert 'columns' in info
        assert 'dtypes' in info
        assert info['shape'] == (100, 8)
    
    def test_data_quality_assessment(self):
        """Test data quality assessment"""
        self.eda.load_data()
        quality_report = self.eda.data_quality_assessment()
        
        assert isinstance(quality_report, pd.DataFrame)
        assert 'Missing_Count' in quality_report.columns
        assert 'Missing_Percentage' in quality_report.columns
        assert 'Unique_Values' in quality_report.columns
    
    def test_descriptive_statistics(self):
        """Test descriptive statistics calculation"""
        self.eda.load_data()
        stats = self.eda.descriptive_statistics()
        
        assert isinstance(stats, pd.DataFrame)
        assert len(stats) > 0
    
    def test_analyze_loss_ratio(self):
        """Test loss ratio analysis"""
        self.eda.load_data()
        results = self.eda.analyze_loss_ratio()
        
        assert isinstance(results, dict)
        assert 'overall' in results
        assert isinstance(results['overall'], (int, float))
    
    def test_temporal_analysis(self):
        """Test temporal analysis"""
        self.eda.load_data()
        results = self.eda.temporal_analysis()
        
        assert isinstance(results, dict)
    
    def test_vehicle_analysis(self):
        """Test vehicle analysis"""
        self.eda.load_data()
        results = self.eda.vehicle_analysis()
        
        assert isinstance(results, dict)
    
    def test_outlier_detection(self):
        """Test outlier detection"""
        self.eda.load_data()
        results = self.eda.outlier_detection()
        
        assert isinstance(results, dict)
        for col in results:
            assert 'outlier_count' in results[col]
            assert 'outlier_percentage' in results[col]
    
    def test_create_visualizations(self):
        """Test visualization creation"""
        self.eda.load_data()
        
        # Create plots directory if it doesn't exist
        if not os.path.exists('plots'):
            os.makedirs('plots')
        
        self.eda.create_visualizations()
        
        # Check if plots were created
        plot_files = os.listdir('plots')
        assert len(plot_files) > 0
    
    def test_generate_report(self):
        """Test report generation"""
        self.eda.load_data()
        
        # Create reports directory if it doesn't exist
        if not os.path.exists('reports'):
            os.makedirs('reports')
        
        report_path = self.eda.generate_report()
        
        assert isinstance(report_path, str)
        assert os.path.exists(report_path)
    
    def test_missing_data_handling(self):
        """Test handling of missing data"""
        # Create data with missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0:10, 'TotalClaims'] = np.nan
        
        temp_file = 'temp_missing_data.csv'
        data_with_missing.to_csv(temp_file, index=False)
        
        eda_missing = InsuranceEDA(temp_file)
        eda_missing.load_data()
        
        quality_report = eda_missing.data_quality_assessment()
        
        # Check that missing values are detected
        missing_counts = quality_report['Missing_Count']
        assert missing_counts['TotalClaims'] > 0
        
        os.remove(temp_file)
    
    def test_invalid_data_path(self):
        """Test handling of invalid data path"""
        eda_invalid = InsuranceEDA('nonexistent_file.txt')
        df = eda_invalid.load_data()
        
        assert df is None

def test_eda_class_initialization():
    """Test EDA class initialization"""
    eda = InsuranceEDA('test_path.txt')
    
    assert eda.data_path == 'test_path.txt'
    assert eda.df is None
    assert eda.plots_dir == 'plots'

if __name__ == "__main__":
    pytest.main([__file__]) 