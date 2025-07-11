"""
Main entry point for Insurance Analytics Project
"""

import sys
import os
from eda import InsuranceEDA
from statistical_analysis import InsuranceHypothesisTesting

def main():
    """
    Main function to run the insurance analytics pipeline
    """
    print("=" * 60)
    print("INSURANCE ANALYTICS PROJECT")
    print("=" * 60)
    print("Task 3: Statistical Hypothesis Testing")
    print("=" * 60)
    
    # Check if data file exists
    data_path = "data/MachineLearningRating_v3.txt"
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the data file is in the data/ directory")
        return
    
    # Initialize and run EDA
    try:
        eda = InsuranceEDA(data_path)
        
        # Load data
        print("\n1. Loading and preprocessing data...")
        df = eda.load_data()
        if df is None:
            print("Failed to load data. Please check the file format.")
            return
        
        # Perform comprehensive EDA
        print("\n2. Performing basic data exploration...")
        eda.basic_info()
        
        print("\n3. Assessing data quality...")
        eda.data_quality_assessment()
        
        print("\n4. Calculating descriptive statistics...")
        eda.descriptive_statistics()
        
        print("\n5. Analyzing loss ratios...")
        eda.analyze_loss_ratio()
        
        print("\n6. Analyzing temporal trends...")
        eda.temporal_analysis()
        
        print("\n7. Analyzing vehicle patterns...")
        eda.vehicle_analysis()
        
        print("\n8. Detecting outliers...")
        eda.outlier_detection()
        
        print("\n9. Creating visualizations...")
        eda.create_visualizations()
        
        print("\n10. Generating comprehensive report...")
        report_path = eda.generate_report()
        
        # Run hypothesis testing
        print("\n11. Running hypothesis tests...")
        hypothesis_tester = InsuranceHypothesisTesting(df)
        results = hypothesis_tester.run_all_tests()
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"✓ EDA analysis completed successfully")
        print(f"✓ Hypothesis testing completed")
        print(f"✓ Visualizations saved to 'plots/' directory")
        print(f"✓ Reports generated: {report_path}")
        print("\nNext steps:")
        print("1. Review the generated plots and reports")
        print("2. Commit your changes to git")
        print("3. Push to your GitHub repository")
        print("4. Continue with advanced modeling")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return

if __name__ == "__main__":
    main()