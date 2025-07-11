"""
Predictive Modeling Module for Insurance Risk-Based Pricing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Dict, List, Tuple, Optional
import warnings
import os

warnings.filterwarnings('ignore')

class InsurancePredictiveModeling:
    """
    Predictive modeling for insurance risk-based pricing
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with insurance data
        
        Args:
            df (pd.DataFrame): Insurance dataset
        """
        self.df = df.copy()
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
    def prepare_data(self) -> pd.DataFrame:
        """
        Prepare data for modeling
        
        Returns:
            pd.DataFrame: Prepared dataset
        """
        print("=" * 60)
        print("DATA PREPARATION")
        print("=" * 60)
        
        # Create target variables
        self.df['HasClaim'] = (self.df['TotalClaims'] > 0).astype(int)
        self.df['ClaimSeverity'] = np.where(
            self.df['TotalClaims'] > 0,
            self.df['TotalClaims'],
            np.nan
        )
        
        # Feature engineering
        self._engineer_features()
        
        # Handle missing values
        self._handle_missing_values()
        
        # Encode categorical variables
        self._encode_categorical()
        
        print(f"Data shape after preparation: {self.df.shape}")
        print(f"Features available: {list(self.df.columns)}")
        
        return self.df
    
    def _engineer_features(self):
        """Create new features"""
        # Age-related features
        if 'RegistrationYear' in self.df.columns:
            self.df['VehicleAge'] = 2024 - self.df['RegistrationYear']
        
        # Financial ratios
        if 'TotalPremium' in self.df.columns and 'TotalClaims' in self.df.columns:
            self.df['LossRatio'] = self.df['TotalClaims'] / self.df['TotalPremium']
            self.df['Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']
        
        # Risk indicators
        if 'SumInsured' in self.df.columns:
            self.df['PremiumToSumInsuredRatio'] = self.df['TotalPremium'] / self.df['SumInsured']
        
        print("Feature engineering completed")
    
    def _handle_missing_values(self):
        """Handle missing values"""
        # Numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        
        # Categorical columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        self.df[categorical_cols] = self.df[categorical_cols].fillna('Unknown')
        
        print(f"Missing values handled. Remaining missing: {self.df.isnull().sum().sum()}")
    
    def _encode_categorical(self):
        """Encode categorical variables"""
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if self.df[col].nunique() < 50:  # Only encode if not too many categories
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
        
        print(f"Categorical encoding completed for {len(categorical_cols)} columns")
    
    def build_claim_severity_model(self) -> Dict:
        """
        Build model to predict claim severity (TotalClaims for policies with claims)
        
        Returns:
            Dict: Model results
        """
        print("\n" + "=" * 60)
        print("CLAIM SEVERITY PREDICTION MODEL")
        print("=" * 60)
        
        # Filter for policies with claims
        claim_data = self.df[self.df['TotalClaims'] > 0].copy()
        
        if len(claim_data) == 0:
            print("No claim data available")
            return {}
        
        # Select features (exclude target and derived variables)
        exclude_cols = ['TotalClaims', 'HasClaim', 'ClaimSeverity', 'LossRatio', 'Margin']
        feature_cols = [col for col in claim_data.columns if col not in exclude_cols and claim_data[col].dtype in ['int64', 'float64']]
        
        X = claim_data[feature_cols]
        y = claim_data['TotalClaims']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Evaluate
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred,
                'actual': y_test
            }
            
            print(f"{name} - RMSE: {rmse:.2f}, R²: {r2:.4f}")
        
        # Store best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        self.models['claim_severity'] = results[best_model_name]['model']
        self.results['claim_severity'] = results
        
        return results
    
    def build_claim_probability_model(self) -> Dict:
        """
        Build model to predict claim probability (binary classification)
        
        Returns:
            Dict: Model results
        """
        print("\n" + "=" * 60)
        print("CLAIM PROBABILITY PREDICTION MODEL")
        print("=" * 60)
        
        # Select features
        exclude_cols = ['TotalClaims', 'HasClaim', 'ClaimSeverity', 'LossRatio', 'Margin']
        feature_cols = [col for col in self.df.columns if col not in exclude_cols and self.df[col].dtype in ['int64', 'float64']]
        
        X = self.df[feature_cols]
        y = self.df['HasClaim']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'actual': y_test
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}")
            print(classification_report(y_test, y_pred))
        
        # Store best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        self.models['claim_probability'] = results[best_model_name]['model']
        self.results['claim_probability'] = results
        
        return results
    
    def build_premium_model(self) -> Dict:
        """
        Build model to predict appropriate premium
        
        Returns:
            Dict: Model results
        """
        print("\n" + "=" * 60)
        print("PREMIUM PREDICTION MODEL")
        print("=" * 60)
        
        # Use CalculatedPremiumPerTerm as target if available, otherwise TotalPremium
        target_col = 'CalculatedPremiumPerTerm' if 'CalculatedPremiumPerTerm' in self.df.columns else 'TotalPremium'
        
        # Select features
        exclude_cols = ['TotalClaims', 'HasClaim', 'ClaimSeverity', 'LossRatio', 'Margin', 'TotalPremium', 'CalculatedPremiumPerTerm']
        feature_cols = [col for col in self.df.columns if col not in exclude_cols and self.df[col].dtype in ['int64', 'float64']]
        
        X = self.df[feature_cols]
        y = self.df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Evaluate
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred,
                'actual': y_test
            }
            
            print(f"{name} - RMSE: {rmse:.2f}, R²: {r2:.4f}")
        
        # Store best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        self.models['premium'] = results[best_model_name]['model']
        self.results['premium'] = results
        
        return results
    
    def analyze_feature_importance(self, model_type: str = 'claim_severity'):
        """
        Analyze feature importance using SHAP
        
        Args:
            model_type (str): Type of model to analyze
        """
        print("\n" + "=" * 60)
        print(f"FEATURE IMPORTANCE ANALYSIS - {model_type.upper()}")
        print("=" * 60)
        
        if model_type not in self.models:
            print(f"No model found for {model_type}")
            return
        
        model = self.models[model_type]
        
        # Get feature names
        exclude_cols = ['TotalClaims', 'HasClaim', 'ClaimSeverity', 'LossRatio', 'Margin']
        feature_cols = [col for col in self.df.columns if col not in exclude_cols and self.df[col].dtype in ['int64', 'float64']]
        
        # Prepare data for SHAP
        if model_type == 'claim_severity':
            data = self.df[self.df['TotalClaims'] > 0][feature_cols]
        else:
            data = self.df[feature_cols]
        
        # Scale data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # SHAP analysis
        if hasattr(model, 'feature_importances_'):
            # Tree-based model
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(data_scaled)
        else:
            # Linear model
            explainer = shap.LinearExplainer(model, data_scaled)
            shap_values = explainer.shap_values(data_scaled)
        
        # Calculate feature importance
        feature_importance = np.abs(shap_values).mean(0)
        feature_importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance_df.head(10))
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        feature_importance_df.head(10).plot(x='feature', y='importance', kind='barh')
        plt.title(f'Top 10 Feature Importance - {model_type.replace("_", " ").title()}')
        plt.xlabel('SHAP Importance')
        plt.tight_layout()
        plt.savefig(f'plots/feature_importance_{model_type}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store results
        self.feature_importance[model_type] = feature_importance_df
        
        return feature_importance_df
    
    def create_visualizations(self):
        """Create model performance visualizations"""
        print("\n" + "=" * 60)
        print("CREATING MODEL VISUALIZATIONS")
        print("=" * 60)
        
        # Model comparison plots
        for model_type, results in self.results.items():
            if model_type == 'claim_probability':
                continue  # Skip classification for now
            
            # Create comparison plot
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Actual vs Predicted
            best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
            y_pred = results[best_model_name]['predictions']
            y_actual = results[best_model_name]['actual']
            
            axes[0].scatter(y_actual, y_pred, alpha=0.5)
            axes[0].plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
            axes[0].set_xlabel('Actual Values')
            axes[0].set_ylabel('Predicted Values')
            axes[0].set_title(f'{model_type.replace("_", " ").title()} - Actual vs Predicted')
            
            # Residuals
            residuals = y_actual - y_pred
            axes[1].scatter(y_pred, residuals, alpha=0.5)
            axes[1].axhline(y=0, color='r', linestyle='--')
            axes[1].set_xlabel('Predicted Values')
            axes[1].set_ylabel('Residuals')
            axes[1].set_title(f'{model_type.replace("_", " ").title()} - Residuals')
            
            plt.tight_layout()
            plt.savefig(f'plots/model_performance_{model_type}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_report(self) -> str:
        """
        Generate comprehensive modeling report
        
        Returns:
            str: Report file path
        """
        print("\n" + "=" * 60)
        print("GENERATING MODELING REPORT")
        print("=" * 60)
        
        # Create reports directory if it doesn't exist
        reports_dir = 'reports'
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        
        report_path = f"{reports_dir}/predictive_modeling_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("INSURANCE ANALYTICS - PREDICTIVE MODELING REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write("This report presents the development and evaluation of predictive models\n")
            f.write("for insurance risk-based pricing, including claim severity, claim probability,\n")
            f.write("and premium prediction models.\n\n")
            
            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write("-" * 30 + "\n")
            
            for model_type, results in self.results.items():
                f.write(f"\n{model_type.replace('_', ' ').title()}:\n")
                
                if model_type == 'claim_probability':
                    best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
                    accuracy = results[best_model]['accuracy']
                    f.write(f"  Best Model: {best_model}\n")
                    f.write(f"  Accuracy: {accuracy:.4f}\n")
                else:
                    best_model = max(results.keys(), key=lambda x: results[x]['r2'])
                    r2 = results[best_model]['r2']
                    rmse = results[best_model]['rmse']
                    f.write(f"  Best Model: {best_model}\n")
                    f.write(f"  R² Score: {r2:.4f}\n")
                    f.write(f"  RMSE: {rmse:.2f}\n")
            
            f.write("\nFEATURE IMPORTANCE INSIGHTS\n")
            f.write("-" * 30 + "\n")
            
            for model_type, importance_df in self.feature_importance.items():
                f.write(f"\n{model_type.replace('_', ' ').title()} - Top 5 Features:\n")
                for idx, row in importance_df.head(5).iterrows():
                    f.write(f"  • {row['feature']}: {row['importance']:.4f}\n")
            
            f.write("\nBUSINESS RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            f.write("• Use the developed models for risk-based pricing strategies\n")
            f.write("• Implement feature importance insights for premium adjustments\n")
            f.write("• Consider model ensemble approaches for improved accuracy\n")
            f.write("• Regular model retraining with new data is recommended\n")
        
        print(f"Report saved to: {report_path}")
        return report_path
    
    def run_all_models(self) -> Dict:
        """
        Run all predictive models
        
        Returns:
            Dict: All model results
        """
        print("INSURANCE PREDICTIVE MODELING")
        print("=" * 60)
        
        # Prepare data
        self.prepare_data()
        
        # Build models
        self.build_claim_severity_model()
        self.build_claim_probability_model()
        self.build_premium_model()
        
        # Analyze feature importance
        for model_type in ['claim_severity', 'claim_probability', 'premium']:
            if model_type in self.models:
                self.analyze_feature_importance(model_type)
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate report
        self.generate_report()
        
        return self.results

def main():
    """Main function to run predictive modeling"""
    from eda import InsuranceEDA
    
    # Load data
    eda = InsuranceEDA("data/MachineLearningRating_v3.txt")
    df = eda.load_data()
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Run predictive modeling
    modeler = InsurancePredictiveModeling(df)
    results = modeler.run_all_models()
    
    print("\n" + "=" * 60)
    print("PREDICTIVE MODELING COMPLETE")
    print("=" * 60)
    print("Check the 'plots' directory for visualizations")
    print("Check the 'reports' directory for detailed report")

if __name__ == "__main__":
    main() 