# Insurance Analytics Project

## Overview
This project focuses on analyzing insurance data from February 2014 to August 2015 to understand risk patterns, profitability, and predictive analytics in the insurance industry.

## Project Structure
```
├── data/                          # Data files
│   └── MachineLearningRating_v3.txt
├── src/                           # Source code
│   ├── main.py                    # Main analysis pipeline
│   ├── eda.py                     # Exploratory Data Analysis
│   ├── statistical_analysis.py    # Statistical modeling
│   └── visualization.py           # Visualization utilities
├── notebooks/                     # Jupyter notebooks
│   └── eda_notebook.ipynb
├── plots/                         # Generated plots and visualizations
├── reports/                       # Analysis reports
├── tests/                         # Unit tests
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## Data Description
The dataset contains insurance policy information from Feb 2014 to Aug 2015 with the following key components:

### Policy Information
- UnderwrittenCoverID, PolicyID, TransactionMonth

### Client Information
- IsVATRegistered, Citizenship, LegalType, Title, Language
- Bank, AccountType, MaritalStatus, Gender

### Location Information
- Country, Province, PostalCode, MainCrestaZone, SubCrestaZone

### Vehicle Information
- ItemType, Mmcode, VehicleType, RegistrationYear, Make, Model
- Cylinders, Cubiccapacity, Kilowatts, Bodytype, NumberOfDoors
- VehicleIntroDate, CustomValueEstimate, AlarmImmobiliser
- TrackingDevice, CapitalOutstanding, NewVehicle, WrittenOff
- Rebuilt, Converted, CrossBorder, NumberOfVehiclesInFleet

### Plan Information
- SumInsured, TermFrequency, CalculatedPremiumPerTerm
- ExcessSelected, CoverCategory, CoverType, CoverGroup
- Section, Product, StatutoryClass, StatutoryRiskType

### Financial Information
- TotalPremium, TotalClaims

## Learning Outcomes
- Understanding data structure and algorithms in EDA and ML pipelines
- Modular and object-oriented Python code writing
- Statistical modeling and analysis
- A/B testing design and implementation
- Data versioning and management

## Setup Instructions

### Prerequisites
- Python 3.8+
- Git
- GitHub account

### Installation
1. Clone the repository:
```bash
git clone <repository-url>
cd insurance-analytics
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Analysis
1. Basic EDA:
```bash
python src/main.py
```

2. Jupyter notebook:
```bash
jupyter notebook notebooks/eda_notebook.ipynb
```

## Task Structure

### Task 1: EDA & Statistical Analysis
- [x] Git repository setup
- [x] Data understanding and exploration
- [x] Descriptive statistics
- [x] Data quality assessment
- [x] Univariate and multivariate analysis
- [x] Visualization creation

### Task 2: Statistical Modeling
- [ ] Hypothesis testing
- [ ] A/B testing design
- [ ] Statistical distributions analysis
- [ ] Predictive modeling

### Task 3: Machine Learning Pipeline
- [ ] Feature engineering
- [ ] Model development
- [ ] Model evaluation
- [ ] Deployment preparation

## Key Performance Indicators
- Loss Ratio analysis (TotalClaims / TotalPremium)
- Geographic and demographic risk patterns
- Temporal trends in claims and premiums
- Vehicle make/model risk assessment
- Outlier detection and analysis

## Contributing
1. Create a feature branch: `git checkout -b feature-name`
2. Make your changes
3. Commit with descriptive messages: `git commit -m "Add feature description"`
4. Push to your branch: `git push origin feature-name`
5. Create a pull request

## License
This project is for educational purposes as part of the 10Academy program.

## Contact
For questions or issues, please contact the project maintainers. 