import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing, load_iris
print("=== REGRESSION EXAMPLE ===")

# Load regression dataset
housing = fetch_california_housing()
df_regression = pd.DataFrame(housing.data, columns=housing.feature_names)
df_regression['Price'] = housing.target * 100000  # Convert to actual price

print("Dataset shape:", df_regression.shape)
print("Features:", df_regression.columns.tolist())
print("\nFirst few rows:")
print(df_regression.head())

# Test different regression models
regression_models = ['linear', 'randomforest', 'xgboost', 'decisiontree']

for model_name in regression_models:
    print(f"\n{'='*60}")
    print(f"TRAINING {model_name.upper()} REGRESSION MODEL")
    print(f"{'='*60}")
    
    try:
        result = Regr(
            data=df_regression,
            model=model_name,
            target='Price',
            test_size=0.2,
            random_state=42,
            verbose=True
        )
        
        # Access specific metrics
        test_r2 = result['metrics']['test']['R2']
        test_rmse = result['metrics']['test']['RMSE']
        print(f"{model_name}: R² = {test_r2:.4f}, RMSE = ${test_rmse:,.2f}")
        
    except Exception as e:
        print(f"{model_name} failed: {e}")

# Example with custom dataset
print("\n" + "="*60)
print("CUSTOM REGRESSION EXAMPLE")
print("="*60)

# Create synthetic regression data
np.random.seed(42)
n_samples = 1000

custom_reg_data = pd.DataFrame({
    'house_size': np.random.normal(2000, 500, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.randint(1, 4, n_samples),
    'age': np.random.randint(0, 50, n_samples),
    'location_score': np.random.normal(7, 2, n_samples)
})

# Create target variable with some relationship to features
custom_reg_data['price'] = (
    custom_reg_data['house_size'] * 100 +
    custom_reg_data['bedrooms'] * 50000 +
    custom_reg_data['bathrooms'] * 30000 -
    custom_reg_data['age'] * 1000 +
    custom_reg_data['location_score'] * 20000 +
    np.random.normal(0, 50000, n_samples)  # Noise
)

print("Custom regression dataset:")
print(custom_reg_data.head())
print(f"\nTarget statistics:")
print(f"Mean price: ${custom_reg_data['price'].mean():,.2f}")
print(f"Std price: ${custom_reg_data['price'].std():,.2f}")

# Train on custom data
custom_result = Regr(
    data=custom_reg_data,
    model='randomforest',
    target='price',
    test_size=0.2,
    random_state=42,
    verbose=True
)
