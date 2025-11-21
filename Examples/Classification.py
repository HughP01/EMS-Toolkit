import pandas as pd
import numpy as np
from sklearn.datasets import load_iris


print("=== CLASSIFICATION EXAMPLE ===")

# Load classification dataset
iris = load_iris()
df_classification = pd.DataFrame(iris.data, columns=iris.feature_names)
df_classification['species'] = iris.target
# Convert to actual species names for better interpretation
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df_classification['species'] = df_classification['species'].map(species_map)

print("Dataset shape:", df_classification.shape)
print("Features:", df_classification.columns.tolist())
print("\nClass distribution:")
print(df_classification['species'].value_counts())
print("\nFirst few rows:")
print(df_classification.head())

# Test different classification models
classification_models = ['logistic', 'randomforest', 'xgboost', 'svc']

for model_name in classification_models:
    print(f"\n{'='*60}")
    print(f"TRAINING {model_name.upper()} CLASSIFICATION MODEL")
    print(f"{'='*60}")
    
    try:
        result = Classif(
            data=df_classification,
            model=model_name,
            target='species',
            test_size=0.2,
            random_state=42,
            verbose=True
        )
        
        # Access specific metrics
        test_accuracy = result['metrics']['test']['Accuracy']
        test_f1 = result['metrics']['test']['F1-Score']
        print(f"{model_name}: Accuracy = {test_accuracy:.4f}, F1-Score = {test_f1:.4f}")
        
    except Exception as e:
        print(f"{model_name} failed: {e}")

# Example with custom classification dataset
print("\n" + "="*60)
print("CUSTOM CLASSIFICATION EXAMPLE")
print("="*60)

# Create synthetic classification data
np.random.seed(42)
n_samples = 1500

custom_class_data = pd.DataFrame({
    'feature_1': np.random.normal(0, 1, n_samples),
    'feature_2': np.random.normal(0, 1, n_samples),
    'feature_3': np.random.normal(0, 1, n_samples),
    'feature_4': np.random.normal(0, 1, n_samples)
})

# Create complex decision boundary for classification
custom_class_data['target'] = (
    (custom_class_data['feature_1']**2 + custom_class_data['feature_2']**2 > 2) &
    (custom_class_data['feature_3'] > 0)
).astype(int)

# Add some noise and create 3 classes
conditions = [
    (custom_class_data['feature_1'] + custom_class_data['feature_2'] < -1),
    (custom_class_data['feature_1'] + custom_class_data['feature_2'] > 1),
    (custom_class_data['feature_1'] + custom_class_data['feature_2'] >= -1) & 
    (custom_class_data['feature_1'] + custom_class_data['feature_2'] <= 1)
]
choices = ['Class_A', 'Class_B', 'Class_C']
custom_class_data['multi_class'] = np.select(conditions, choices)

print("Custom classification dataset:")
print(custom_class_data.head())
print(f"\nMulti-class distribution:")
print(custom_class_data['multi_class'].value_counts())

# Binary classification
print("\n--- BINARY CLASSIFICATION ---")
binary_result = Classif(
    data=custom_class_data[['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target']],
    model='randomforest',
    target='target',
    test_size=0.2,
    random_state=42,
    verbose=True
)

# Multi-class classification
print("\n--- MULTI-CLASS CLASSIFICATION ---")
multi_result = Classif(
    data=custom_class_data[['feature_1', 'feature_2', 'feature_3', 'feature_4', 'multi_class']],
    model='randomforest',
    target='multi_class',
    test_size=0.2,
    random_state=42,
    verbose=True
)
