#Regr Mods
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def _get_model_from_string(model_str, model_type, use, random_state):
    """
    Helper function to get model instance from string for both regression and classification.
    
    Parameters:
    -----------
    model_str : str
        Model identifier string
    model_type : str
        Type of model - "regression" or "classification"
    use : str
        Computational hardware to use - "CPU" or "GPU"
    random_state : int
        Random seed for reproducibility
    """
    model_str = model_str.lower()
    
    if model_type == "regression":
        if model_str == "linear":
            from sklearn.linear_model import LinearRegression
            return LinearRegression()
        
        elif model_str == "randomforest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(random_state=random_state, n_estimators=100)
        
        elif model_str == "xgboost":
            if use == "GPU":
                try:
                    from xgboost import XGBRegressor
                    return XGBRegressor(random_state=random_state, tree_method='gpu_hist')
                except:
                    from xgboost import XGBRegressor
                    return XGBRegressor(random_state=random_state)
            else:
                from xgboost import XGBRegressor
                return XGBRegressor(random_state=random_state)
        
        elif model_str == "svr":
            from sklearn.svm import SVR
            return SVR()
        
        elif model_str == "decisiontree":
            from sklearn.tree import DecisionTreeRegressor
            return DecisionTreeRegressor(random_state=random_state)
        
        elif model_str == "gradientboosting":
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(random_state=random_state)
        
        else:
            raise ValueError(f"Unsupported regression model type: {model_str}")
    
    elif model_type == "classification":
        # This will be used by the Classif function
        if model_str == "logistic":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(random_state=random_state)
        
        elif model_str == "randomforest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(random_state=random_state, n_estimators=100)
        
        elif model_str == "xgboost":
            if use == "GPU":
                try:
                    from xgboost import XGBClassifier
                    return XGBClassifier(random_state=random_state, tree_method='gpu_hist')
                except:
                    from xgboost import XGBClassifier
                    return XGBClassifier(random_state=random_state)
            else:
                from xgboost import XGBClassifier
                return XGBClassifier(random_state=random_state)
        
        elif model_str == "svc":
            from sklearn.svm import SVC
            return SVC(random_state=random_state)
        
        elif model_str == "decisiontree":
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier(random_state=random_state)
        
        elif model_str == "gradientboosting":
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(random_state=random_state)
        
        else:
            raise ValueError(f"Unsupported classification model type: {model_str}")
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")



#Regression Model

def Regr(df, model, use="CPU", target=None, test_size=0.2, random_state=42, 
         preprocess=True, scale_features=True, verbose=True):
    """
    Builds and evaluates a regression model pipeline with automated preprocessing.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing both features and target variable
    model : sklearn estimator object or str
        Regression model to use. If string, will map to appropriate model.
        Supported strings: "Linear", "RandomForest", "XGBoost", "SVR", "DecisionTree", "GradientBoosting"
    use : str, optional, default="CPU"
        Computational hardware to use. Options: "CPU" or "GPU" 
    target : str, optional, default=None
        Name of the target variable column. If None raises an error
    test_size : float, optional, default=0.2
        Proportion of dataset to use for testing (0.0 to 1.0)
    random_state : int, optional, default=42
        Random seed for reproducibility
    preprocess : bool, optional, default=True
        Whether to perform automated preprocessing (handle missing values, 
        encode categorical variables)
    scale_features : bool, optional, default=True
        Whether to scale numeric features (standardization)
    verbose : bool, optional, default=True
        Whether to print detailed progress and evaluation metrics
    
    Returns:
    --------
    dict : Dictionary containing trained model, preprocessing objects, 
           and evaluation metrics
    """
    
    # Validate inputs
    if target is None:
        raise ValueError("Target variable name must be provided")
    
    if target not in df.columns:
        raise ValueError(f"Target variable '{target}' not found in DataFrame columns")
    
    if use not in ["CPU", "GPU"]:
        raise ValueError("use parameter must be 'CPU' or 'GPU'")
    # Handle string model inputs
    if isinstance(model, str):
        model = _get_model_from_string(model, "regression", use, random_state)
    
    if verbose:
        print("=" * 50)
        print("REGRESSION PIPELINE INITIALIZED")
        print("=" * 50)
        print(f"Dataset shape: {df.shape}")
        print(f"Target variable: {target}")
        print(f"Using: {use}")
        print(f"Model: {type(model).__name__}")
    
    # Create copy to avoid modifying original data
    df_processed = df.copy()
    
    # Separate features and target
    X = df_processed.drop(columns=[target])
    y = df_processed[target]
    
    if verbose:
        print(f"Features: {X.shape[1]}")
        print(f"Target distribution - Mean: {y.mean():.4f}, Std: {y.std():.4f}")
        print(f"Target range: {y.min():.4f} to {y.max():.4f}")
    
    # Preprocessing pipeline
    preprocessing_objects = {}
    
    if preprocess:
        if verbose:
            print("\n--- PREPROCESSING ---")
        
        # Handle missing values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        # Numeric imputation
        if not numeric_cols.empty and X[numeric_cols].isnull().any().any():
            numeric_imputer = SimpleImputer(strategy='median')
            X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
            preprocessing_objects['numeric_imputer'] = numeric_imputer
            if verbose:
                print(f"Applied median imputation to {len(numeric_cols)} numeric features")
        
        # Categorical imputation and encoding
        if not categorical_cols.empty:
            # Handle missing categorical values
            if X[categorical_cols].isnull().any().any():
                X[categorical_cols] = X[categorical_cols].fillna('Missing')
                if verbose:
                    print("Filled missing categorical values with 'Missing'")
            
            # Encode categorical variables
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
            
            preprocessing_objects['label_encoders'] = label_encoders
            if verbose:
                print(f"Label encoded {len(categorical_cols)} categorical features")
    
    # Feature scaling
    if scale_features:
        if verbose:
            print("\n--- FEATURE SCALING ---")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        preprocessing_objects['scaler'] = scaler
        if verbose:
            print("Applied StandardScaler to all features")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    if verbose:
        print(f"\n--- DATA SPLITTING ---")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")
    
    # Model training
    if verbose:
        print(f"\n--- MODEL TRAINING ---")
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train': {
            'MAE': mean_absolute_error(y_train, y_pred_train),
            'MSE': mean_squared_error(y_train, y_pred_train),
            'RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'R2': r2_score(y_train, y_pred_train)
        },
        'test': {
            'MAE': mean_absolute_error(y_test, y_pred_test),
            'MSE': mean_squared_error(y_test, y_pred_test),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'R2': r2_score(y_test, y_pred_test)
        }
    }
    
    # Print results
    if verbose:
        print(f"\n--- MODEL EVALUATION ---")
        print("TRAINING SET:")
        print(f"  MAE:  {metrics['train']['MAE']:.4f}")
        print(f"  MSE:  {metrics['train']['MSE']:.4f}")
        print(f"  RMSE: {metrics['train']['RMSE']:.4f}")
        print(f"  R²:   {metrics['train']['R2']:.4f}")
        
        print("\nTEST SET:")
        print(f"  MAE:  {metrics['test']['MAE']:.4f}")
        print(f"  MSE:  {metrics['test']['MSE']:.4f}")
        print(f"  RMSE: {metrics['test']['RMSE']:.4f}")
        print(f"  R²:   {metrics['test']['R2']:.4f}")
        
        print("\n" + "=" * 50)
        print("REGRESSION PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 50)
    
    # Return comprehensive results
    return {
        'model': model,
        'preprocessing_objects': preprocessing_objects,
        'metrics': metrics,
        'feature_names': X.columns.tolist(),
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'model_type': 'regression'
    }


#Classif Mods
def Classif(df, model, use="CPU", target=None, test_size=0.2, random_state=42, 
         preprocess=True, scale_features=True, verbose=True):
    """
    Builds and evaluates a regression model pipeline with automated preprocessing.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing both features and target variable
    model : sklearn estimator object or str
        Classification model to use.
    use : str, optional, default="CPU"
        Computational hardware to use. Options: "CPU" or "GPU" 
    target : str, optional, default=None
        Name of the target variable column. If None raises an error
    test_size : float, optional, default=0.2
        Proportion of dataset to use for testing (0.0 to 1.0)
    random_state : int, optional, default=42
        Random seed for reproducibility
    preprocess : bool, optional, default=True
        Whether to perform automated preprocessing (handle missing values, 
        encode categorical variables)
    scale_features : bool, optional, default=True
        Whether to scale numeric features (standardization)
    verbose : bool, optional, default=True
        Whether to print detailed progress and evaluation metrics
    """
    print("Under Construction")
