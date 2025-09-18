#Regr Mods
def Regr(df, model, use="CPU", target=None, test_size=0.2, random_state=42, 
         preprocess=True, scale_features=True, verbose=True):
    """
    Builds and evaluates a regression model pipeline with automated preprocessing.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing both features and target variable
    model : sklearn estimator object or str
        Regression model to use. 
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
