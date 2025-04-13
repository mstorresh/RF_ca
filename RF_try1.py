# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# import matplotlib.pyplot as plt
# from category_encoders import TargetEncoder

# # 1. Load the dataset
# def load_data(file_path='car_data.csv'):
#     try:
#         df = pd.read_csv(file_path)
#         print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
#         return df
#     except Exception as e:
#         print(f"Error loading dataset: {e}")
#         return None

# # 2. Preprocess the data
# def preprocess_data(df):
#     # Make a copy to avoid warnings
#     df_processed = df.copy()
    
#     # Handle missing values
#     print("Handling missing values...")
#     for col in df_processed.columns:
#         if df_processed[col].dtype in [np.float64, np.int64]:
#             df_processed[col].fillna(df_processed[col].median(), inplace=True)
#         else:
#             df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    
#     # Extract year and month from fecha_venta
#     print("Extracting date features...")
#     try:
#         df_processed['fecha_venta'] = pd.to_datetime(df_processed['fecha_venta'])
#         df_processed['year_venta'] = df_processed['fecha_venta'].dt.year
#         df_processed['month_venta'] = df_processed['fecha_venta'].dt.month
#         # Drop the original fecha_venta column
#         df_processed.drop('fecha_venta', axis=1, inplace=True)
#     except Exception as e:
#         print(f"Error processing date column: {e}")
    
#     # Identify different types of columns
#     # combustible_int is an encoded categorical feature (fuel type)
#     categorical_cols = ['combustible_int']
    
#     # Cod_fasecolda will be handled with target encoding
#     target_encode_cols = ['Cod_fasecolda']
    
#     # True numerical features
#     numerical_cols = ['kilometers', 'year_model', 'description_int', 'gamma_int', 
#                      'demand', 'Popularidad', 'year_venta', 'month_venta']
    
#     # Binary features
#     binary_cols = ['Blindaje']
    
#     # Define target variable
#     y = df_processed['pricing']
    
#     # Input features (all columns except the target)
#     X = df_processed.drop('pricing', axis=1)
    
#     # Keep track of column names for later interpretation
#     feature_names = numerical_cols + categorical_cols + target_encode_cols + binary_cols
    
#     return X, y, numerical_cols, categorical_cols, target_encode_cols, binary_cols, feature_names

# # 3. Split the data and train the model
# def build_and_train_model(X, y, numerical_cols, categorical_cols, target_encode_cols, binary_cols):
#     # Split the data into training and test sets (80% train, 20% test)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
#     # Create preprocessing steps for different column types
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), numerical_cols),
#             ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
#             ('target_enc', TargetEncoder(smoothing=10), target_encode_cols),
#             # Binary columns pass through without transformation
#         ],
#         remainder='passthrough'  # This will pass through the binary columns
#     )
    
#     # Create a pipeline that first preprocesses the data then applies the Random Forest Regressor
#     model = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
#     ])
    
#     # Train the model
#     print("Training the Random Forest model...")
#     model.fit(X_train, y_train)
#     print("Model training completed!")
    
#     return model, X_train, X_test, y_train, y_test

# # 4. Evaluate the model
# def evaluate_model(model, X_test, y_test, numerical_cols, categorical_cols, target_encode_cols, binary_cols):
#     # Make predictions on the test set
#     y_pred = model.predict(X_test)
    
#     # Calculate performance metrics
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
    
#     print(f"Model Evaluation Metrics:")
#     print(f"Mean Squared Error (MSE): {mse:.2f}")
#     print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
#     print(f"Mean Absolute Error (MAE): {mae:.2f}")
#     print(f"R² Score: {r2:.4f}")
    
#     # Extract feature importances
#     rf_model = model.named_steps['regressor']
    
#     # Get the preprocessor from the pipeline
#     preprocessor = model.named_steps['preprocessor']
    
#     # Get one-hot encoded feature names (for categorical columns)
#     ohe_features = []
#     if categorical_cols:
#         ohe = preprocessor.named_transformers_['cat']
#         ohe_features = list(ohe.get_feature_names_out(categorical_cols))
    
#     # Combine all transformed feature names in the correct order
#     feature_names_after_preprocessing = []
    
#     # Add scaled numerical features
#     feature_names_after_preprocessing.extend(numerical_cols)
    
#     # Add one-hot encoded categorical features
#     feature_names_after_preprocessing.extend(ohe_features)
    
#     # Add target encoded features
#     feature_names_after_preprocessing.extend(target_encode_cols)
    
#     # Add binary features that were passed through
#     feature_names_after_preprocessing.extend(binary_cols)
    
#     # Get feature importances
#     importances = rf_model.feature_importances_
    
#     # Create a dictionary of feature importances
#     feature_importance_dict = {}
#     for i, importance in enumerate(importances):
#         if i < len(feature_names_after_preprocessing):
#             feature_importance_dict[feature_names_after_preprocessing[i]] = importance
    
#     # Sort feature importances in descending order
#     sorted_importances = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    
#     # Print top 10 feature importances
#     print("\nFeature Importance Ranking:")
#     for i, (feature, importance) in enumerate(sorted_importances[:10]):
#         print(f"{i+1}. {feature}: {importance:.4f}")
    
#     # Plot actual vs predicted values
#     plt.figure(figsize=(10, 6))
#     plt.scatter(y_test, y_pred, alpha=0.5)
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
#     plt.xlabel('Actual Car Prices')
#     plt.ylabel('Predicted Car Prices')
#     plt.title('Actual vs Predicted Car Prices')
#     plt.tight_layout()
#     plt.show()
    
#     # Create residual plot
#     plt.figure(figsize=(10, 6))
#     residuals = y_test - y_pred
#     plt.scatter(y_pred, residuals, alpha=0.5)
#     plt.axhline(y=0, color='r', linestyle='--')
#     plt.xlabel('Predicted Car Prices')
#     plt.ylabel('Residuals')
#     plt.title('Residual Plot')
#     plt.tight_layout()
#     plt.show()
    
#     return {
#         'mse': mse,
#         'rmse': rmse,
#         'r2': r2,
#         'mae': mae,
#         'y_pred': y_pred,
#         'feature_importances': feature_importance_dict
#     }

# # Function to tune hyperparameters (optional)
# def tune_hyperparameters(X, y, numerical_cols, categorical_cols, target_encode_cols, binary_cols):
#     from sklearn.model_selection import GridSearchCV
    
#     # Create the preprocessing pipeline
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), numerical_cols),
#             ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
#             ('target_enc', TargetEncoder(smoothing=10), target_encode_cols),
#         ],
#         remainder='passthrough'
#     )
    
#     # Create the full pipeline
#     pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('regressor', RandomForestRegressor(random_state=42))
#     ])
    
#     # Define the parameter grid to search
#     param_grid = {
#         'regressor__n_estimators': [50, 100, 200],
#         'regressor__max_depth': [None, 10, 20, 30],
#         'regressor__min_samples_split': [2, 5, 10],
#         'regressor__min_samples_leaf': [1, 2, 4]
#     }
    
#     # Create the grid search object
#     grid_search = GridSearchCV(
#         pipeline, 
#         param_grid, 
#         cv=5, 
#         scoring='neg_mean_squared_error',
#         n_jobs=-1,
#         verbose=1
#     )
    
#     # Fit the grid search
#     print("Starting hyperparameter tuning. This may take some time...")
#     grid_search.fit(X, y)
    
#     # Print the best parameters
#     print(f"Best parameters: {grid_search.best_params_}")
#     print(f"Best score: {-grid_search.best_score_:.2f} (MSE)")
    
#     return grid_search.best_estimator_, grid_search.best_params_

# # Main function to run the entire process
# def main(file_path='car_data.csv', tune_params=False):
#     # Load the data
#     df = load_data(file_path)
    
#     if df is None:
#         return
    
#     # Preprocess the data
#     X, y, numerical_cols, categorical_cols, target_encode_cols, binary_cols, feature_names = preprocess_data(df)
    
#     if tune_params:
#         # Tune hyperparameters
#         best_model, best_params = tune_hyperparameters(X, y, numerical_cols, categorical_cols, target_encode_cols, binary_cols)
        
#         # Evaluate the tuned model
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         results = evaluate_model(best_model, X_test, y_test, numerical_cols, categorical_cols, target_encode_cols, binary_cols)
#         return best_model, results
#     else:
#         # Build and train the model with default parameters
#         model, X_train, X_test, y_train, y_test = build_and_train_model(X, y, numerical_cols, categorical_cols, target_encode_cols, binary_cols)
        
#         # Evaluate the model
#         results = evaluate_model(model, X_test, y_test, numerical_cols, categorical_cols, target_encode_cols, binary_cols)
#         return model, results

# # Run the script if executed directly
# if __name__ == "__main__":
#     # Set tune_params=True to perform hyperparameter tuning (takes longer)
#     model, results = main(tune_params=False)
#     print("\nModel training and evaluation completed successfully!")
#     print(results)

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# import matplotlib.pyplot as plt
# import seaborn as sns
# from category_encoders import TargetEncoder
# from sklearn.model_selection import cross_val_score, KFold

# # Set style for plots
# plt.style.use('seaborn-v0_8-whitegrid')
# sns.set_palette("deep")

# # 1. Load the dataset
# def load_data(file_path='car_data.csv'):
#     try:
#         df = pd.read_csv(file_path)
#         print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
#         return df
#     except Exception as e:
#         print(f"Error loading dataset: {e}")
#         return None

# # 2. Preprocess the data
# def preprocess_data(df):
#     # Make a copy to avoid warnings
#     df_processed = df.copy()
    
#     # Handle missing values
#     print("Handling missing values...")
#     for col in df_processed.columns:
#         if df_processed[col].dtype in [np.float64, np.int64]:
#             df_processed[col].fillna(df_processed[col].median(), inplace=True)
#         else:
#             df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    
#     # Extract year and month from fecha_venta
#     print("Extracting date features...")
#     try:
#         df_processed['fecha_venta'] = pd.to_datetime(df_processed['fecha_venta'])
#         df_processed['year_venta'] = df_processed['fecha_venta'].dt.year
#         df_processed['month_venta'] = df_processed['fecha_venta'].dt.month
#         # Drop the original fecha_venta column
#         df_processed.drop('fecha_venta', axis=1, inplace=True)
#     except Exception as e:
#         print(f"Error processing date column: {e}")
    
#     # Identify different types of columns
#     # combustible_int is an encoded categorical feature (fuel type)
#     categorical_cols = ['combustible_int']
    
#     # Cod_fasecolda will be handled with target encoding
#     target_encode_cols = ['Cod_fasecolda']
    
#     # True numerical features
#     numerical_cols = ['kilometers', 'year_model', 'description_int', 'gamma_int', 
#                      'demand', 'Popularidad', 'year_venta', 'month_venta']
    
#     # Binary features
#     binary_cols = ['Blindaje']
    
#     # Define target variable
#     y = df_processed['pricing']
    
#     # Input features (all columns except the target)
#     X = df_processed.drop('pricing', axis=1)
    
#     # Keep track of column names for later interpretation
#     feature_names = numerical_cols + categorical_cols + target_encode_cols + binary_cols
    
#     return X, y, numerical_cols, categorical_cols, target_encode_cols, binary_cols, feature_names

# # 3. Split the data and train the model
# def build_and_train_model(X, y, numerical_cols, categorical_cols, target_encode_cols, binary_cols):
#     # Split the data into training and test sets (80% train, 20% test)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
#     # Create preprocessing steps for different column types
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), numerical_cols),
#             ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
#             ('target_enc', TargetEncoder(smoothing=10), target_encode_cols),
#             # Binary columns pass through without transformation
#         ],
#         remainder='passthrough'  # This will pass through the binary columns
#     )
    
#     # Create a pipeline that first preprocesses the data then applies the Random Forest Regressor
#     model = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('regressor', RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1))
#     ])
    
#     # Train the model
#     print("Training the Random Forest model...")
#     model.fit(X_train, y_train)
#     print("Model training completed!")
    
#     return model, X_train, X_test, y_train, y_test

# # 4. Evaluate the model with visualization of metrics
# def evaluate_model(model, X, y, X_test, y_test, numerical_cols, categorical_cols, target_encode_cols, binary_cols):
#     # Make predictions on the test set
#     y_pred = model.predict(X_test)
    
#     # Calculate performance metrics
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
    
#     print(f"\nModel Evaluation Metrics:")
#     print(f"Mean Squared Error (MSE): {mse:.2f}")
#     print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
#     print(f"Mean Absolute Error (MAE): {mae:.2f}")
#     print(f"R² Score: {r2:.4f}")
    
#     # Create a figure with multiple subplots for different evaluation metrics
#     fig = plt.figure(figsize=(20, 15))
    
#     # 1. Actual vs Predicted values plot
#     ax1 = fig.add_subplot(2, 2, 1)
#     ax1.scatter(y_test, y_pred, alpha=0.5, color='blue')
#     ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
#     ax1.set_xlabel('Actual Car Prices', fontsize=12)
#     ax1.set_ylabel('Predicted Car Prices', fontsize=12)
#     ax1.set_title('Actual vs Predicted Car Prices', fontsize=14)
    
#     # 2. Residuals plot
#     ax2 = fig.add_subplot(2, 2, 2)
#     residuals = y_test - y_pred
#     ax2.scatter(y_pred, residuals, alpha=0.5, color='green')
#     ax2.axhline(y=0, color='r', linestyle='--')
#     ax2.set_xlabel('Predicted Car Prices', fontsize=12)
#     ax2.set_ylabel('Residuals', fontsize=12)
#     ax2.set_title('Residual Plot', fontsize=14)
    
#     # 3. Distribution of residuals (histogram)
#     ax3 = fig.add_subplot(2, 2, 3)
#     sns.histplot(residuals, kde=True, ax=ax3, color='purple')
#     ax3.axvline(x=0, color='r', linestyle='--')
#     ax3.set_xlabel('Residuals', fontsize=12)
#     ax3.set_ylabel('Frequency', fontsize=12)
#     ax3.set_title('Distribution of Residuals', fontsize=14)
    
#     # 4. Cross-validation scores
#     ax4 = fig.add_subplot(2, 2, 4)
    
#     # Perform cross-validation
#     cv = KFold(n_splits=5, shuffle=True, random_state=42)
#     cv_scores_r2 = cross_val_score(model, X, y, cv=cv, scoring='r2')
#     cv_scores_neg_mae = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
#     cv_scores_neg_rmse = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error'))
    
#     # Prepare data for plotting
#     cv_metrics = pd.DataFrame({
#         'R²': cv_scores_r2,
#         'MAE': cv_scores_neg_mae,
#         'RMSE': cv_scores_neg_rmse
#     })
    
#     # Plot cross-validation metrics
#     cv_metrics_long = pd.melt(cv_metrics.reset_index(), id_vars=['index'], 
#                              value_vars=['R²', 'MAE', 'RMSE'],
#                              var_name='Metric', value_name='Value')
    
#     # Scale MAE and RMSE for visualization alongside R²
#     metrics_mean = cv_metrics_long.groupby('Metric')['Value'].mean().to_dict()
#     scale_factor = metrics_mean['R²'] / max(metrics_mean['MAE'], metrics_mean['RMSE'])
    
#     cv_metrics_long.loc[cv_metrics_long['Metric'].isin(['MAE', 'RMSE']), 'Value'] *= scale_factor
    
#     sns.boxplot(x='Metric', y='Value', data=cv_metrics_long, ax=ax4)
#     ax4.set_title('Cross-Validation Metrics (Scaled)', fontsize=14)
#     ax4.set_ylabel('Score', fontsize=12)
    
#     # Add the original values as text annotations
#     for i, metric in enumerate(['R²', 'MAE', 'RMSE']):
#         ax4.annotate(f"Mean: {metrics_mean[metric]:.4f}", 
#                     xy=(i, cv_metrics_long[cv_metrics_long['Metric'] == metric]['Value'].min()),
#                     xytext=(0, -30), textcoords='offset points',
#                     ha='center', va='top', fontsize=10)
    
#     # Add a note about scaling
#     ax4.annotate("Note: MAE and RMSE scaled for visualization", 
#                 xy=(0.5, 0), xytext=(0, -50), textcoords='offset points',
#                 ha='center', va='top', fontsize=10, xycoords='axes fraction')
    
#     plt.tight_layout()
#     fig.suptitle('Model Evaluation Metrics Visualization', fontsize=16)
#     plt.subplots_adjust(top=0.92)
#     plt.show()
    
#     # Extract feature importances and create a separate visualization
#     rf_model = model.named_steps['regressor']
    
#     # Get the preprocessor from the pipeline
#     preprocessor = model.named_steps['preprocessor']
    
#     # Get one-hot encoded feature names (for categorical columns)
#     ohe_features = []
#     if categorical_cols:
#         ohe = preprocessor.named_transformers_['cat']
#         ohe_features = list(ohe.get_feature_names_out(categorical_cols))
    
#     # Combine all transformed feature names in the correct order
#     feature_names_after_preprocessing = []
    
#     # Add scaled numerical features
#     feature_names_after_preprocessing.extend(numerical_cols)
    
#     # Add one-hot encoded categorical features
#     feature_names_after_preprocessing.extend(ohe_features)
    
#     # Add target encoded features
#     feature_names_after_preprocessing.extend(target_encode_cols)
    
#     # Add binary features that were passed through
#     feature_names_after_preprocessing.extend(binary_cols)
    
#     # Get feature importances
#     importances = rf_model.feature_importances_
    
#     # Create a dictionary of feature importances
#     feature_importance_dict = {}
#     for i, importance in enumerate(importances):
#         if i < len(feature_names_after_preprocessing):
#             feature_importance_dict[feature_names_after_preprocessing[i]] = importance
    
#     # Sort feature importances in descending order
#     sorted_importances = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    
#     # Print top 10 feature importances
#     print("\nFeature Importance Ranking:")
#     for i, (feature, importance) in enumerate(sorted_importances[:10]):
#         print(f"{i+1}. {feature}: {importance:.4f}")
    
#     # Plot feature importances
#     plt.figure(figsize=(12, 8))
#     # Get top 15 features
#     top_features = dict(sorted_importances[:15])
#     plt.barh(list(top_features.keys()), list(top_features.values()))
#     plt.xlabel('Importance')
#     plt.ylabel('Feature')
#     plt.title('Top 15 Feature Importances')
#     plt.tight_layout()
#     plt.show()
    
#     # Plot error analysis by important features
#     # Select top 3 numerical features
#     top_numerical_features = [feat for feat, _ in sorted_importances if feat in numerical_cols][:3]
    
#     if top_numerical_features:
#         fig, axes = plt.subplots(1, len(top_numerical_features), figsize=(18, 6))
#         if len(top_numerical_features) == 1:
#             axes = [axes]  # Make axes iterable if there's only one subplot
        
#         for i, feature in enumerate(top_numerical_features):
#             # Get the feature values from the test set
#             feature_values = X_test[feature]
            
#             # Create scatter plot
#             axes[i].scatter(feature_values, residuals, alpha=0.5)
#             axes[i].axhline(y=0, color='r', linestyle='--')
#             axes[i].set_xlabel(feature)
#             axes[i].set_ylabel('Residuals')
#             axes[i].set_title(f'Residuals vs {feature}')
        
#         plt.tight_layout()
    
#     return {
#         'mse': mse,
#         'rmse': rmse,
#         'r2': r2,
#         'mae': mae,
#         'y_pred': y_pred,
#         'feature_importances': feature_importance_dict,
#         'cross_val_scores': {
#             'r2': cv_scores_r2,
#             'mae': cv_scores_neg_mae,
#             'rmse': cv_scores_neg_rmse
#         }
#     }

# # Main function to run the entire process
# def main(file_path='car_data.csv'):
#     # Load the data
#     df = load_data(file_path)
    
#     if df is None:
#         return
    
#     # Preprocess the data
#     X, y, numerical_cols, categorical_cols, target_encode_cols, binary_cols, feature_names = preprocess_data(df)
    
#     # Build and train the model
#     model, X_train, X_test, y_train, y_test = build_and_train_model(X, y, numerical_cols, categorical_cols, target_encode_cols, binary_cols)
    
#     # Evaluate the model with visualizations
#     results = evaluate_model(model, X, y, X_test, y_test, numerical_cols, categorical_cols, target_encode_cols, binary_cols)
    
#     print("\nModel training and evaluation completed successfully!")
#     print("\nTo interpret these plots:")
#     print("1. Actual vs Predicted: The closer points are to the diagonal line, the better the predictions.")
#     print("2. Residual Plot: Points should be randomly scattered around the horizontal line at y=0.")
#     print("3. Residual Distribution: Should be approximately normal and centered at zero.")
#     print("4. Cross-Validation Metrics: Shows model consistency across different data splits.")
#     print("5. Feature Importance: Shows which features most influence the model's predictions.")
    
#     return model, results

# # Run the script if executed directly
# if __name__ == "__main__":
#     model, results = main()


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
# from sklearn.linear_model import ElasticNet, Ridge
# from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# import matplotlib.pyplot as plt
# import seaborn as sns
# from category_encoders import TargetEncoder
# from sklearn.base import BaseEstimator, TransformerMixin
# import xgboost as xgb
# import lightgbm as lgb
# import warnings
# warnings.filterwarnings('ignore')

# # Set style for plots
# plt.style.use('seaborn-v0_8-whitegrid')
# sns.set_palette("deep")

# # Custom transformer for handling outliers
# class OutlierRemover(BaseEstimator, TransformerMixin):
#     def __init__(self, column, n_std=3):
#         self.column = column
#         self.n_std = n_std
#         self.lower_bound = None
#         self.upper_bound = None
        
#     def fit(self, X, y=None):
#         mean = X[self.column].mean()
#         std = X[self.column].std()
#         self.lower_bound = mean - self.n_std * std
#         self.upper_bound = mean + self.n_std * std
#         return self
    
#     def transform(self, X, y=None):
#         X_copy = X.copy()
#         mask = (X_copy[self.column] >= self.lower_bound) & (X_copy[self.column] <= self.upper_bound)
#         return X_copy[mask], y[mask] if y is not None else X_copy

# # Custom transformer for handling infinity and extreme values
# class InfinityHandler(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.column_max = {}
#         self.column_min = {}
        
#     def fit(self, X, y=None):
#         for col in X.columns:
#             if X[col].dtype in [np.float64, np.int64]:
#                 self.column_max[col] = X[col].replace([np.inf, -np.inf], np.nan).quantile(0.999)
#                 self.column_min[col] = X[col].replace([np.inf, -np.inf], np.nan).quantile(0.001)
#         return self
    
#     def transform(self, X, y=None):
#         X_copy = X.copy()
#         for col in X_copy.columns:
#             if col in self.column_max:
#                 # Replace infinities with NaN first
#                 X_copy[col] = X_copy[col].replace([np.inf, -np.inf], np.nan)
                
#                 # Cap extreme values at 99.9th percentile
#                 X_copy[col] = X_copy[col].clip(lower=self.column_min[col], upper=self.column_max[col])
                
#                 # Fill any remaining NaNs with median
#                 X_copy[col] = X_copy[col].fillna(X_copy[col].median())
        
#         return X_copy

# # 1. Load the dataset
# def load_data(file_path='car_data.csv'):
#     try:
#         df = pd.read_csv(file_path)
#         print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
#         return df
#     except Exception as e:
#         print(f"Error loading dataset: {e}")
#         return None

# # 2. Preprocess the data with advanced feature engineering
# def preprocess_data(df):
#     # Make a copy to avoid warnings
#     df_processed = df.copy()
    
#     # Handle missing values
#     print("Handling missing values...")
#     for col in df_processed.columns:
#         if df_processed[col].dtype in [np.float64, np.int64]:
#             df_processed[col].fillna(df_processed[col].median(), inplace=True)
#         else:
#             df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    
#     # Check for and handle infinity values
#     print("Checking for infinity values...")
#     for col in df_processed.select_dtypes(include=[np.number]).columns:
#         inf_count = np.isinf(df_processed[col]).sum()
#         if inf_count > 0:
#             print(f"Found {inf_count} infinity values in column {col}")
#             # Replace infinity with NaN, then with median
#             df_processed[col] = df_processed[col].replace([np.inf, -np.inf], np.nan)
#             df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
#     # Extract year and month from fecha_venta if it exists
#     print("Extracting date features...")
#     try:
#         if 'fecha_venta' in df_processed.columns:
#             df_processed['fecha_venta'] = pd.to_datetime(df_processed['fecha_venta'], errors='coerce')
#             df_processed['year_venta'] = df_processed['fecha_venta'].dt.year
#             df_processed['month_venta'] = df_processed['fecha_venta'].dt.month
#             # Drop the original fecha_venta column
#             df_processed.drop('fecha_venta', axis=1, inplace=True)
#         else:
#             print("Column 'fecha_venta' not found, using current year for calculations")
#             # Use a default year and month if fecha_venta is not available
#             df_processed['year_venta'] = 2023
#             df_processed['month_venta'] = 6
#     except Exception as e:
#         print(f"Error processing date column: {e}")
#         # Use a default year and month if processing fails
#         df_processed['year_venta'] = 2023
#         df_processed['month_venta'] = 6
    
#     # ADVANCED FEATURE ENGINEERING
#     print("Applying advanced feature engineering...")
    
#     # Ensure 'year_model' exists
#     if 'year_model' not in df_processed.columns:
#         print("Column 'year_model' not found, using a default value")
#         df_processed['year_model'] = 2015  # Use a reasonable default value
    
#     # Car age feature (current year - model year) with safety checks
#     df_processed['car_age'] = df_processed['year_venta'] - df_processed['year_model']
    
#     # Handle negative or extreme car ages (data errors)
#     df_processed['car_age'] = df_processed['car_age'].clip(lower=0, upper=50)
    
#     # Ensure kilometers exists
#     if 'kilometers' not in df_processed.columns:
#         print("Column 'kilometers' not found, using a default value")
#         df_processed['kilometers'] = 50000  # Use a reasonable default value
    
#     # Ensure kilometers is not zero to avoid division by zero
#     df_processed['kilometers'] = df_processed['kilometers'].replace(0, 1)
    
#     # Price per km (to be used for outlier detection only)
#     if 'pricing' in df_processed.columns:
#         df_processed['price_per_km'] = df_processed['pricing'] / df_processed['kilometers']
#         # Cap extreme values
#         q999 = df_processed['price_per_km'].quantile(0.999)
#         q001 = df_processed['price_per_km'].quantile(0.001)
#         df_processed['price_per_km'] = df_processed['price_per_km'].clip(lower=q001, upper=q999)
    
#     # Mileage per year (annual usage intensity) with safety checks
#     df_processed['km_per_year'] = df_processed['kilometers'] / (df_processed['car_age'] + 1)  # Add 1 to avoid division by zero
#     # Cap extreme values
#     q999 = df_processed['km_per_year'].quantile(0.999)
#     q001 = df_processed['km_per_year'].quantile(0.001)
#     df_processed['km_per_year'] = df_processed['km_per_year'].clip(lower=q001, upper=q999)
    
#     # Demand-popularity interaction if both columns exist
#     if all(col in df_processed.columns for col in ['demand', 'Popularidad']):
#         df_processed['demand_popularity'] = df_processed['demand'] * df_processed['Popularidad']
#         # Check for and handle extreme values
#         q999 = df_processed['demand_popularity'].quantile(0.999)
#         q001 = df_processed['demand_popularity'].quantile(0.001)
#         df_processed['demand_popularity'] = df_processed['demand_popularity'].clip(lower=q001, upper=q999)
    
#     # Categorical encoding of year_model into age groups
#     df_processed['age_group'] = pd.cut(df_processed['car_age'], 
#                                       bins=[-1, 2, 5, 10, 20, 100], 
#                                       labels=['Very New', 'New', 'Medium', 'Old', 'Very Old'])
    
#     # Create season from month
#     df_processed['season'] = (df_processed['month_venta'] % 12 + 3) // 3
    
#     # Identify different types of columns
#     categorical_cols = []
#     if 'combustible_int' in df_processed.columns:
#         categorical_cols.append('combustible_int')
#     categorical_cols.extend(['age_group', 'season'])
    
#     target_encode_cols = []
#     if 'Cod_fasecolda' in df_processed.columns:
#         target_encode_cols.append('Cod_fasecolda')
    
#     numerical_cols = []
#     for col in ['kilometers', 'year_model', 'description_int', 'gamma_int', 
#                'demand', 'Popularidad', 'year_venta', 'month_venta',
#                'car_age', 'km_per_year']:
#         if col in df_processed.columns:
#             numerical_cols.append(col)
    
#     if 'demand_popularity' in df_processed.columns:
#         numerical_cols.append('demand_popularity')
    
#     binary_cols = []
#     if 'Blindaje' in df_processed.columns:
#         binary_cols.append('Blindaje')
    
#     # Define target variable and log transform
#     if 'pricing' in df_processed.columns:
#         # Check for and remove extreme outliers manually before the log transform
#         q1 = df_processed['pricing'].quantile(0.01)
#         q3 = df_processed['pricing'].quantile(0.99)
#         iqr = q3 - q1
#         lower_bound = max(0, q1 - 1.5 * iqr)  # Ensure it's not negative
#         upper_bound = q3 + 1.5 * iqr
        
#         outliers_mask = (df_processed['pricing'] >= lower_bound) & (df_processed['pricing'] <= upper_bound)
#         df_processed = df_processed[outliers_mask]
        
#         # Make sure pricing is positive before log transform
#         df_processed['pricing'] = df_processed['pricing'].clip(lower=1)
        
#         # Apply log transformation to the target
#         print("Applying log transformation to target variable...")
#         y_original = df_processed['pricing'].copy()
#         y = np.log1p(df_processed['pricing'])
#         print(f"Original pricing range: {y_original.min()} to {y_original.max()}")
#         print(f"Log-transformed pricing range: {y.min():.2f} to {y.max():.2f}")
#     else:
#         print("WARNING: 'pricing' column not found in dataset. Model cannot be trained without target variable.")
#         return None, None, None, None, None, None, None, None
    
#     # Input features (all columns except the target and any helper columns used for preprocessing)
#     drop_cols = ['pricing']
#     if 'price_per_km' in df_processed.columns:
#         drop_cols.append('price_per_km')
    
#     X = df_processed.drop(drop_cols, axis=1)
    
#     # Convert categorical columns to string to ensure proper encoding
#     for col in categorical_cols:
#         if col in X.columns:
#             X[col] = X[col].astype(str)
    
#     # Keep track of column names for later interpretation
#     feature_names = numerical_cols + categorical_cols + target_encode_cols + binary_cols
    
#     # Final check for infinity or extreme values
#     for col in X.select_dtypes(include=[np.number]).columns:
#         # Replace infinities with NaN
#         X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        
#         # Replace NaNs with median
#         X[col] = X[col].fillna(X[col].median())
        
#         # Clip extreme values
#         q999 = X[col].quantile(0.999)
#         q001 = X[col].quantile(0.001)
#         X[col] = X[col].clip(lower=q001, upper=q999)
    
#     print("Data preprocessing completed successfully.")
    
#     return X, y, y_original, numerical_cols, categorical_cols, target_encode_cols, binary_cols, feature_names

# # 3. Build model with advanced techniques
# def build_and_train_model(X, y, numerical_cols, categorical_cols, target_encode_cols, binary_cols):
#     # Split the data into training and test sets (80% train, 20% test)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
#     # Add the InfinityHandler to the preprocessing pipeline
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', Pipeline([
#                 ('infinity_handler', InfinityHandler()),
#                 ('scaler', StandardScaler()),
#                 ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True))
#             ]), numerical_cols),
#             ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
#             ('target_enc', TargetEncoder(smoothing=10), target_encode_cols),
#             # Binary columns pass through without transformation
#         ],
#         remainder='passthrough'  # This will pass through the binary columns
#     )
    
#     # MULTIPLE MODELS - Create a stacking ensemble with safer hyperparameters
    
#     # Base models with reduced complexity to avoid overfitting
#     rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
#     gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
#     xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=-1)
#     lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=-1)
    
#     # Create the stacking ensemble
#     estimators = [
#         ('rf', rf),
#         ('gbr', gbr),
#         ('xgb', xgb_model),
#         ('lgb', lgb_model)
#     ]
    
#     # Final meta-model
#     final_estimator = Ridge(alpha=1.0)
    
#     # Create a stacking regressor
#     stacked_model = StackingRegressor(
#         estimators=estimators,
#         final_estimator=final_estimator,
#         cv=5,
#         n_jobs=-1
#     )
    
#     # Create the full pipeline
#     model = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('regressor', stacked_model)
#     ])
    
#     # Train the model with error handling
#     print("Training the stacked ensemble model...")
#     try:
#         model.fit(X_train, y_train)
#         print("Model training completed!")
#         return model, X_train, X_test, y_train, y_test
#     except Exception as e:
#         print(f"Error training model: {e}")
#         # Try a simpler model if the ensemble fails
#         print("Attempting to train a simpler Random Forest model...")
#         simple_model = Pipeline(steps=[
#             ('preprocessor', preprocessor),
#             ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
#         ])
#         simple_model.fit(X_train, y_train)
#         print("Simple model training completed!")
#         return simple_model, X_train, X_test, y_train, y_test

# # 4. Evaluate the model with visualization of metrics
# def evaluate_model(model, X, y, y_original, X_test, y_test, numerical_cols, categorical_cols, target_encode_cols, binary_cols):
#     # Make predictions on the test set
#     try:
#         y_pred_log = model.predict(X_test)
        
#         # Transform predictions back to original scale
#         y_pred = np.expm1(y_pred_log)
#         y_test_original = np.expm1(y_test)
        
#         # Calculate performance metrics on log scale
#         mse_log = mean_squared_error(y_test, y_pred_log)
#         rmse_log = np.sqrt(mse_log)
#         r2_log = r2_score(y_test, y_pred_log)
#         mae_log = mean_absolute_error(y_test, y_pred_log)
        
#         # Calculate performance metrics on original scale
#         mse = mean_squared_error(y_test_original, y_pred)
#         rmse = np.sqrt(mse)
#         r2 = r2_score(y_test_original, y_pred)
#         mae = mean_absolute_error(y_test_original, y_pred)
        
#         # Calculate percentage error
#         mape = np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100
        
#         print(f"\nModel Evaluation Metrics (Log Scale):")
#         print(f"Mean Squared Error (MSE): {mse_log:.6f}")
#         print(f"Root Mean Squared Error (RMSE): {rmse_log:.6f}")
#         print(f"Mean Absolute Error (MAE): {mae_log:.6f}")
#         print(f"R² Score: {r2_log:.4f}")
        
#         print(f"\nModel Evaluation Metrics (Original Scale):")
#         print(f"Mean Squared Error (MSE): {mse:.2f}")
#         print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
#         print(f"Mean Absolute Error (MAE): {mae:.2f}")
#         print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
#         print(f"R² Score: {r2:.4f}")
        
#         # Create a figure with multiple subplots for different evaluation metrics
#         fig = plt.figure(figsize=(20, 15))
        
#         # 1. Actual vs Predicted values plot (log scale)
#         ax1 = fig.add_subplot(2, 3, 1)
#         ax1.scatter(y_test, y_pred_log, alpha=0.5, color='blue')
#         ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
#         ax1.set_xlabel('Actual Car Prices (log scale)', fontsize=12)
#         ax1.set_ylabel('Predicted Car Prices (log scale)', fontsize=12)
#         ax1.set_title('Actual vs Predicted Car Prices (Log Scale)', fontsize=14)
        
#         # 2. Actual vs Predicted values plot (original scale)
#         ax2 = fig.add_subplot(2, 3, 2)
#         ax2.scatter(y_test_original, y_pred, alpha=0.5, color='green')
#         ax2.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--')
#         ax2.set_xlabel('Actual Car Prices', fontsize=12)
#         ax2.set_ylabel('Predicted Car Prices', fontsize=12)
#         ax2.set_title('Actual vs Predicted Car Prices (Original Scale)', fontsize=14)
        
#         # 3. Residuals plot (log scale)
#         ax3 = fig.add_subplot(2, 3, 3)
#         residuals_log = y_test - y_pred_log
#         ax3.scatter(y_pred_log, residuals_log, alpha=0.5, color='purple')
#         ax3.axhline(y=0, color='r', linestyle='--')
#         ax3.set_xlabel('Predicted Car Prices (log scale)', fontsize=12)
#         ax3.set_ylabel('Residuals (log scale)', fontsize=12)
#         ax3.set_title('Residual Plot (Log Scale)', fontsize=14)
        
#         # 4. Distribution of residuals (histogram, log scale)
#         ax4 = fig.add_subplot(2, 3, 4)
#         sns.histplot(residuals_log, kde=True, ax=ax4, color='orange')
#         ax4.axvline(x=0, color='r', linestyle='--')
#         ax4.set_xlabel('Residuals (log scale)', fontsize=12)
#         ax4.set_ylabel('Frequency', fontsize=12)
#         ax4.set_title('Distribution of Residuals (Log Scale)', fontsize=14)
        
#         # 5. Percentage error distribution
#         ax5 = fig.add_subplot(2, 3, 5)
#         percentage_error = ((y_test_original - y_pred) / y_test_original) * 100
#         # Clip extreme values for better visualization
#         percentage_error_clipped = np.clip(percentage_error, -100, 100)
#         sns.histplot(percentage_error_clipped, kde=True, ax=ax5, color='brown')
#         ax5.axvline(x=0, color='r', linestyle='--')
#         ax5.set_xlabel('Percentage Error (%)', fontsize=12)
#         ax5.set_ylabel('Frequency', fontsize=12)
#         ax5.set_title('Percentage Error Distribution', fontsize=14)
        
#         # 6. Error by price segment
#         ax6 = fig.add_subplot(2, 3, 6)
#         # Create price segments
#         price_segments = pd.qcut(y_test_original, 5)
#         segment_errors = pd.DataFrame({
#             'segment': price_segments,
#             'abs_error': np.abs(y_test_original - y_pred),
#             'percentage_error': np.abs(percentage_error)
#         })
#         segment_mean_errors = segment_errors.groupby('segment')['percentage_error'].mean()
#         segment_mean_errors.plot(kind='bar', ax=ax6, color='teal')
#         ax6.set_xlabel('Price Segment', fontsize=12)
#         ax6.set_ylabel('Mean Absolute Percentage Error (%)', fontsize=12)
#         ax6.set_title('Error by Price Segment', fontsize=14)
#         ax6.tick_params(axis='x', rotation=45)
        
#         plt.tight_layout()
#         fig.suptitle('Model Evaluation Metrics Visualization', fontsize=16)
#         plt.subplots_adjust(top=0.92)
#         plt.show()
        
#         # Cross-validation with robust error handling
#         try:
#             # Cross-validation with original scale conversion
#             cv = KFold(n_splits=5, shuffle=True, random_state=42)
            
#             def rmse_exp_transform(estimator, X, y):
#                 try:
#                     y_pred = estimator.predict(X)
#                     return -np.sqrt(mean_squared_error(np.expm1(y), np.expm1(y_pred)))
#                 except:
#                     return -np.inf  # Return a large error if transformation fails
            
#             def r2_exp_transform(estimator, X, y):
#                 try:
#                     y_pred = estimator.predict(X)
#                     return r2_score(np.expm1(y), np.expm1(y_pred))
#                 except:
#                     return -np.inf  # Return a poor score if transformation fails
            
#             def mape_exp_transform(estimator, X, y):
#                 try:
#                     y_pred = estimator.predict(X)
#                     y_true_orig = np.expm1(y)
#                     y_pred_orig = np.expm1(y_pred)
#                     return -np.mean(np.abs((y_true_orig - y_pred_orig) / y_true_orig)) * 100
#                 except:
#                     return -np.inf  # Return a large error if transformation fails
            
#             print("\nPerforming cross-validation...")
#             cv_scores_rmse = cross_val_score(model, X, y, cv=cv, scoring=rmse_exp_transform)
#             cv_scores_r2 = cross_val_score(model, X, y, cv=cv, scoring=r2_exp_transform)
#             cv_scores_mape = cross_val_score(model, X, y, cv=cv, scoring=mape_exp_transform)
            
#             # Filter out any -inf values that might have occurred
#             cv_scores_rmse = cv_scores_rmse[cv_scores_rmse > -np.inf]
#             cv_scores_r2 = cv_scores_r2[cv_scores_r2 > -np.inf]
#             cv_scores_mape = cv_scores_mape[cv_scores_mape > -np.inf]
            
#             if len(cv_scores_rmse) > 0:
#                 print(f"Cross-validated RMSE: {-np.mean(cv_scores_rmse):.2f} (±{np.std(cv_scores_rmse):.2f})")
#             if len(cv_scores_r2) > 0:
#                 print(f"Cross-validated R²: {np.mean(cv_scores_r2):.4f} (±{np.std(cv_scores_r2):.4f})")
#             if len(cv_scores_mape) > 0:
#                 print(f"Cross-validated MAPE: {-np.mean(cv_scores_mape):.2f}% (±{np.std(cv_scores_mape):.2f})")
#         except Exception as e:
#             print(f"Error during cross-validation: {e}")
#             cv_scores_rmse = []
#             cv_scores_r2 = []
#             cv_scores_mape = []
        
#         # Extract and plot feature importances - using a simpler approach for robustness
#         print("\nCalculating feature importances...")
#         try:
#             # Check if the model is a pipeline and if the regressor is a stacking regressor
#             if hasattr(model, 'named_steps') and hasattr(model.named_steps['regressor'], 'estimators_'):
#                 # Extract the first base estimator that has feature_importances_
#                 for name, estimator in model.named_steps['regressor'].estimators_:
#                     if hasattr(estimator, 'feature_importances_'):
#                         feature_importances = estimator.feature_importances_
#                         break
#             elif hasattr(model, 'named_steps') and hasattr(model.named_steps['regressor'], 'feature_importances_'):
#                 # Extract feature importances from the regressor directly
#                 feature_importances = model.named_steps['regressor'].feature_importances_
#             else:
#                 # Train a separate random forest to get feature importances
#                 rf_for_importance = RandomForestRegressor(n_estimators=50, random_state=42)
#                 X_processed = model.named_steps['preprocessor'].transform(X_train)
#                 rf_for_importance.fit(X_processed, y_train)
#                 feature_importances = rf_for_importance.feature_importances_
            
#             # Get feature names from preprocessor if possible
#             try:
#                 feature_names_out = model.named_steps['preprocessor'].get_feature_names_out()
#                 sorted_importances = sorted(zip(feature_names_out, feature_importances), key=lambda x: x[1], reverse=True)
#             except Exception as e:
#                 print(f"Could not extract feature names: {e}")
#                 # Just use indices as feature names
#                 sorted_importances = sorted([(f"Feature {i}", imp) for i, imp in enumerate(feature_importances)], 
#                                         key=lambda x: x[1], reverse=True)
            
#             # Print top 15 feature importances
#             print("\nTop 15 Feature Importances:")
#             for i, (feature, importance) in enumerate(sorted_importances[:15]):
#                 print(f"{i+1}. {feature}: {importance:.4f}")
            
#             # Plot feature importances
#             plt.figure(figsize=(12, 8))
#             # Get top 15 features (or fewer if there aren't 15)
#             top_n = min(15, len(sorted_importances))
#             top_features = dict(sorted_importances[:top_n])
            
#             plt.barh(list(top_features.keys()), list(top_features.values()))
#             plt.xlabel('Importance')
#             plt.ylabel('Feature')
#             plt.title(f'Top {top_n} Feature Importances')
#             plt.tight_layout()
#             plt.show()
            
#         except Exception as e:
#             print(f"Error calculating feature importances: {e}")
#             sorted_importances = []
        
#         return {
#             'mse': mse,
#             'rmse': rmse,
#             'r2': r2,
#             'mae': mae,
#             'mape': mape,
#             'y_pred': y_pred,
#             'feature_importances': dict(sorted_importances) if len(sorted_importances) > 0 else {},
#             'cross_val_scores': {
#                 'rmse': -cv_scores_rmse if len(cv_scores_rmse) > 0 else [],
#                 'r2': cv_scores_r2 if len(cv_scores_r2) > 0 else [],
#                 'mape': -cv_scores_mape if len(cv_scores_mape) > 0 else []
#             }
#         }
#     except Exception as e:
#         print(f"Error during model evaluation: {e}")
#         return {
#             'error': str(e),
#             'mse': None,
#             'rmse': None,
#             'r2': None,
#             'mae': None,
#             'mape': None,
#             'y_pred': None,
#             'feature_importances': {},
#             'cross_val_scores': {
#                 'rmse': [],
#                 'r2': [],
#                 'mape': []
#             }
#         }

# # Main function to run the entire process
# def main(file_path='car_data.csv'):
#     # Load the data
#     df = load_data(file_path)
    
#     if df is None:
#         return
    
#     # Preprocess the data with advanced feature engineering
#     X, y, y_original, numerical_cols, categorical_cols, target_encode_cols, binary_cols, feature_names = preprocess_data(df)
    
#     if X is None or y is None:
#         print("Error during preprocessing. Cannot continue.")
#         return
    
#     # Final check for any remaining problematic values before model building
#     for col in X.select_dtypes(include=[np.number]).columns:
#         # Check for and fix infinity values
#         if np.isinf(X[col]).any():
#             print(f"Warning: Column {col} still has infinity values after preprocessing. Fixing...")
#             X[col] = X[col].replace([np.inf, -np.inf], np.nan)
#             X[col] = X[col].fillna(X[col].median())
        
#         # Check for extremely large values
#         max_val = X[col].max()
#         if max_val > 1e10:  # Very large threshold
#             print(f"Warning: Column {col} has extremely large values. Scaling down...")
#             X[col] = X[col] / 1000  # Scale down by a factor
    
#     # Build and train the model with advanced techniques
#     try:
#         model, X_train, X_test, y_train, y_test = build_and_train_model(X, y, numerical_cols, categorical_cols, target_encode_cols, binary_cols)
        
#         # Evaluate the model with visualizations
#         results = evaluate_model(model, X, y, y_original, X_test, y_test, numerical_cols, categorical_cols, target_encode_cols, binary_cols)
        
#         print("\nModel training and evaluation completed successfully!")
#         print("\nSummary of Improvements Made:")
#         print("1. Added robust handling for infinity and extreme values")
#         print("2. Log transformation of target variable to handle skewed price distribution")
#         print("3. Advanced feature engineering (car age, km_per_year, demand-popularity interaction, etc.)")
#         print("4. Outlier removal and capping to improve model stability")
#         print("5. Polynomial feature interactions for numerical variables")
#         print("6. Ensemble model stacking with error handling fallback to simpler models")
#         print("7. Age grouping categorical feature to capture non-linear depreciation")
#         print("8. Seasonal effects modeling with the season feature")
#         print("9. Added comprehensive error handling throughout the pipeline")
#         print("\nThe model now reports metrics on both log-transformed and original scales.")
#         print("The mean absolute percentage error (MAPE) provides an intuitive measure of prediction accuracy.")
        
#         return model, results
#     except Exception as e:
#         print(f"Critical error during model building or evaluation: {e}")
#         print("Attempting to build a very simple model as fallback...")
        
#         # Fallback to an extremely simple model
#         simple_model = Pipeline([
#             ('scaler', StandardScaler()),
#             ('estimator', RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42))
#         ])
        
#         # Keep only the most reliable numerical features
#         safe_features = []
#         for col in X.columns:
#             if X[col].dtype in [np.float64, np.int64] and not np.isinf(X[col]).any() and not np.isnan(X[col]).any():
#                 safe_features.append(col)
                
#         print(f"Training with {len(safe_features)} safe features: {safe_features}")
        
#         # Use only these safe features for the fallback model
#         X_safe = X[safe_features]
        
#         # Split the data
#         X_train_safe, X_test_safe, y_train, y_test = train_test_split(X_safe, y, test_size=0.2, random_state=42)
        
#         # Train the simple model
#         simple_model.fit(X_train_safe, y_train)
        
#         # Evaluate
#         y_pred_log = simple_model.predict(X_test_safe)
#         y_pred = np.expm1(y_pred_log)
#         y_test_original = np.expm1(y_test)
        
#         mse = mean_squared_error(y_test_original, y_pred)
#         r2 = r2_score(y_test_original, y_pred)
#         mae = mean_absolute_error(y_test_original, y_pred)
        
#         print(f"\nFallback Model Results:")
#         print(f"MSE: {mse:.2f}")
#         print(f"R²: {r2:.4f}")
#         print(f"MAE: {mae:.2f}")
        
#         return simple_model, {
#             'mse': mse,
#             'r2': r2,
#             'mae': mae,
#             'fallback': True
#         }

# # Run the script if executed directly
# if __name__ == "__main__":
#     model, results = main()
#     print(results)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Function to handle infinity and extreme values
def clean_dataset(df):
    """Clean the dataset by handling infinity and extreme values."""
    # Make a copy to avoid warnings
    df_clean = df.copy()
    
    # Handle infinity values
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        # Replace infinity values with NaN
        df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
        
        # Calculate safe bounds for clipping (using quantiles to avoid being affected by extreme values)
        if df_clean[col].notnull().any():  # Only if we have some non-null values
            q_low = df_clean[col].quantile(0.001)
            q_high = df_clean[col].quantile(0.999)
            
            # Clip values to remove extremes
            df_clean[col] = df_clean[col].clip(lower=q_low, upper=q_high)
        
        # Fill remaining NaN values with median
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    return df_clean

# Load the dataset
df = pd.read_csv('car_data.csv')
print(f"Original dataset shape: {df.shape}")

# Clean the dataset
df_clean = clean_dataset(df)
print("Data cleaned - infinity values and extremes handled")
try:
    df_clean['fecha_venta'] = pd.to_datetime(df_clean['fecha_venta'])
    df_clean['year_venta'] = df_clean['fecha_venta'].dt.year
    df_clean['month_venta'] = df_clean['fecha_venta'].dt.month
    # Drop the original fecha_venta column
    df_clean.drop('fecha_venta', axis=1, inplace=True)
except Exception as e:
    print(f"Error processing date column: {e}")
    
# Define features and target (adjust column names as needed)
X = df_clean.drop('pricing', axis=1)  # Assuming 'pricing' is your target
y = df_clean['pricing']

# Log-transform the target if it's skewed (common for prices)
y_log = np.log1p(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Create a simple pipeline
model = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
print("Training model...")
model.fit(X_train, y_train)
print("Model training completed!")

# Make predictions and evaluate
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
r2 = r2_score(y_test_orig, y_pred)
mape = np.mean(np.abs((y_test_orig - y_pred) / y_test_orig)) * 100

print(f"\nModel Evaluation Results:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")