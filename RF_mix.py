# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, r2_score

# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor

# # Clean the dataset: infinity, extreme values, NaNs
# def clean_dataset(df):
#     df_clean = df.copy()
#     for col in df_clean.select_dtypes(include=[np.number]).columns:
#         df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
#         if df_clean[col].notnull().any():
#             q_low = df_clean[col].quantile(0.001)
#             q_high = df_clean[col].quantile(0.999)
#             df_clean[col] = df_clean[col].clip(lower=q_low, upper=q_high)
#         df_clean[col] = df_clean[col].fillna(df_clean[col].median())
#     return df_clean

# # Load dataset
# df = pd.read_csv('car_data.csv')
# print(f"Original dataset shape: {df.shape}")

# # Clean
# df_clean = clean_dataset(df)
# print("Data cleaned")

# # Process date
# try:
#     df_clean['fecha_venta'] = pd.to_datetime(df_clean['fecha_venta'])
#     df_clean['year_venta'] = df_clean['fecha_venta'].dt.year
#     df_clean['month_venta'] = df_clean['fecha_venta'].dt.month
#     df_clean.drop('fecha_venta', axis=1, inplace=True)
# except Exception as e:
#     print(f"Date error: {e}")

# # Separate features and target
# X = df_clean.drop(columns=['pricing'])
# y = df_clean['pricing']
# y_log = np.log1p(y)

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# # Define models
# models = {
#     "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
#     "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
#     "LightGBM": LGBMRegressor(n_estimators=100, random_state=42)
# }

# results = {}
# output_df = X_test.copy()
# y_test_orig = np.expm1(y_test)

# # Train + Evaluate each model
# for name, model in models.items():
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('regressor', model)
#     ])
    
#     print(f" Training {name}...")
#     pipeline.fit(X_train, y_train)
    
#     y_pred_log = pipeline.predict(X_test)
#     y_pred = np.expm1(y_pred_log)

#     rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
#     r2 = r2_score(y_test_orig, y_pred)
#     mape = np.mean(np.abs((y_test_orig - y_pred) / y_test_orig)) * 100

#     print(f" {name} Results:")
#     print(f"   - RMSE: {rmse:.2f}")
#     print(f"   - R²: {r2:.4f}")
#     print(f"   - MAPE: {mape:.2f}%")

#     # Store predictions and residuals
#     pred_col = f"{name.lower().replace(' ', '_')}_pred"
#     resid_col = f"{name.lower().replace(' ', '_')}_residual"
#     output_df[pred_col] = y_pred
#     output_df[resid_col] = y_test_orig - y_pred

# # Add actual values to output
# output_df['actual_pricing'] = y_test_orig

# # Save to CSV
# output_df.to_csv("car_data_model_results.csv", index=False)
# print(" Results saved to car_data_model_results.csv")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

#  Clean the dataset
def clean_dataset(df):
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
        if df_clean[col].notnull().any():
            q_low = df_clean[col].quantile(0.001)
            q_high = df_clean[col].quantile(0.999)
            df_clean[col] = df_clean[col].clip(lower=q_low, upper=q_high)
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    return df_clean

#  Load and preprocess
df = pd.read_csv('car_data.csv')
df_clean = clean_dataset(df)

try:
    df_clean['fecha_venta'] = pd.to_datetime(df_clean['fecha_venta'])
    df_clean['year_venta'] = df_clean['fecha_venta'].dt.year
    df_clean['month_venta'] = df_clean['fecha_venta'].dt.month
    df_clean.drop('fecha_venta', axis=1, inplace=True)
except Exception as e:
    print(f"Date error: {e}")

X = df_clean.drop(columns=['pricing'])
y = df_clean['pricing']
y_log = np.log1p(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
y_test_orig = np.expm1(y_test)

#  Define models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    "LightGBM": LGBMRegressor(n_estimators=100, random_state=42)
}

output_df = X_test.copy()
output_df['actual_pricing'] = y_test_orig

#  Train, evaluate, and plot
for name, model in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', model)
    ])
    pipeline.fit(X_train, y_train)

    y_pred_log = pipeline.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    r2 = r2_score(y_test_orig, y_pred)
    mape = np.mean(np.abs((y_test_orig - y_pred) / y_test_orig)) * 100

    print(f" {name}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   R²: {r2:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    pred_col = f"{name.lower().replace(' ', '_')}_pred"
    resid_col = f"{name.lower().replace(' ', '_')}_resid"
    output_df[pred_col] = y_pred
    output_df[resid_col] = y_test_orig - y_pred

    #  PLOTS
    plt.figure(figsize=(16, 4))

    # Actual vs Predicted
    plt.subplot(1, 3, 1)
    sns.scatterplot(x=y_test_orig, y=y_pred, alpha=0.6)
    plt.plot([y_test_orig.min(), y_test_orig.max()],
         [y_test_orig.min(), y_test_orig.max()],
         color='red', linestyle='--') 
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"{name} - Actual vs Predicted")

    # Residual Plot
    plt.subplot(1, 3, 2)
    sns.histplot(y_test_orig - y_pred, kde=True, bins=30)
    plt.title(f"{name} - Residual Distribution")

    # Feature Importance (if available)
    plt.subplot(1, 3, 3)
    try:
        importances = model.feature_importances_
        feature_names = X.columns
        indices = np.argsort(importances)[::-1][:10]
        sns.barplot(x=importances[indices], y=feature_names[indices])
        plt.title(f"{name} - Top 10 Feature Importances")
    except:
        plt.text(0.1, 0.5, "Feature importances not available", fontsize=12)

    plt.tight_layout()
    plt.show()

# Save final output
output_df.to_csv("car_data_model_results.csv", index=False)
print(" Output saved as car_data_model_results.csv")
