import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# --- Load and preprocess dataset ---
df = pd.read_csv("car_data.csv")

# Process date
df['fecha_venta'] = pd.to_datetime(df['fecha_venta'], errors='coerce')
df['year_venta'] = df['fecha_venta'].dt.year
df['month_venta'] = df['fecha_venta'].dt.month
df.drop(columns=['fecha_venta'], inplace=True)

# Encode 'Cod_fasecolda'
df['Cod_fasecolda'] = df['Cod_fasecolda'].astype(str)
label_encoder = LabelEncoder()
df['Cod_fasecolda_encoded'] = label_encoder.fit_transform(df['Cod_fasecolda'])
df.drop(columns=['Cod_fasecolda'], inplace=True)

# Drop missing values
df = df.dropna()

# --- Define features and target ---
feature_cols = [
    'Cod_fasecolda_encoded',
    'kilometers',
    'year_model',
    'description_int',
    'gamma_int',
    'demand',
    'Popularidad',
    'combustible_int',
    'Blindaje',
    'year_venta',
    'month_venta'
]

X = df[feature_cols]
y = df['pricing']

# --- Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Models ---
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    "LightGBM": LGBMRegressor(n_estimators=100, random_state=42)
}

results = {}

# --- Train and evaluate each model ---
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'rmse': rmse,
        'r2': r2
    }
    print(f"📊 {name}")
    print(f"  - RMSE: {rmse:.2f}")
    print(f"  - R²: {r2:.4f}\n")

# --- Plot: Actual vs Predicted ---
plt.figure(figsize=(10, 6))
for name, res in results.items():
    sns.scatterplot(x=y_test, y=res['y_pred'], label=name, alpha=0.6)

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label="Ideal")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot: Feature Importances (Random Forest example) ---
importances = results["Random Forest"]['model'].feature_importances_
feature_imp_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df)
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# --- Output CSV with predictions ---
output_df = X_test.copy()
output_df['actual_pricing'] = y_test

for name, res in results.items():
    pred_col = f"{name.replace(' ', '_').lower()}_pred"
    resid_col = f"{name.replace(' ', '_').lower()}_residual"
    output_df[pred_col] = res['y_pred']
    output_df[resid_col] = y_test - res['y_pred']

# Save to CSV
output_df.to_csv("car_data_output.csv", index=False)
print("car_data_output.csv has been saved with predictions and residuals.")
