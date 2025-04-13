import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)

# Number of samples to simulate
n_samples = 1000

# Simulate data
data = {
    'Cod_fasecolda': np.random.choice([str(100567 + i) for i in range(100)], size=n_samples),
    'kilometers': np.random.randint(0, 300000, size=n_samples),
    'year_model': np.random.randint(2000, 2024, size=n_samples),
    'description_int': np.random.randint(0, 6, size=n_samples),
    'gamma_int': np.random.randint(0, 5, size=n_samples),
    'demand': np.round(np.random.uniform(0, 5, size=n_samples), 2),
    'Popularidad': np.round(np.random.uniform(0, 5, size=n_samples), 2),
    'combustible_int': np.random.randint(0, 5, size=n_samples),
    'fecha_venta': [
        (datetime(2015, 1, 1) + timedelta(days=np.random.randint(0, 365*8))).strftime('%Y-%m-%d')
        for _ in range(n_samples)
    ],
    
    'Blindaje': np.random.choice([0, 1], size=n_samples),
    'pricing': np.round(np.random.uniform(5000, 60000, size=n_samples), 2)
}

# Create DataFrame
df_simulated = pd.DataFrame(data)

# Save to CSV
df_simulated.to_csv("car_data.csv", index=False)

print("Simulated dataset 'car_data.csv' created successfully.")
