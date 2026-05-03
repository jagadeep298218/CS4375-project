# CS4375-project
Final project for CS 4375 for Prof Anurag Nagar

Link to colab: https://colab.research.google.com/drive/1bVrxawPTMPQRfQ8UkKZyqlZhuNuMkIXx?usp=sharing

dataset link: https://www.kaggle.com/datasets/jayc00009/atlantic-hurricane-dataset-hurdat2?resource=download

## Steps Taken
 
**1. Data Loading** — Downloaded HURDAT2 CSV from GitHub into Colab using `wget`.
 
**2. Feature Selection** — Kept `maximum_sustained_wind_knots`, `central_pressure_mb`, `status_of_system`, and `radius_of_max_wind_nm`. Dropped coordinates and irrelevant columns.
 
**3. Cleaning** — Replaced `-999` sentinel values with `NaN`, dropped rows where the target was null, then forward/backward filled remaining nulls within each storm.
 
**4. Encoding** — Mapped storm status labels (HU, TS, TD, etc.) to integers.
 
**5. Train/Test Split** — 80/20 split by storm ID to prevent data leakage.
 
**6. Normalization** — Applied `MinMaxScaler` fit on training data only. Saved a separate wind scaler for inverse transforming predictions.
 
**7. Sequence Creation** — Built sliding windows over each storm's timesteps producing input shape `(samples, seq_len, 4)` with target as the next timestep's wind speed.
 
**8. LSTM (from scratch)** — Implemented using only NumPy with orthogonal weight initialization, four gates (forget, input, candidate, output), a linear output layer, BPTT, and gradient clipping at ±0.5.
 
**9. Experiments** — Ran 8 experiments across `hidden_size` ∈ {64, 128}, `learning_rate` ∈ {0.001, 0.0001}, `seq_len` ∈ {5, 10} for 50 epochs each.
 
**10. Evaluation** — Computed RMSE, MAE, and R² on inverse-transformed predictions.
 
---
