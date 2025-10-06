import os
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Feature engineering
def create_damage_potential(mag, depth):
    return 0.6 * mag + 0.2 * (700 - depth) / 700 * 10

def create_risk_category(score):
    if score < 4.0: return 0
    if score < 6.0: return 1
    return 2

# Get the absolute path of the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR)) # Go up two levels to the project root
DATA_PATH = os.path.join(PROJECT_DIR, "Data_base_files", "preprocessed_earthquake_data.csv")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

# Load data
df = pd.read_csv(DATA_PATH)

# Recommended numeric features only
features = [
    "Latitude",
    "Longitude",
    "Depth",
    "Magnitude",
    "Root Mean Square"
]

# Drop missing
df = df.dropna(subset=features).copy()

# Targets
df["Damage_Potential"] = create_damage_potential(df.Magnitude, df.Depth)
df["Risk_Category"]    = df["Damage_Potential"].apply(create_risk_category)

# Prepare feature matrix and target vector
X = df[features]
y = df["Damage_Potential"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train LightGBM regressor
model = lgb.LGBMRegressor(
    n_estimators=600,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.3f}, R²: {r2:.3f}")

# Save model and feature list
os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(
    {"model": model, "features": features},
    os.path.join(MODELS_DIR, "lgb_damage_model.pkl") # This path is correct, ensure script is run
)
print("Saved model and feature list")
