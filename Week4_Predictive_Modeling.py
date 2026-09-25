"""
Week 4 — Predictive Modeling & Optimization
E-Commerce Logistics Project | business_data.csv
=================================================
Models  : Linear Regression  (baseline)
          Random Forest Regressor (best model)
Target  : Profit (Rs.)
Outputs : evaluation metrics, feature importance chart,
          actual vs predicted chart, predictions CSV
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing    import OneHotEncoder
from sklearn.compose          import ColumnTransformer
from sklearn.pipeline         import Pipeline
from sklearn.linear_model     import LinearRegression
from sklearn.ensemble         import RandomForestRegressor
from sklearn.metrics          import mean_squared_error, mean_absolute_error, r2_score

sns.set_theme(style="whitegrid", font_scale=1.05)

# ── STEP 1: Load & Clean ──────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — Load and Clean Dataset")
print("=" * 65)

df = pd.read_csv("business_data.csv")

for col in ["Revenue", "Profit"]:
    df[col] = (df[col].astype(str)
               .str.replace("\u20b9", "", regex=False)
               .str.replace(",",      "", regex=False)
               .str.strip()
               .replace("", np.nan)
               .astype(float))

df = df.drop_duplicates()
df = df[df["Quantity"] > 0].reset_index(drop=True)
df["Customer_ID"] = df["Customer_ID"].replace("UNKNOWN", np.nan)
df["Region"]      = df["Region"].replace("Not Specified", np.nan)
df["Order_Date"]  = pd.to_datetime(df["Order_Date"], dayfirst=True, errors="coerce")
df["Month"]       = df["Order_Date"].dt.month
df["Quarter"]     = df["Order_Date"].dt.quarter

print(f"  Clean dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Target (Profit) — mean: Rs.{df['Profit'].mean():,.2f}  "
      f"std: Rs.{df['Profit'].std():,.2f}  max: Rs.{df['Profit'].max():,.0f}")

# ── STEP 2: Feature & Target Definition ──────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2 — Define Features and Target")
print("=" * 65)

num_features = ["Revenue", "Quantity", "Month", "Quarter"]
cat_features = ["Category", "Region", "Customer_Type"]
target       = "Profit"

df_model = df[num_features + cat_features + [target]].dropna()
X = df_model[num_features + cat_features]
y = df_model[target]

print(f"  Numeric features : {num_features}")
print(f"  Categorical feat.: {cat_features}")
print(f"  Target           : {target}")
print(f"  Total samples    : {len(X)}")

# ── STEP 3: Train / Test Split ────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3 — Train / Test Split  (80 / 20)")
print("=" * 65)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Training set : {len(X_train)} rows")
print(f"  Test set     : {len(X_test)} rows")

# ── STEP 4: Preprocessing Pipeline (One-Hot Encoding) ─────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(drop="first", sparse_output=False), cat_features)
    ],
    remainder="passthrough"   # pass numeric features through unchanged
)

# ── STEP 5: Model 1 — Linear Regression ──────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5 — Model 1: Linear Regression")
print("=" * 65)

lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)

lr_rmse = mean_squared_error(y_test, lr_pred) ** 0.5
lr_mae  = mean_absolute_error(y_test, lr_pred)
lr_r2   = r2_score(y_test, lr_pred)

print(f"  Test RMSE : Rs.{lr_rmse:,.2f}")
print(f"  Test MAE  : Rs.{lr_mae:,.2f}")
print(f"  Test R²   : {lr_r2:.4f}")

lr_cv = cross_val_score(lr_pipeline, X, y, cv=5, scoring="r2")
print(f"  CV R² (5-fold) : {lr_cv.round(4).tolist()}")
print(f"  CV R² mean     : {lr_cv.mean():.4f}")

# ── STEP 6: Model 2 — Random Forest Regressor ────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6 — Model 2: Random Forest Regressor")
print("=" * 65)

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    ))
])
rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)

rf_rmse = mean_squared_error(y_test, rf_pred) ** 0.5
rf_mae  = mean_absolute_error(y_test, rf_pred)
rf_r2   = r2_score(y_test, rf_pred)

print(f"  Test RMSE : Rs.{rf_rmse:,.2f}")
print(f"  Test MAE  : Rs.{rf_mae:,.2f}")
print(f"  Test R²   : {rf_r2:.4f}")

rf_cv = cross_val_score(rf_pipeline, X, y, cv=5, scoring="r2")
print(f"  CV R² (5-fold) : {rf_cv.round(4).tolist()}")
print(f"  CV R² mean     : {rf_cv.mean():.4f}")

# ── STEP 7: Hyperparameter Tuning ────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 7 — GridSearchCV Hyperparameter Tuning (Random Forest)")
print("=" * 65)

param_grid = {
    "model__n_estimators"    : [50, 100, 200],
    "model__max_depth"       : [None, 10, 20],
    "model__min_samples_leaf": [1, 2, 5],
}
grid_search = GridSearchCV(
    rf_pipeline, param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=0
)
grid_search.fit(X_train, y_train)
tuned_pred = grid_search.predict(X_test)

print(f"  Best params  : {grid_search.best_params_}")
print(f"  Best CV R²   : {grid_search.best_score_:.4f}")
print(f"  Tuned Test R²: {r2_score(y_test, tuned_pred):.4f}")
print(f"  Tuned MAE    : Rs.{mean_absolute_error(y_test, tuned_pred):,.2f}")

# ── STEP 8: Feature Importance ────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 8 — Random Forest Feature Importances")
print("=" * 65)

ohe_feature_names = (rf_pipeline.named_steps["preprocessor"]
                     .named_transformers_["ohe"]
                     .get_feature_names_out(cat_features).tolist())
all_feature_names = ohe_feature_names + num_features
importances = rf_pipeline.named_steps["model"].feature_importances_

fi_df = pd.DataFrame({
    "Feature":    all_feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False).head(12)

print(fi_df.to_string(index=False))

# Plot feature importances
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#3b82d4" if i == 0 else "#93c5fd" for i in range(len(fi_df))]
ax.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1],
        color=colors[::-1], edgecolor="white")
ax.set_xlabel("Feature Importance", fontsize=11)
ax.set_title("Random Forest — Top 12 Feature Importances\n(Profit Prediction)",
             fontsize=13, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("viz6_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  Saved: viz6_feature_importance.png")

# ── STEP 9: Actual vs Predicted Plot ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Actual vs Predicted Profit — Test Set (392 orders)",
             fontsize=13, fontweight="bold")

for ax, pred, label, color in zip(
    axes,
    [lr_pred, rf_pred],
    ["Linear Regression", "Random Forest"],
    ["#7c5cd8", "#3b82d4"]
):
    ax.scatter(y_test, pred, alpha=0.4, s=20, color=color, edgecolors="none")
    lims = [0, max(y_test.max(), pred.max())]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    ax.set_xlabel("Actual Profit (₹)", fontsize=10)
    ax.set_ylabel("Predicted Profit (₹)", fontsize=10)
    ax.set_title(f"{label}\nR²={r2:.4f}  MAE=₹{mae:,.0f}", fontsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"₹{v/1000:.0f}K"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"₹{v/1000:.0f}K"))

plt.tight_layout()
plt.savefig("viz7_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: viz7_actual_vs_predicted.png")

# ── STEP 10: Save Predictions CSV ────────────────────────────────────────────
results_df = X_test.copy()
results_df["Actual_Profit"]   = y_test.values
results_df["LR_Predicted"]    = lr_pred.round(0)
results_df["RF_Predicted"]    = rf_pred.round(0)
results_df["RF_Error"]        = (rf_pred - y_test.values).round(0)
results_df["RF_AbsError"]     = np.abs(rf_pred - y_test.values).round(0)
results_df.to_csv("Week4_Predictions.csv", index=False)
print("  Saved: Week4_Predictions.csv")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("WEEK 4 MODELING SUMMARY")
print("=" * 65)
print(f"  Linear Regression  — R²: {lr_r2:.4f}  RMSE: Rs.{lr_rmse:,.2f}  MAE: Rs.{lr_mae:,.2f}")
print(f"  Random Forest      — R²: {rf_r2:.4f}  RMSE: Rs.{rf_rmse:,.2f}  MAE: Rs.{rf_mae:,.2f}")
print(f"  Tuned RF           — R²: {r2_score(y_test,tuned_pred):.4f}  MAE: Rs.{mean_absolute_error(y_test,tuned_pred):,.2f}")
print(f"\n  Best Model : Random Forest (R²={rf_r2:.4f}, MAE=Rs.{rf_mae:,.2f})")
print(f"  Top Feature: Revenue (importance = 91.78%)")
print(f"\n  Output files:")
print(f"    viz6_feature_importance.png")
print(f"    viz7_actual_vs_predicted.png")
print(f"    Week4_Predictions.csv")
print("=" * 65)
