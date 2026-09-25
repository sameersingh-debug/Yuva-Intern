"""
Week 2 — Data Collection, Cleaning & Preprocessing
E-Commerce Logistics Project | business_data.csv
====================================================
Run this script to produce: business_data_clean.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────────────
# STEP 0 — Load Raw Data
# ──────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("business_data.csv")

print("=" * 60)
print("STEP 0 — Raw Data Loaded")
print(f"  Shape        : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Columns      : {list(df.columns)}")
print(f"  Dtypes:\n{df.dtypes}")
print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Currency String Conversion (Revenue & Profit: ₹X,XXX -> float)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1 — Convert Currency Strings to float64")

for col in ["Revenue", "Profit"]:
    before_dtype = df[col].dtype
    df[col] = (
        df[col]
        .astype(str)
        .str.replace("\u20b9", "", regex=False)   # remove ₹ (Unicode U+20B9)
        .str.replace(",", "", regex=False)          # remove thousand separators
        .str.strip()                                # remove whitespace
        .replace("", np.nan)                        # blank string -> NaN
        .astype(float)
    )
    print(f"  {col}: {before_dtype} -> {df[col].dtype}")

print(f"\nRevenue range : ₹{df['Revenue'].min():,.0f} — ₹{df['Revenue'].max():,.0f}")
print(f"Profit range  : ₹{df['Profit'].min():,.0f} — ₹{df['Profit'].max():,.0f}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Remove Duplicate Rows
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Remove Duplicate Rows")

n_before = len(df)
dupes = df[df.duplicated()]
print(f"  Duplicate rows found: {len(dupes)}")
if len(dupes) > 0:
    print(f"  Duplicate Order_IDs  : {sorted(dupes['Order_ID'].unique().tolist())}")

df = df.drop_duplicates().reset_index(drop=True)
print(f"  Rows before: {n_before}  |  Rows after: {len(df)}  |  Removed: {n_before - len(df)}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Replace Sentinel Strings with NaN
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Replace Sentinel Values with NaN")

sentinel_map = {
    "Customer_ID": ["UNKNOWN"],
    "Region":      ["Not Specified"],
    "Product":     ["Not Specified"],
}
for col, sentinels in sentinel_map.items():
    for val in sentinels:
        count = (df[col] == val).sum()
        df[col] = df[col].replace(val, np.nan)
        print(f"  '{val}' in {col}: {count} cells -> NaN")

print(f"\nNull counts after sentinel replacement:")
print(df.isnull().sum().to_string())

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — Impute Missing Region via Category Mode
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Impute Missing Region (Category-Level Mode)")

region_mode = (
    df.groupby("Category")["Region"]
    .apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
)
print(f"  Category-level mode regions:\n{region_mode.to_string()}")

null_region_before = df["Region"].isna().sum()
df["Region"] = df.apply(
    lambda row: region_mode[row["Category"]] if pd.isna(row["Region"]) else row["Region"],
    axis=1,
)
null_region_after = df["Region"].isna().sum()
print(f"\n  Region nulls: {null_region_before} -> {null_region_after}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — Remove Zero-Quantity Orders
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Remove Zero-Quantity Orders")

zero_qty_rows = df[df["Quantity"] == 0]
print(f"  Zero-quantity rows found: {len(zero_qty_rows)}")
if len(zero_qty_rows) > 0:
    print(zero_qty_rows[["Order_ID", "Product", "Region", "Revenue", "Profit"]].to_string())

df = df[df["Quantity"] > 0].reset_index(drop=True)
print(f"  Rows remaining: {len(df)}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — Parse Order_Date, Extract Temporal Features
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Parse Order_Date and Extract Temporal Features")

df["Order_Date"] = pd.to_datetime(df["Order_Date"], dayfirst=True, errors="coerce")
unparsed = df["Order_Date"].isna().sum()
print(f"  Unparseable dates: {unparsed}")

df["Month"]      = df["Order_Date"].dt.month
df["Quarter"]    = df["Order_Date"].dt.quarter
df["DayOfWeek"]  = df["Order_Date"].dt.dayofweek   # 0=Monday, 6=Sunday

print(f"  Date range: {df['Order_Date'].min().date()} to {df['Order_Date'].max().date()}")
print(f"  Months present: {sorted(df['Month'].unique().tolist())}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 7 — Outlier Detection & Winsorization (IQR Method on Revenue)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — Revenue Outlier Detection (IQR Method)")

Q1  = df["Revenue"].quantile(0.25)
Q3  = df["Revenue"].quantile(0.75)
IQR = Q3 - Q1
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

print(f"  Q1={Q1:,.0f}  Q3={Q3:,.0f}  IQR={IQR:,.0f}")
print(f"  Lower fence: {lower_fence:,.0f}  |  Upper fence: {upper_fence:,.0f}")

outliers = df[(df["Revenue"] < lower_fence) | (df["Revenue"] > upper_fence)]
print(f"  Outlier rows: {len(outliers)}")
if len(outliers) > 0:
    print(outliers[["Order_ID", "Product", "Region", "Quantity", "Revenue"]].to_string())

# Winsorize: cap at fences
df["Revenue_clean"] = df["Revenue"].clip(lower=lower_fence, upper=upper_fence)
print(f"\n  Revenue_clean  max: {df['Revenue_clean'].max():,.0f} (capped)")
print(f"  Original Revenue   max: {df['Revenue'].max():,.0f}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 8 — Feature Engineering (Derived KPI Columns)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 — Feature Engineering")

df["Profit_Margin_Rate"]  = (df["Profit"] / df["Revenue"]) * 100
df["Revenue_per_Unit"]    = df["Revenue"] / df["Quantity"]
df["Zero_Profit_Flag"]    = (df["Profit"] == 0).astype(int)

zero_profit_count = df["Zero_Profit_Flag"].sum()
print(f"  Zero-profit rows flagged: {zero_profit_count}")
print(f"\n  Profit_Margin_Rate stats:\n{df['Profit_Margin_Rate'].describe().round(2).to_string()}")
print(f"\n  Revenue_per_Unit stats:\n{df['Revenue_per_Unit'].describe().round(2).to_string()}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 9 — Encode Categorical Columns
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9 — Label Encode Categorical Columns")

from sklearn.preprocessing import LabelEncoder

encode_cols = ["Category", "Region", "Customer_Type", "Profit Category"]
le = LabelEncoder()
for col in encode_cols:
    df[col + "_enc"] = le.fit_transform(df[col].fillna("Unknown"))
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"  {col}: {mapping}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 10 — EDA Quick Plots
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10 — Generating EDA Plots")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Week 2 — EDA after Cleaning  |  business_data.csv", fontsize=13, fontweight="bold")

# Plot 1: Revenue distribution (cleaned)
axes[0, 0].hist(df["Revenue_clean"], bins=35, color="#3b82d4", edgecolor="white")
axes[0, 0].set_title("Revenue Distribution (Winsorized)")
axes[0, 0].set_xlabel("Revenue (₹)")
axes[0, 0].set_ylabel("Count")

# Plot 2: Profit Margin Rate by Region
pmr_region = df.groupby("Region")["Profit_Margin_Rate"].mean().sort_values(ascending=False)
axes[0, 1].barh(pmr_region.index, pmr_region.values, color="#7c5cd8")
axes[0, 1].set_title("Avg Profit Margin Rate by Region (%)")
axes[0, 1].set_xlabel("PMR (%)")

# Plot 3: Order count by Category
cat_counts = df["Category"].value_counts()
axes[1, 0].bar(cat_counts.index, cat_counts.values, color="#3b82d4")
axes[1, 0].set_title("Order Count by Category")
axes[1, 0].tick_params(axis="x", rotation=20)
axes[1, 0].set_ylabel("Orders")

# Plot 4: Profit Margin Rate heatmap (Region x Category)
pivot = df.pivot_table(values="Profit_Margin_Rate", index="Region", columns="Category", aggfunc="mean")
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu",
            linewidths=0.5, ax=axes[1, 1], cbar_kws={"shrink": 0.8})
axes[1, 1].set_title("Avg PMR (%) — Region × Category")

plt.tight_layout()
plt.savefig("Week2_EDA_Plots.png", dpi=150, bbox_inches="tight")
print("  Saved: Week2_EDA_Plots.png")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 11 — Save Cleaned Dataset
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 11 — Save Cleaned Dataset")

df.to_csv("business_data_clean.csv", index=False)

print(f"  Saved: business_data_clean.csv")
print(f"  Final shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nCleaning Summary:")
print(f"  Original rows    : {n_before}")
print(f"  After deduplication + zero-qty removal: {len(df)}")
print(f"  New columns added: Month, Quarter, DayOfWeek, Revenue_clean,")
print(f"                     Profit_Margin_Rate, Revenue_per_Unit,")
print(f"                     Zero_Profit_Flag, *_enc columns")
print("=" * 60)
print("Week 2 cleaning complete. Ready for Week 3 modelling.")
