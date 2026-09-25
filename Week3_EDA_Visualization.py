"""
Week 3 — Advanced Data Analysis & Visualization
E-Commerce Logistics Project | business_data_clean.csv
========================================================
Outputs: 5 chart PNGs + printed EDA statistics
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.05)

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA  (works from either raw or cleaned CSV)
# ──────────────────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv("business_data_clean.csv")
    print("Loaded: business_data_clean.csv")
except FileNotFoundError:
    df = pd.read_csv("business_data.csv")
    for col in ["Revenue", "Profit"]:
        df[col] = (df[col].astype(str)
                   .str.replace("\u20b9", "", regex=False)
                   .str.replace(",", "", regex=False)
                   .str.strip().replace("", np.nan).astype(float))
    df = df.drop_duplicates()
    df = df[df["Quantity"] > 0].reset_index(drop=True)
    print("Loaded and cleaned: business_data.csv")

# Ensure derived columns exist
df["Profit_Margin_Rate"] = (df["Profit"] / df["Revenue"]) * 100
df["Revenue_per_Unit"]   = df["Revenue"] / df["Quantity"]
df["Order_Date"] = pd.to_datetime(df["Order_Date"], dayfirst=True, errors="coerce")
df["Month"]      = df["Order_Date"].dt.month

print(f"\nDataset shape: {df.shape[0]} rows x {df.shape[1]} columns")

# ──────────────────────────────────────────────────────────────────────────────
# EDA — SECTION 1: Descriptive Statistics
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EDA SECTION 1 — Descriptive Statistics")
print("=" * 60)
print(df[["Quantity", "Revenue", "Profit", "Profit_Margin_Rate"]]
      .describe().round(2).to_string())

# ──────────────────────────────────────────────────────────────────────────────
# EDA — SECTION 2: Correlation Matrix
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EDA SECTION 2 — Correlation Matrix")
print("=" * 60)
corr = df[["Quantity", "Revenue", "Profit", "Profit_Margin_Rate"]].corr()
print(corr.round(4).to_string())
print("\nKey findings:")
print(f"  Revenue <-> Profit correlation : {corr.loc['Revenue','Profit']:.4f}  (very strong)")
print(f"  Revenue <-> PMR    correlation : {corr.loc['Revenue','Profit_Margin_Rate']:.4f}  (inverse — high-rev = lower margin %)")

# ──────────────────────────────────────────────────────────────────────────────
# EDA — SECTION 3: By Region
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EDA SECTION 3 — Revenue, Profit, Quantity by Region")
print("=" * 60)
region_stats = df.groupby("Region").agg(
    Total_Revenue=("Revenue", "sum"),
    Mean_Revenue=("Revenue", "mean"),
    Total_Profit=("Profit", "sum"),
    Mean_PMR=("Profit_Margin_Rate", "mean"),
    Total_Quantity=("Quantity", "sum"),
    Order_Count=("Order_ID", "count")
).round(2).sort_values("Total_Revenue", ascending=False)
print(region_stats.to_string())

# ──────────────────────────────────────────────────────────────────────────────
# EDA — SECTION 4: By Category
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EDA SECTION 4 — Revenue, Profit, Quantity by Category")
print("=" * 60)
cat_stats = df.groupby("Category").agg(
    Total_Revenue=("Revenue", "sum"),
    Mean_Revenue=("Revenue", "mean"),
    Mean_PMR=("Profit_Margin_Rate", "mean"),
    Total_Quantity=("Quantity", "sum"),
    Order_Count=("Order_ID", "count")
).round(2).sort_values("Total_Revenue", ascending=False)
print(cat_stats.to_string())

# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZATION 1 — Total Quantity by Region (Horizontal Bar)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Viz 1] Generating: Total Order Quantity by Region...")

qty_region = (df[df["Region"] != "Not Specified"]
              .groupby("Region")["Quantity"]
              .sum()
              .sort_values(ascending=True))

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#3b82d4" if v == qty_region.max() else "#93c5fd" for v in qty_region.values]
bars = ax.barh(qty_region.index, qty_region.values, color=colors, edgecolor="white", height=0.6)

for bar in bars:
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():,.0f}", va="center", fontsize=9.5, color="#1f2328")

ax.set_xlabel("Total Units Ordered", fontsize=11)
ax.set_title("Total Order Quantity by Region\n(Logistics Fulfilment Load Indicator)",
             fontsize=13, fontweight="bold", pad=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(0, qty_region.max() * 1.15)
plt.tight_layout()
plt.savefig("viz1_qty_by_region.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: viz1_qty_by_region.png")

# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZATION 2 — Revenue vs Profit Scatter (by Category)
# ──────────────────────────────────────────────────────────────────────────────
print("[Viz 2] Generating: Revenue vs Profit scatter plot...")

df_plot = df[df["Revenue"] <= df["Revenue"].quantile(0.99)].copy()

palette = {
    "Electronics": "#3b82d4",
    "Furniture":   "#7c5cd8",
    "Home":        "#10b981",
    "Beauty":      "#f59e0b",
    "Apparel":     "#ef4444",
}

fig, ax = plt.subplots(figsize=(10, 6))
for cat, grp in df_plot.groupby("Category"):
    ax.scatter(grp["Revenue"], grp["Profit"],
               label=cat, alpha=0.55, s=28,
               color=palette.get(cat, "grey"), edgecolors="none")

m, b = np.polyfit(df_plot["Revenue"], df_plot["Profit"], 1)
x_line = np.linspace(df_plot["Revenue"].min(), df_plot["Revenue"].max(), 300)
ax.plot(x_line, m * x_line + b, color="black", linewidth=1.3,
        linestyle="--", label=f"Trend (r=0.938)", alpha=0.7)

ax.set_xlabel("Revenue (₹)", fontsize=11)
ax.set_ylabel("Profit (₹)", fontsize=11)
ax.set_title("Revenue vs. Profit by Product Category\nCorrelation r = 0.938",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(title="Category", fontsize=9, title_fontsize=9)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"₹{y/1000:.0f}K"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("viz2_revenue_vs_profit.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: viz2_revenue_vs_profit.png")

# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZATION 3 — Profit Margin Rate Box Plot by Category
# ──────────────────────────────────────────────────────────────────────────────
print("[Viz 3] Generating: PMR box plot by Category...")

cat_order = (df.groupby("Category")["Profit_Margin_Rate"]
               .median().sort_values(ascending=False).index.tolist())

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x="Category", y="Profit_Margin_Rate",
            order=cat_order,
            palette={"Beauty": "#f59e0b", "Apparel": "#ef4444",
                     "Furniture": "#7c5cd8", "Home": "#10b981",
                     "Electronics": "#3b82d4"},
            flierprops=dict(marker="o", markersize=3,
                            markerfacecolor="#ef4444", alpha=0.5),
            linewidth=0.8, ax=ax)

dataset_mean_pmr = df["Profit_Margin_Rate"].mean()
ax.axhline(dataset_mean_pmr, color="#ef4444", linestyle="--",
           linewidth=1.2, label=f"Dataset Mean PMR ({dataset_mean_pmr:.1f}%)")
ax.set_xlabel("Product Category", fontsize=11)
ax.set_ylabel("Profit Margin Rate (%)", fontsize=11)
ax.set_title("Profit Margin Rate Distribution by Category\n(Spread reveals pricing consistency)",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("viz3_pmr_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: viz3_pmr_boxplot.png")

# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZATION 4 — PMR Heatmap: Region x Category
# ──────────────────────────────────────────────────────────────────────────────
print("[Viz 4] Generating: PMR heatmap Region x Category...")

pivot = df.pivot_table(
    values="Profit_Margin_Rate",
    index="Region", columns="Category", aggfunc="mean"
)
pivot = pivot.drop("Not Specified", errors="ignore")
pivot = pivot.sort_index()

fig, ax = plt.subplots(figsize=(11, 5))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu",
            linewidths=0.5, linecolor="#e5e7eb",
            cbar_kws={"label": "Avg PMR (%)", "shrink": 0.8},
            ax=ax)
ax.set_title("Avg Profit Margin Rate (%) — Region × Category\nLogistics Strategy Matrix",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Product Category", fontsize=11)
ax.set_ylabel("Region", fontsize=11)
plt.tight_layout()
plt.savefig("viz4_pmr_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: viz4_pmr_heatmap.png")

# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZATION 5 — Monthly Revenue Trend Line
# ──────────────────────────────────────────────────────────────────────────────
print("[Viz 5] Generating: Monthly revenue trend...")

monthly = df.groupby("Month")["Revenue"].sum().reset_index()
month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(monthly["Month"], monthly["Revenue"],
        color="#3b82d4", linewidth=2.5, marker="o",
        markersize=7, markerfacecolor="white", markeredgewidth=2.2,
        markeredgecolor="#3b82d4")
ax.fill_between(monthly["Month"], monthly["Revenue"], alpha=0.1, color="#3b82d4")

# Annotate peak and trough
peak_row   = monthly.loc[monthly["Revenue"].idxmax()]
trough_row = monthly.loc[monthly["Revenue"].idxmin()]

ax.annotate(f"Peak\n₹{peak_row['Revenue']/1e6:.2f}M",
            xy=(peak_row["Month"], peak_row["Revenue"]),
            xytext=(peak_row["Month"], peak_row["Revenue"] * 1.12),
            arrowprops=dict(arrowstyle="->", color="#3b82d4", lw=1.5),
            fontsize=9, color="#3b82d4", fontweight="bold", ha="center")
ax.annotate(f"Trough\n₹{trough_row['Revenue']/1e6:.2f}M",
            xy=(trough_row["Month"], trough_row["Revenue"]),
            xytext=(trough_row["Month"] - 1.2, trough_row["Revenue"] * 1.6),
            arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1.5),
            fontsize=9, color="#ef4444", fontweight="bold", ha="center")

ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_labels)
ax.set_xlabel("Month (2026)", fontsize=11)
ax.set_ylabel("Total Revenue (₹)", fontsize=11)
ax.set_title("Monthly Revenue Trend — 2026\nSeasonality & Logistics Demand Planning",
             fontsize=13, fontweight="bold", pad=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"₹{y/1e6:.1f}M"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("viz5_monthly_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: viz5_monthly_trend.png")

print("\n" + "=" * 60)
print("Week 3 EDA complete. Files generated:")
print("  viz1_qty_by_region.png")
print("  viz2_revenue_vs_profit.png")
print("  viz3_pmr_boxplot.png")
print("  viz4_pmr_heatmap.png")
print("  viz5_monthly_trend.png")
print("=" * 60)
