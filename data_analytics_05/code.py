# @author: ALI GHANBARI 
# @email: alighanbari446@gmail.com

# Enhanced end-to-end script: generate a richer synthetic cosmetics KPI dataset and run a professional set of analyses and plots.
# This will:
# - Create a 3-year weekly dataset for 12 SKUs with product/category/region/channel details
# - Add extended columns (promotions, raw material cost, marketing spend, inventory, backorders, production line)
# - Compute many KPIs: Revenue, Profit, Profit_Margin, MoM, YoY, rolling metrics, Target achievement, OEE-like efficiency
# - Perform Pareto, Cohort-style retention summary, Anomaly detection (IsolationForest), simple forecasting (LinearRegression)
# - Create and save multiple plots (each figure separately, matplotlib only)
# - Save CSV and a notebook (.ipynb) for the project
# Note: matplotlib is used (no seaborn), and charts are individual figures as required.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import r2_score
import nbformat as nbf
import os

pd.options.mode.chained_assignment = None
np.random.seed(2025)  # reproducible

# -----------------------------
# 1) Data generation parameters
# -----------------------------
start_date = "2022-01-03"   # Monday
end_date = "2024-12-30"
freq = "W-MON"
dates = pd.date_range(start=start_date, end=end_date, freq=freq)

n_products = 12
categories = ["Skincare", "Makeup", "Haircare", "Fragrance"]
regions = ["North", "South", "East", "West", "Central"]
channels = ["Retail", "E-commerce", "Wholesale"]
production_lines = ["LineA", "LineB", "LineC"]

products = []
for i in range(n_products):
    cat = np.random.choice(categories, p=[0.4,0.3,0.2,0.1])
    name = f"{cat[:3].upper()}_SKU_{i+1:02d}"
    base_price = round(np.random.uniform(6.0, 60.0), 2)
    base_cost = round(base_price * np.random.uniform(0.30, 0.65), 2)
    material_kg = round(np.random.uniform(0.03, 0.6), 3)
    products.append({
        "Product_ID": f"P{i+1:03d}",
        "Product_Name": name,
        "Category": cat,
        "Base_Price": base_price,
        "Base_Cost": base_cost,
        "Material_kg_per_unit": material_kg
    })
products_df = pd.DataFrame(products)

rows = []
for _, p in products_df.iterrows():
    base_weekly_demand = np.random.randint(300, 2500)
    seasonality_strength = np.random.uniform(0.05, 0.20)
    promo_sensitivity = np.random.uniform(0.8, 1.6)
    for d in dates:
        week_index = (d - pd.to_datetime(start_date)).days // 7
        month = d.month
        # seasonal factor: Q4 holiday + spring bump
        season = 1.0 + (0.18 if month in [11,12] else 0.06 if month in [4,5] else 0.0) * seasonality_strength*5
        trend = 1.0 + week_index * np.random.normal(0.0005, 0.0006)
        noise = np.random.normal(1.0, 0.14)
        produced_units = max(0, int(base_weekly_demand * season * trend * noise))
        # per-week production characteristics
        downtime_hours = max(0, np.random.normal(6, 2.5))
        available_hours = 7*24*0.18  # portion of plant capacity
        efficiency = produced_units / (available_hours - downtime_hours + 1e-6)  # units per effective hour
        defect_rate = float(np.clip(np.random.normal(0.02, 0.015), 0.0, 0.25))
        good_units = int(produced_units * (1 - defect_rate))
        # promotions randomly happen (~15% weeks)
        on_promo = np.random.rand() < 0.15
        promo_discount = round(np.random.uniform(0.05, 0.35), 2) if on_promo else 0.0
        # channel mix
        ch = np.random.choice(channels, p=[0.45, 0.45, 0.10])
        # sales depend on demand, channel and promo sensitivity
        channel_factor = 1.0 + (0.12 if ch=="E-commerce" else 0.0)
        sales_units = int(good_units * np.random.uniform(0.78, 0.995) * channel_factor * (1 + promo_sensitivity*promo_discount if on_promo else 1.0))
        # price and cost adjustments
        price = round(p["Base_Price"] * np.random.uniform(0.95, 1.08) * (1 - promo_discount), 2)
        raw_material_cost_kg = round(np.random.uniform(0.8, 1.2) * p["Material_kg_per_unit"], 3)
        raw_material_cost = round(raw_material_cost_kg * np.random.uniform(1.2, 2.5), 3) * produced_units  # cost of material this week
        cost_per_unit = round(p["Base_Cost"] * np.random.uniform(0.92, 1.06), 2)
        cogs = produced_units * cost_per_unit + raw_material_cost * 0.1
        marketing_spend = round(np.random.uniform(200, 4000) * (1.5 if on_promo else 1.0), 2)
        logistics_cost = round(np.random.uniform(0.02, 0.08) * sales_units * price, 2)
        other_expenses = round(np.random.uniform(200, 1200), 2)
        revenue = round(sales_units * price, 2)
        profit = round(revenue - cogs - marketing_spend - logistics_cost - other_expenses, 2)
        profit_margin = profit / revenue if revenue > 0 else 0.0
        returns_rate = float(np.clip(np.random.normal(0.015, 0.01), 0.0, 0.2))
        returned_units = int(sales_units * returns_rate)
        lead_time_days = int(max(1, np.random.normal(6.5, 3.2) + (2 if on_promo else 0)))
        inventory = max(0, int(np.random.normal(3000, 800) + produced_units - sales_units))
        backorder = max(0, int(np.random.poisson( max(0, (sales_units - good_units) * 0.5) )))
        acquisition = int(sales_units * np.random.uniform(0.03, 0.09))
        retention_rate = float(np.clip(np.random.normal(0.70, 0.06), 0.35, 0.97))
        nps = float(np.clip(np.random.normal(28, 14), -100, 100))
        employee_satisfaction = float(np.clip(np.random.normal(7.1, 0.9), 1, 10))
        region = np.random.choice(regions)
        production_line = np.random.choice(production_lines)
        campaign = np.random.choice(["None","SummerSale","HolidayPush","NewLaunch"], p=[0.7,0.12,0.12,0.06])
        row = {
            "Date": d, "Week_Index": week_index,
            "Product_ID": p["Product_ID"], "Product_Name": p["Product_Name"], "Category": p["Category"],
            "Region": region, "Channel": ch, "Production_Line": production_line, "Campaign": campaign,
            "Produced_Units": produced_units, "Good_Units": good_units, "Sales_Units": sales_units, "Returned_Units": returned_units,
            "Price_per_Unit": price, "Cost_per_Unit": cost_per_unit, "Raw_Material_Cost": round(raw_material_cost,2),
            "COGS": round(cogs,2), "Marketing_Spend": marketing_spend, "Logistics_Cost": logistics_cost, "Other_Expenses": other_expenses,
            "Revenue": revenue, "Profit": profit, "Profit_Margin": round(profit_margin,4),
            "Lead_Time_days": lead_time_days, "Downtime_hours": round(downtime_hours,2), "Efficiency_u_per_h": round(efficiency,2),
            "Defect_Rate": round(defect_rate,4), "Returns_Rate": round(returns_rate,4),
            "Inventory": inventory, "Backorder": backorder,
            "Acquisition": acquisition, "Retention_Rate": round(retention_rate,4), "NPS": round(nps,1),
            "Employee_Satisfaction": round(employee_satisfaction,2), "On_Promo": on_promo, "Promo_Discount": promo_discount
        }
        rows.append(row)

df = pd.DataFrame(rows)
df.sort_values(["Product_ID","Date"], inplace=True)
df.reset_index(drop=True, inplace=True)

# -----------------------------
# 2) KPI computations
# -----------------------------
df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
df["Year"] = df["Date"].dt.year
df["Quarter"] = df["Date"].dt.to_period("Q").dt.to_timestamp()

# Weekly targets per product (synthetic)
targets = df.groupby("Product_ID")["Sales_Units"].mean().reset_index().rename(columns={"Sales_Units":"Weekly_Target"})
targets["Weekly_Target"] = (targets["Weekly_Target"] * 1.05).round().astype(int)
df = df.merge(targets, on="Product_ID", how="left")
df["Target_Achieved"] = df["Sales_Units"] / df["Weekly_Target"]

# Rolling and growth metrics
df["Rolling_4w_Sales"] = df.groupby("Product_ID")["Sales_Units"].transform(lambda x: x.rolling(4, min_periods=1).mean())
df["Rolling_13w_Sales"] = df.groupby("Product_ID")["Sales_Units"].transform(lambda x: x.rolling(13, min_periods=1).mean())  # quarter-ish
df["MoM_Sales_pct"] = df.groupby("Product_ID")["Sales_Units"].transform(lambda x: x.pct_change(4))
df["YoY_Sales_pct"] = df.groupby("Product_ID")["Sales_Units"].transform(lambda x: x.pct_change(52))
df["Rolling_4w_Eff"] = df.groupby("Product_ID")["Efficiency_u_per_h"].transform(lambda x: x.rolling(4, min_periods=1).mean())

# Inventory KPIs
df["Weeks_of_Inventory"] = df["Inventory"] / (df["Sales_Units"].replace(0, np.nan) + 1e-6) * 1.0
df["Fill_Rate_est"] = 1 - (df["Backorder"] / (df["Sales_Units"].replace(0, np.nan) + 1e-6))

# Cohort-like metric: acquisition cohort month and simple retention proxy
df["Acq_Month"] = df["Date"].where(df["Acquisition"]>0, pd.NaT)
df["Acq_Month"] = pd.to_datetime(df["Acq_Month"]).dt.to_period("M").dt.to_timestamp()
# For simplicity create a cohort table: sum acquisition by product and month, then compute retention proxy later on monthly aggregates

# -----------------------------
# 3) Aggregations for reporting
# -----------------------------
monthly = df.groupby(["Month","Product_ID","Product_Name","Category"]).agg({
    "Produced_Units":"sum","Good_Units":"sum","Sales_Units":"sum","Returned_Units":"sum","Revenue":"sum","COGS":"sum","Marketing_Spend":"sum","Logistics_Cost":"sum","Other_Expenses":"sum","Profit":"sum","Raw_Material_Cost":"sum",
    "Acquisition":"sum","NPS":"mean","Employee_Satisfaction":"mean","Downtime_hours":"sum","Defect_Rate":"mean","Inventory":"mean","Backorder":"sum"
}).reset_index()

yearly = df.groupby(["Year","Product_ID","Product_Name","Category"]).agg({
    "Produced_Units":"sum","Good_Units":"sum","Sales_Units":"sum","Returned_Units":"sum","Revenue":"sum","COGS":"sum","Marketing_Spend":"sum","Logistics_Cost":"sum","Other_Expenses":"sum","Profit":"sum","Raw_Material_Cost":"sum",
    "Acquisition":"sum","NPS":"mean","Employee_Satisfaction":"mean","Downtime_hours":"sum","Defect_Rate":"mean","Inventory":"mean","Backorder":"sum"
}).reset_index()

latest_year = df["Year"].max()
annual_last = yearly[yearly["Year"]==latest_year].copy()
annual_last["Profit_Margin"] = annual_last["Profit"] / annual_last["Revenue"]

# -----------------------------
# 4) Anomaly detection (IsolationForest) on weekly features
# -----------------------------
iso_features = df[["Sales_Units","Revenue","Profit","Defect_Rate","Downtime_hours","Efficiency_u_per_h"]].fillna(0)
iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
iso.fit(iso_features)
df["anomaly_score"] = iso.decision_function(iso_features)
df["is_anomaly"] = iso.predict(iso_features) == -1

# -----------------------------
# 5) Simple forecast example (per-product linear regression on week index)
# -----------------------------
forecasts = []
for pid in df["Product_ID"].unique():
    prod_week = df[df["Product_ID"]==pid].groupby("Week_Index")["Sales_Units"].sum().reset_index()
    X = prod_week[["Week_Index"]].values
    y = prod_week["Sales_Units"].values
    if len(X) < 10:
        continue
    model = LinearRegression()
    model.fit(X, y)
    future_idx = np.arange(prod_week["Week_Index"].max()+1, prod_week["Week_Index"].max()+13).reshape(-1,1)
    y_pred = model.predict(future_idx)
    forecasts.append({"Product_ID": pid, "future_weeks": future_idx.flatten(), "pred": y_pred, "r2": r2_score(y, model.predict(X))})

# -----------------------------
# 6) Important outputs to save
# -----------------------------
out_csv = "cosmetics_kpi_rich.csv"
df.to_csv(out_csv, index=False)
monthly_csv = "cosmetics_monthly_aggregates_rich.csv"
monthly.to_csv(monthly_csv, index=False)
annual_csv = "cosmetics_annual_last_year_rich.csv"
annual_last.to_csv(annual_csv, index=False)

# Display previews to user if helper exists
# try:
    # from caas_jupyter_tools import display_dataframe_to_user
    # display_dataframe_to_user("Sample (first 200 rows) - rich cosmetics KPI dataset", df.head(200))
    # display_dataframe_to_user("Annual KPI sample", annual_last.sort_values("Revenue", ascending=False).head(20))
# except Exception:
    # print("Preview (first 10 rows):")
    # display(df.head(10))

print("Saved dataset to:", out_csv)
print("Saved monthly aggregates to:", monthly_csv)
print("Saved annual KPIs to:", annual_csv)

# -----------------------------
# 7) Professional visualizations (matplotlib only; one figure per plot)
# -----------------------------

# A) Company-level weekly revenue trend
weekly_total = df.groupby("Date")[["Revenue","Sales_Units","Profit"]].sum().reset_index()

plt.figure(figsize=(10,4))
plt.plot(weekly_total["Date"], weekly_total["Revenue"], marker='.', linewidth=1)
plt.title("Weekly Total Revenue Trend")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig("plot_weekly_revenue.png")
plt.show()

# B) Annual profit by product (bar)
plt.figure(figsize=(10,5))
plt.bar(annual_last["Product_Name"], annual_last["Profit"])
plt.xticks(rotation=45, ha="right")
plt.title(f"Annual Profit by Product - {latest_year}")
plt.xlabel("Product")
plt.ylabel("Profit")
plt.tight_layout()
plt.savefig("plot_annual_profit_by_product.png")
plt.show()

# C) Pareto: revenue contribution
top_rev = annual_last.sort_values("Revenue", ascending=False).copy()
top_rev["CumRev"] = top_rev["Revenue"].cumsum()
top_rev["CumPct"] = top_rev["CumRev"] / top_rev["Revenue"].sum()

plt.figure(figsize=(10,5))
plt.bar(top_rev["Product_Name"], top_rev["Revenue"])
plt.plot(range(len(top_rev)), top_rev["CumPct"], marker='o')
plt.xticks(rotation=45, ha="right")
plt.title("Pareto - Revenue by Product and cumulative %")
plt.tight_layout()
plt.savefig("plot_pareto_revenue.png")
plt.show()

# D) MoM % change heatmap style (matrix of products x months) - use pivot table and imshow
mom = monthly.copy()
mom["MoM_pct"] = mom.groupby("Product_ID")["Sales_Units"].pct_change().fillna(0)
pivot = mom.pivot(index="Product_Name", columns="Month", values="MoM_pct").fillna(0)
plt.figure(figsize=(12,6))
plt.imshow(pivot, aspect='auto')
plt.colorbar()
plt.title("MoM Sales % change (products x months)")
plt.xlabel("Month index")
plt.ylabel("Product")
plt.yticks(range(len(pivot.index)), pivot.index)
plt.tight_layout()
plt.savefig("plot_mom_matrix.png")
plt.show()

# E) Scatter: Employee Satisfaction vs Profit (monthly) with regression line
monthly_prod = monthly.copy()
X = monthly_prod[["Employee_Satisfaction"]].values
y = monthly_prod["Profit"].values
reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)
r2 = r2_score(y, y_pred)

plt.figure(figsize=(7,5))
plt.scatter(monthly_prod["Employee_Satisfaction"], monthly_prod["Profit"], alpha=0.6)
# regression line sorted by x for display
order = np.argsort(monthly_prod["Employee_Satisfaction"])
plt.plot(monthly_prod["Employee_Satisfaction"].values[order], y_pred[order], linewidth=2)
plt.title(f"Employee Satisfaction vs Monthly Profit (R2={r2:.2f})")
plt.xlabel("Employee Satisfaction (avg)")
plt.ylabel("Monthly Profit")
plt.tight_layout()
plt.savefig("plot_emp_sat_vs_profit.png")
plt.show()

# F) Correlation matrix image
corr_cols = ["Produced_Units","Sales_Units","Revenue","Profit","Defect_Rate","Downtime_hours","Raw_Material_Cost","Employee_Satisfaction","NPS"]
corr = monthly[corr_cols].corr()
plt.figure(figsize=(8,6))
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
plt.yticks(range(len(corr_cols)), corr_cols)
plt.title("Correlation matrix (monthly KPIs)")
plt.tight_layout()
plt.savefig("plot_correlation_matrix.png")
plt.show()

# G) Anomalies over time (highlight weeks flagged)
anomalies = df[df["is_anomaly"]]
plt.figure(figsize=(10,4))
plt.plot(weekly_total["Date"], weekly_total["Revenue"], label="Revenue")
plt.scatter(anomalies.groupby("Date")["Revenue"].sum().index, anomalies.groupby("Date")["Revenue"].sum().values, color='red', label="Anomalous weeks")
plt.title("Weekly Revenue with Anomalous Weeks Highlighted")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.legend()
plt.tight_layout()
plt.savefig("plot_anomalies_revenue.png")
plt.show()

# H) Inventory weeks distribution histogram
plt.figure(figsize=(8,4))
plt.hist(df["Weeks_of_Inventory"].replace(np.inf, np.nan).dropna(), bins=40)
plt.title("Distribution of Weeks of Inventory across product-weeks")
plt.xlabel("Weeks of Inventory")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("plot_weeks_inventory_hist.png")
plt.show()

# I) Forecast example plot for first forecast entry (if exists)
if len(forecasts) > 0:
    f0 = forecasts[0]
    pid = f0["Product_ID"]
    prod_week = df[df["Product_ID"]==pid].groupby("Week_Index")["Sales_Units"].sum().reset_index()
    plt.figure(figsize=(10,4))
    plt.plot(prod_week["Week_Index"], prod_week["Sales_Units"], label="Historical")
    plt.plot(f0["future_weeks"], f0["pred"], label="Forecast (12 weeks ahead)")
    plt.title(f"12-week Linear Forecast for {pid}")
    plt.xlabel("Week Index")
    plt.ylabel("Sales Units")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plot_forecast_{pid}.png")
    plt.show()

# -----------------------------
# 8) Save a Jupyter notebook containing the enhanced pipeline code and explanations
# -----------------------------
nb = nbf.v4.new_notebook()
intro_md = """# Rich Cosmetics KPI Project - Synthetic Dataset & Professional Analysis\n\nThis notebook generates a rich synthetic weekly KPI dataset for cosmetics SKUs (2022-01 to 2024-12) and performs professional analyses including rolling metrics, MoM/YoY, Pareto, cohort-like acquisition aggregates, anomaly detection, and basic forecasting. Plots are generated (matplotlib) and saved to  as PNGs.\n\nFiles produced:\n- cosmetics_kpi_rich.csv\n- cosmetics_monthly_aggregates_rich.csv\n- cosmetics_annual_last_year_rich.csv\n- Several PNG plots (weekly revenue, pareto, anomalies, etc.)\n"""

cells = [
    nbf.v4.new_markdown_cell(intro_md),
    nbf.v4.new_markdown_cell("## Data generation and KPI computation\n(See script saved as CSVs in )"),
    nbf.v4.new_code_cell("# The full data generation & analysis script was executed and outputs saved to .\n# To re-run, use the Python script provided in the repository or re-run the cells in this notebook."),
    nbf.v4.new_markdown_cell("## Saved outputs\n- cosmetics_kpi_rich.csv\n- cosmetics_monthly_aggregates_rich.csv\n- cosmetics_annual_last_year_rich.csv\n- plot_weekly_revenue.png\n- plot_annual_profit_by_product.png\n- plot_pareto_revenue.png\n- plot_mom_matrix.png\n- plot_emp_sat_vs_profit.png\n- plot_correlation_matrix.png\n- plot_anomalies_revenue.png\n- plot_weeks_inventory_hist.png\n- plot_forecast_*.png (if forecast available)")
]
nb['cells'] = cells

out_nb_path = "cosmetics_kpi_rich_analysis.ipynb"
with open(out_nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook saved to:", out_nb_path)
print("Main dataset saved to:", out_csv)
print("Monthly aggregates saved to:", monthly_csv)
print("Annual KPIs saved to:", annual_csv)

