# @author: ALI GHANBARI 
# @email: alighanbari446@gmail.com

# cosmetics_kpi_time_analysis_en.py
"""
Professional time-series KPI analysis for a synthetic cosmetics company.
Features:
- Generates or loads a rich weekly dataset (3 years, ~12 SKUs) if not present
- Operational KPIs included: Lead Time, Efficiency, Downtime, Defect Rate, Inventory, Backorder
- Time-based metrics: Week-over-Week (WoW), Month-over-Month (MoM), Quarter-over-Quarter (QoQ), Year-over-Year (YoY)
- Rolling metrics, targets, weeks-of-inventory, fill-rate estimate
- Summary tables for top products, alert flags for large drops/increases
- Professional plots (matplotlib only) saved as PNGs
- All code, text, filenames and plot labels in English
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Settings
OUT_DIR = "cosmetics_kpi_outputs_en"
DATA_CSV = "cosmetics_kpi_rich_en.csv"
TOP_N_PRODUCTS = 6
PCT_CHANGE_ALERT = 0.10  # 10% threshold for alerts

os.makedirs(OUT_DIR, exist_ok=True)
pd.options.mode.chained_assignment = None
np.random.seed(12345)

# ----------------------------
# 1) Generate sample dataset if not present
# ----------------------------
def generate_sample_dataset(path=DATA_CSV):
    start_date = "2022-01-03"
    end_date = "2024-12-30"
    dates = pd.date_range(start=start_date, end=end_date, freq="W-MON")
    categories = ["Skincare", "Makeup", "Haircare", "Fragrance"]
    channels = ["Retail", "E-commerce", "Wholesale"]
    regions = ["North", "South", "East", "West", "Central"]
    production_lines = ["LineA", "LineB", "LineC"]

    n_products = 12
    products = []
    for i in range(n_products):
        cat = np.random.choice(categories, p=[0.4,0.3,0.2,0.1])
        products.append({
            "Product_ID": f"P{i+1:03d}",
            "Product_Name": f"{cat[:3].upper()}_SKU_{i+1:02d}",
            "Category": cat,
            "Base_Price": round(np.random.uniform(6,60),2),
            "Base_Cost": round(np.random.uniform(2,40),2),
            "Material_kg_per_unit": round(np.random.uniform(0.03,0.6),3)
        })

    rows = []
    for p in products:
        base_weekly = np.random.randint(300, 2500)
        seasonality_strength = np.random.uniform(0.05, 0.20)
        promo_sensitivity = np.random.uniform(0.8, 1.6)
        for d in dates:
            week_index = (d - pd.to_datetime(start_date)).days // 7
            month = d.month
            season = 1.0 + (0.18 if month in [11,12] else 0.06 if month in [4,5] else 0.0) * seasonality_strength * 5
            trend = 1.0 + week_index * np.random.normal(0.0005, 0.0006)
            noise = np.random.normal(1.0, 0.14)
            produced_units = max(0, int(base_weekly * season * trend * noise))
            downtime_hours = max(0, np.random.normal(6, 2.5))
            available_hours = 7*24*0.18
            efficiency = produced_units / (available_hours - downtime_hours + 1e-6)
            defect_rate = float(np.clip(np.random.normal(0.02, 0.015), 0.0, 0.25))
            good_units = int(produced_units * (1 - defect_rate))
            on_promo = np.random.rand() < 0.15
            promo_discount = round(np.random.uniform(0.05, 0.35), 2) if on_promo else 0.0
            channel = np.random.choice(channels, p=[0.45,0.45,0.10])
            channel_factor = 1.0 + (0.12 if channel == "E-commerce" else 0.0)
            sales_units = int(good_units * np.random.uniform(0.78, 0.995) * channel_factor * (1 + promo_sensitivity*promo_discount if on_promo else 1.0))
            price = round(p["Base_Price"] * np.random.uniform(0.95, 1.08) * (1 - promo_discount), 2)
            raw_material_cost_kg = round(np.random.uniform(0.8,1.2) * p["Material_kg_per_unit"], 3)
            raw_material_cost = round(raw_material_cost_kg * np.random.uniform(1.2, 2.5), 3) * produced_units
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
            backorder = max(0, int(np.random.poisson(max(0, (sales_units - good_units) * 0.5))))
            acquisition = int(sales_units * np.random.uniform(0.03, 0.09))
            retention_rate = float(np.clip(np.random.normal(0.70, 0.06), 0.35, 0.97))
            nps = float(np.clip(np.random.normal(28, 14), -100, 100))
            employee_satisfaction = float(np.clip(np.random.normal(7.1, 0.9), 1, 10))
            region = np.random.choice(regions)
            production_line = np.random.choice(production_lines)
            campaign = np.random.choice(["None","SummerSale","HolidayPush","NewLaunch"], p=[0.7,0.12,0.12,0.06])

            rows.append({
                "Date": d, "Week_Index": week_index,
                "Product_ID": p["Product_ID"], "Product_Name": p["Product_Name"], "Category": p["Category"],
                "Region": region, "Channel": channel, "Production_Line": production_line, "Campaign": campaign,
                "Produced_Units": produced_units, "Good_Units": good_units, "Sales_Units": sales_units, "Returned_Units": returned_units,
                "Price_per_Unit": price, "Cost_per_Unit": cost_per_unit, "Raw_Material_Cost": round(raw_material_cost,2),
                "COGS": round(cogs,2), "Marketing_Spend": marketing_spend, "Logistics_Cost": logistics_cost, "Other_Expenses": other_expenses,
                "Revenue": revenue, "Profit": profit, "Profit_Margin": round(profit_margin,4),
                "Lead_Time_days": lead_time_days, "Downtime_hours": round(downtime_hours,2), "Efficiency_u_per_h": round(efficiency,2),
                "Defect_Rate": round(defect_rate,4), "Returns_Rate": round(returns_rate,4),
                "Inventory": inventory, "Backorder": backorder,
                "Acquisition": acquisition, "Retention_Rate": round(retention_rate,4), "NPS": round(nps,1),
                "Employee_Satisfaction": round(employee_satisfaction,2), "On_Promo": on_promo, "Promo_Discount": promo_discount
            })

    df = pd.DataFrame(rows)
    df.sort_values(["Product_ID","Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Sample dataset saved to {path}")
    return df

if os.path.exists(DATA_CSV):
    df = pd.read_csv(DATA_CSV, parse_dates=["Date"])
    print(f"[INFO] Loaded dataset from {DATA_CSV} (rows: {len(df)})")
else:
    df = generate_sample_dataset(DATA_CSV)

# ----------------------------
# 2) Time formatting and basic KPIs
# ----------------------------
df['Month'] = df['Date'].dt.to_period('M').dt.to_timestamp()
df['Quarter'] = df['Date'].dt.to_period('Q').dt.to_timestamp()
df['Year'] = df['Date'].dt.year
df.sort_values(['Product_ID','Date'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Weekly target (synthetic) and rolling/growth metrics
targets = df.groupby("Product_ID")["Sales_Units"].mean().reset_index().rename(columns={"Sales_Units":"Weekly_Target"})
targets["Weekly_Target"] = (targets["Weekly_Target"] * 1.05).round().astype(int)
df = df.merge(targets, on="Product_ID", how="left")
df["Target_Achieved"] = df["Sales_Units"] / (df["Weekly_Target"] + 1e-9)
df["Rolling_4w_Sales"] = df.groupby("Product_ID")["Sales_Units"].transform(lambda x: x.rolling(4, min_periods=1).mean())
df["Rolling_13w_Sales"] = df.groupby("Product_ID")["Sales_Units"].transform(lambda x: x.rolling(13, min_periods=1).mean())

# ----------------------------
# 3) WoW, MoM, QoQ, YoY computations
# ----------------------------
# Week-over-Week (WoW) on weekly-level columns
for col in ["Sales_Units", "Revenue", "Lead_Time_days", "Efficiency_u_per_h"]:
    df[f"WoW_pct_{col}"] = df.groupby("Product_ID")[col].pct_change(1)

# Monthly aggregation and MoM/YoY
monthly = df.groupby(["Month","Product_ID","Product_Name","Category"])[
    ["Produced_Units","Good_Units","Sales_Units","Revenue","COGS","Profit","Lead_Time_days","Efficiency_u_per_h","Inventory"]
].agg({"Produced_Units":"sum","Good_Units":"sum","Sales_Units":"sum","Revenue":"sum","COGS":"sum","Profit":"sum","Lead_Time_days":"mean","Efficiency_u_per_h":"mean","Inventory":"mean"}).reset_index().sort_values(["Product_ID","Month"])

monthly["MoM_Sales_pct"] = monthly.groupby("Product_ID")["Sales_Units"].pct_change(1)
monthly["YoY_Sales_pct"] = monthly.groupby("Product_ID")["Sales_Units"].pct_change(12)
monthly["MoM_LeadTime_pct"] = monthly.groupby("Product_ID")["Lead_Time_days"].pct_change(1)
monthly["MoM_Eff_pct"] = monthly.groupby("Product_ID")["Efficiency_u_per_h"].pct_change(1)

# Quarterly aggregation and QoQ
quarterly = df.groupby(["Quarter","Product_ID","Product_Name","Category"])[
    ["Produced_Units","Sales_Units","Revenue","Lead_Time_days","Efficiency_u_per_h"]
].agg({"Produced_Units":"sum","Sales_Units":"sum","Revenue":"sum","Lead_Time_days":"mean","Efficiency_u_per_h":"mean"}).reset_index().sort_values(["Product_ID","Quarter"])
quarterly["QoQ_Sales_pct"] = quarterly.groupby("Product_ID")["Sales_Units"].pct_change(1)
quarterly["YoY_Q_Sales_pct"] = quarterly.groupby("Product_ID")["Sales_Units"].pct_change(4)

# Yearly aggregation and YoY
yearly = df.groupby(["Year","Product_ID","Product_Name","Category"])[
    ["Produced_Units","Sales_Units","Revenue","Lead_Time_days","Efficiency_u_per_h"]
].agg({"Produced_Units":"sum","Sales_Units":"sum","Revenue":"sum","Lead_Time_days":"mean","Efficiency_u_per_h":"mean"}).reset_index().sort_values(["Product_ID","Year"])
yearly["YoY_Sales_pct"] = yearly.groupby("Product_ID")["Sales_Units"].pct_change(1)

# ----------------------------
# 4) Summaries for reporting
# ----------------------------
latest_year = yearly["Year"].max()
annual_last = yearly[yearly["Year"] == latest_year].copy()
annual_last["Profit_Margin"] = annual_last.apply(lambda r: r["Revenue"] and (r["Revenue"] != 0) and (r["Profit"] / r["Revenue"]) or 0, axis=1)

top_products = annual_last.sort_values("Revenue", ascending=False).head(TOP_N_PRODUCTS)
top_ids = top_products["Product_ID"].tolist()

summary_tables = {}
for pid in top_ids:
    t = monthly[monthly["Product_ID"] == pid].sort_values("Month").copy()
    t["Rolling_3m_Sales"] = t["Sales_Units"].rolling(3, min_periods=1).mean()
    t["MoM_Drop_Flag"] = t["MoM_Sales_pct"].apply(lambda x: True if (pd.notna(x) and x < -PCT_CHANGE_ALERT) else False)
    summary_tables[pid] = t

company_weekly = df.groupby("Date")[["Sales_Units","Revenue","Profit"]].sum().reset_index().sort_values("Date")
company_weekly["WoW_Sales_pct"] = company_weekly["Sales_Units"].pct_change(1)
company_weekly["WoW_Revenue_pct"] = company_weekly["Revenue"].pct_change(1)

# ----------------------------
# 5) Anomaly detection (IsolationForest)
# ----------------------------
from sklearn.ensemble import IsolationForest
iso_features = df[["Sales_Units","Revenue","Profit","Defect_Rate","Downtime_hours","Efficiency_u_per_h"]].fillna(0)
iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
iso.fit(iso_features)
df["anomaly_score"] = iso.decision_function(iso_features)
df["is_anomaly"] = iso.predict(iso_features) == -1

# ----------------------------
# 6) Forecast example (linear regression per product)
# ----------------------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
forecasts = []
for pid in df["Product_ID"].unique():
    prod_week = df[df["Product_ID"] == pid].groupby("Week_Index")["Sales_Units"].sum().reset_index()
    X = prod_week[["Week_Index"]].values
    y = prod_week["Sales_Units"].values
    if len(X) < 10:
        continue
    model = LinearRegression()
    model.fit(X, y)
    future_idx = np.arange(prod_week["Week_Index"].max() + 1, prod_week["Week_Index"].max() + 13).reshape(-1,1)
    y_pred = model.predict(future_idx)
    forecasts.append({"Product_ID": pid, "future_weeks": future_idx.flatten(), "pred": y_pred, "r2": r2_score(y, model.predict(X))})

# ----------------------------
# 7) Save aggregated outputs (CSV)
# ----------------------------
os.makedirs(OUT_DIR, exist_ok=True)
df.to_csv(os.path.join(OUT_DIR, "cosmetics_kpi_rich_en.csv"), index=False)
monthly.to_csv(os.path.join(OUT_DIR, "cosmetics_monthly_aggregates_en.csv"), index=False)
annual_last.to_csv(os.path.join(OUT_DIR, "cosmetics_annual_last_year_en.csv"), index=False)
print(f"[INFO] Saved CSVs to {OUT_DIR}")

# ----------------------------
# 8) Professional visualizations (matplotlib only)
# ----------------------------
def save_fig(fig, filename):
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {path}")
    return path

# A) Company weekly sales & revenue with WoW bar
fig, axs = plt.subplots(2,1, figsize=(14,8), gridspec_kw={"height_ratios":[3,1]}, sharex=True)
axs[0].plot(company_weekly["Date"], company_weekly["Sales_Units"], marker='.', label="Weekly Sales Units")
axs[0].plot(company_weekly["Date"], company_weekly["Revenue"], marker='.', label="Weekly Revenue")
axs[0].set_title("Company-level Weekly Sales and Revenue")
axs[0].legend()
axs[1].bar(company_weekly["Date"], company_weekly["WoW_Sales_pct"]*100)
axs[1].axhline(0, color='black', linewidth=0.7)
axs[1].set_ylabel("WoW % (sales)")
axs[1].set_xlabel("Date")
fig.tight_layout()
save_fig(fig, "company_weekly_trends_and_wow_en.png")

# B) Top product monthly sales + MoM bars
for pid in top_ids:
    t = summary_tables[pid]
    fig, axs = plt.subplots(2,1, figsize=(14,7), gridspec_kw={"height_ratios":[3,1]}, sharex=True)
    axs[0].plot(t["Month"], t["Sales_Units"], marker='o', label="Monthly Sales Units")
    axs[0].plot(t["Month"], t["Rolling_3m_Sales"], linestyle='--', label="3-month MA")
    axs[0].set_title(f"Monthly Sales Trend - Product {pid} ({t['Product_Name'].iloc[0]})")
    axs[0].legend()
    axs[1].bar(t["Month"], t["MoM_Sales_pct"]*100)
    axs[1].axhline(0, color='black', linewidth=0.7)
    axs[1].set_ylabel("MoM %")
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    save_fig(fig, f"product_{pid}_monthly_sales_and_mom_en.png")

# C) Lead time monthly trend for top products
for pid in top_ids:
    t = monthly[monthly["Product_ID"] == pid].sort_values("Month")
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(t["Month"], t["Lead_Time_days"], marker='o', label="Avg Lead Time (days)")
    ax.set_title(f"Monthly Lead Time Trend - Product {pid}")
    ax.set_ylabel("Days")
    ax.set_xlabel("Month")
    fig.autofmt_xdate(rotation=45)
    save_fig(fig, f"product_{pid}_leadtime_trend_en.png")

# D) Efficiency quarterly trend + QoQ sales% for top products
for pid in top_ids:
    tq = quarterly[quarterly["Product_ID"] == pid].sort_values("Quarter")
    fig, axs = plt.subplots(2,1, figsize=(12,6), gridspec_kw={"height_ratios":[3,1]}, sharex=True)
    axs[0].plot(tq["Quarter"], tq["Efficiency_u_per_h"], marker='o', label="Avg Efficiency (units/hour)")
    axs[0].set_title(f"Quarterly Efficiency - Product {pid}")
    axs[1].bar(tq["Quarter"], tq["QoQ_Sales_pct"]*100)
    axs[1].axhline(0, color='black', linewidth=0.7)
    axs[1].set_ylabel("QoQ % (sales)")
    fig.autofmt_xdate(rotation=45)
    save_fig(fig, f"product_{pid}_efficiency_and_qoq_en.png")

# E) Scatter Efficiency vs Lead Time (one example product)
if top_ids:
    pid0 = top_ids[0]
    t0 = monthly[monthly["Product_ID"] == pid0]
    if len(t0) > 5:
        x = t0["Efficiency_u_per_h"].values
        y = t0["Lead_Time_days"].values
        coef = np.polyfit(x, y, 1)
        p = np.poly1d(coef)
        fig, ax = plt.subplots(figsize=(8,5))
        ax.scatter(x, y, label="Monthly points")
        ax.plot(x, p(x), color='red', label=f"Linear fit: y={coef[0]:.3f}x+{coef[1]:.3f}")
        ax.set_title(f"Efficiency vs Lead Time - Product {pid0}")
        ax.set_xlabel("Efficiency (units/hour)")
        ax.set_ylabel("Lead Time (days)")
        ax.legend()
        save_fig(fig, f"product_{pid0}_eff_vs_leadtime_en.png")

# F) MoM heatmap (products x months)
pivot_mom = monthly.pivot(index="Product_Name", columns="Month", values="MoM_Sales_pct").fillna(0)
fig, ax = plt.subplots(figsize=(14,7))
im = ax.imshow(pivot_mom, aspect='auto', cmap='bwr', vmin=-1, vmax=1)
plt.colorbar(im, label="MoM % (fraction)")
ax.set_yticks(range(len(pivot_mom.index))); ax.set_yticklabels(pivot_mom.index)
ax.set_xticks(range(len(pivot_mom.columns))); ax.set_xticklabels([m.strftime("%Y-%m") for m in pivot_mom.columns], rotation=90)
ax.set_title("Heatmap: MoM % change - Products x Months")
fig.tight_layout()
save_fig(fig, "mom_heatmap_products_months_en.png")

# G) Correlation matrix (monthly KPIs)
corr_cols = ["Produced_Units","Sales_Units","Revenue","Profit","Defect_Rate","Downtime_hours","Raw_Material_Cost","Employee_Satisfaction","NPS"]
corr = monthly[corr_cols].corr()
fig, ax = plt.subplots(figsize=(10,8))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im)
ax.set_xticks(range(len(corr_cols))); ax.set_xticklabels(corr_cols, rotation=45, ha='right')
ax.set_yticks(range(len(corr_cols))); ax.set_yticklabels(corr_cols)
ax.set_title("Correlation matrix (monthly KPIs)")
fig.tight_layout()
save_fig(fig, "plot_correlation_matrix_en.png")

# H) Anomalies over time highlighted on revenue
anomalies = df[df["is_anomaly"]]
fig, ax = plt.subplots(figsize=(14,5))
ax.plot(company_weekly["Date"], company_weekly["Revenue"], label="Revenue")
anomaly_week_dates = anomalies.groupby("Date")["Revenue"].sum().index
anomaly_week_revs = anomalies.groupby("Date")["Revenue"].sum().values
ax.scatter(anomaly_week_dates, anomaly_week_revs, color='red', label="Anomalous weeks")
ax.set_title("Weekly Revenue with Anomalous Weeks Highlighted")
ax.set_xlabel("Date"); ax.set_ylabel("Revenue")
ax.legend()
fig.tight_layout()
save_fig(fig, "plot_anomalies_revenue_en.png")

# I) Inventory weeks distribution histogram
fig, ax = plt.subplots(figsize=(10,5))
ax.hist(df["Weeks_of_Inventory"].replace(np.inf, np.nan).dropna(), bins=40)
ax.set_title("Distribution of Weeks of Inventory")
ax.set_xlabel("Weeks of Inventory")
ax.set_ylabel("Frequency")
fig.tight_layout()
save_fig(fig, "plot_weeks_inventory_hist_en.png")

# J) Forecast example plot if available
if forecasts:
    f0 = forecasts[0]
    pid = f0["Product_ID"]
    prod_week = df[df["Product_ID"] == pid].groupby("Week_Index")["Sales_Units"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(prod_week["Week_Index"], prod_week["Sales_Units"], label="Historical")
    ax.plot(f0["future_weeks"], f0["pred"], label="12-week Linear Forecast")
    ax.set_title(f"12-week Linear Forecast - Product {pid}")
    ax.set_xlabel("Week Index"); ax.set_ylabel("Sales Units")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, f"plot_forecast_{pid}_en.png")

# ----------------------------
# 9) Save summary tables and alerts
# ----------------------------
for pid, t in summary_tables.items():
    t.to_csv(os.path.join(OUT_DIR, f"summary_monthly_{pid}_en.csv"), index=False)

company_weekly.to_csv(os.path.join(OUT_DIR, "company_weekly_wow_en.csv"), index=False)
monthly.groupby("Month")[["Sales_Units","Revenue","Produced_Units"]].sum().reset_index().to_csv(os.path.join(OUT_DIR, "monthly_aggregate_company_en.csv"), index=False)

# Generate simple management alerts (text file)
alerts = []
for pid, t in summary_tables.items():
    if len(t) >= 2:
        last = t.iloc[-1]
        prev = t.iloc[-2]
        lt_pct = (last["Lead_Time_days"] / (prev["Lead_Time_days"] + 1e-9)) - 1
        eff_pct = (last["Efficiency_u_per_h"] / (prev["Efficiency_u_per_h"] + 1e-9)) - 1
        if lt_pct > PCT_CHANGE_ALERT:
            alerts.append(f"ALERT: Lead Time increase for product {pid} in last month: {lt_pct*100:.1f}%")
        if eff_pct < -PCT_CHANGE_ALERT:
            alerts.append(f"ALERT: Efficiency drop for product {pid} in last month: {eff_pct*100:.1f}%")

with open(os.path.join(OUT_DIR, "management_alerts_en.txt"), "w", encoding="utf-8") as f:
    if alerts:
        f.write("\n".join(alerts))
    else:
        f.write("No critical alerts detected.\n")

# Final dataset with time metrics saved
df.to_csv(os.path.join(OUT_DIR, "cosmetics_kpi_with_time_metrics_en.csv"), index=False)
print(f"[INFO] All outputs written to {OUT_DIR}")
