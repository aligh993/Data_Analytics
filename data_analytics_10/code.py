# @author: ALI GHANBARI 
# @email: alighanbari446@gmail.com

# kpi_full_pipeline_fixed.py
# (fixed: .fillna(method='bfill') -> .bfill(); removed incorrect .dt.to_timestamp() calls)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import joblib
from datetime import datetime

# -------- settings --------
OUT_DIR = "outputs_kpi_project"
DATA_CSV = "cosmetics_kpi_rich.csv"  # if missing, the script generates a sample dataset
TOP_N_PRODUCTS = 6
FORECAST_H = 12  # weeks ahead for testing/forecast
ALERT_PCT = 0.10  # 10% threshold for alerts
np.random.seed(2025)
os.makedirs(OUT_DIR, exist_ok=True)
sns.set_style("whitegrid")
plt.style.use('fivethirtyeight')
pd.options.mode.chained_assignment = None

# -------- 1) generate sample dataset if CSV missing --------
def generate_sample_dataset(path=DATA_CSV):
    print("[INFO] Generating sample dataset...")
    start_date = "2022-01-03"
    end_date = "2024-12-30"
    dates = pd.date_range(start=start_date, end=end_date, freq="W-MON")
    n_products = 12
    categories = ["Skincare","Makeup","Haircare","Fragrance"]
    regions = ["North","South","East","West","Central"]
    channels = ["Retail","E-commerce","Wholesale"]
    production_lines = ["LineA","LineB","LineC"]

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

    rows=[]
    for p in products:
        base_weekly = np.random.randint(300,2500)
        seasonality_strength = np.random.uniform(0.05,0.20)
        promo_sensitivity = np.random.uniform(0.8,1.6)
        for d in dates:
            week_idx = (d - pd.to_datetime(dates[0])).days // 7
            month = d.month
            season = 1.0 + (0.18 if month in [11,12] else 0.06 if month in [4,5] else 0.0) * seasonality_strength*5
            trend = 1.0 + week_idx * np.random.normal(0.0005,0.0006)
            noise = np.random.normal(1.0,0.14)
            produced = max(0,int(base_weekly * season * trend * noise))
            downtime = max(0, np.random.normal(6,2.5))
            available_hours = 7*24*0.18
            efficiency = produced / (available_hours - downtime + 1e-6)
            defect = float(np.clip(np.random.normal(0.02,0.015),0,0.25))
            good = int(produced*(1-defect))
            on_promo = np.random.rand() < 0.15
            promo_discount = round(np.random.uniform(0.05,0.35),2) if on_promo else 0.0
            ch = np.random.choice(channels, p=[0.45,0.45,0.10])
            channel_factor = 1.0 + (0.12 if ch=="E-commerce" else 0.0)
            sales = int(good * np.random.uniform(0.78,0.995) * channel_factor * (1 + promo_sensitivity*promo_discount if on_promo else 1.0))
            price = round(p["Base_Price"] * np.random.uniform(0.95,1.08) * (1 - promo_discount),2)
            raw_mat_kg = round(np.random.uniform(0.8,1.2) * p["Material_kg_per_unit"],3)
            raw_cost = round(raw_mat_kg * np.random.uniform(1.2,2.5),3) * produced
            cost_unit = round(p["Base_Cost"] * np.random.uniform(0.92,1.06),2)
            cogs = produced*cost_unit + raw_cost*0.1
            marketing = round(np.random.uniform(200,4000)*(1.5 if on_promo else 1.0),2)
            logistics = round(np.random.uniform(0.02,0.08)*sales*price,2)
            other = round(np.random.uniform(200,1200),2)
            revenue = round(sales*price,2)
            profit = round(revenue - cogs - marketing - logistics - other, 2)
            returns_rate = float(np.clip(np.random.normal(0.015,0.01),0,0.2))
            returned = int(sales * returns_rate)
            lead_time = int(max(1, np.random.normal(6.5,3.2) + (2 if on_promo else 0)))
            inventory = max(0, int(np.random.normal(3000,800) + produced - sales))
            backorder = max(0, int(np.random.poisson(max(0,(sales-good)*0.5))))
            acquisition = int(sales * np.random.uniform(0.03,0.09))
            retention = float(np.clip(np.random.normal(0.70,0.06),0.35,0.97))
            nps = float(np.clip(np.random.normal(28,14), -100, 100))
            emp_sat = float(np.clip(np.random.normal(7.1,0.9),1,10))
            region = np.random.choice(regions)
            prod_line = np.random.choice(production_lines)
            campaign = np.random.choice(["None","SummerSale","HolidayPush","NewLaunch"], p=[0.7,0.12,0.12,0.06])
            rows.append({
                "Date": d, "Week_Index": week_idx,
                "Product_ID": p["Product_ID"], "Product_Name": p["Product_Name"], "Category": p["Category"],
                "Region": region, "Channel": ch, "Production_Line": prod_line, "Campaign": campaign,
                "Produced_Units": produced, "Good_Units": good, "Sales_Units": sales, "Returned_Units": returned,
                "Price_per_Unit": price, "Cost_per_Unit": cost_unit, "Raw_Material_Cost": round(raw_cost,2),
                "COGS": round(cogs,2), "Marketing_Spend": marketing, "Logistics_Cost": logistics, "Other_Expenses": other,
                "Revenue": revenue, "Profit": profit, "Profit_Margin": round((profit/revenue) if revenue>0 else 0,4),
                "Lead_Time_days": lead_time, "Downtime_hours": round(downtime,2), "Efficiency_u_per_h": round(efficiency,2),
                "Defect_Rate": round(defect,4), "Returns_Rate": round(returns_rate,4),
                "Inventory": inventory, "Backorder": backorder,
                "Acquisition": acquisition, "Retention_Rate": round(retention,4), "NPS": round(nps,1),
                "Employee_Satisfaction": round(emp_sat,2), "On_Promo": on_promo, "Promo_Discount": promo_discount
            })
    df = pd.DataFrame(rows)
    df.sort_values(["Product_ID","Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Sample dataset saved to {path}")
    return df

# load or generate
if os.path.exists(DATA_CSV):
    df = pd.read_csv(DATA_CSV, parse_dates=["Date"])
    print(f"[INFO] Loaded dataset from {DATA_CSV} (rows: {len(df)})")
else:
    df = generate_sample_dataset(DATA_CSV)

# -------- 2) preprocessing and time columns --------
df['Month'] = df['Date'].dt.to_period('M').dt.to_timestamp()
df['Quarter'] = df['Date'].dt.to_period('Q').dt.to_timestamp()
df['Year'] = df['Date'].dt.year
df.sort_values(['Product_ID','Date'], inplace=True)
df.reset_index(drop=True, inplace=True)

# compute weekly targets, rolling metrics and basic KPIs
weekly_target = df.groupby("Product_ID")["Sales_Units"].mean().reset_index().rename(columns={"Sales_Units":"Weekly_Target"})
weekly_target["Weekly_Target"] = (weekly_target["Weekly_Target"] * 1.05).round().astype(int)
df = df.merge(weekly_target, on="Product_ID", how="left")
df["Target_Achieved"] = df["Sales_Units"] / (df["Weekly_Target"] + 1e-9)
df["Rolling_4w_Sales"] = df.groupby("Product_ID")["Sales_Units"].transform(lambda x: x.rolling(4, min_periods=1).mean())
df["Rolling_13w_Sales"] = df.groupby("Product_ID")["Sales_Units"].transform(lambda x: x.rolling(13, min_periods=1).mean())

# WoW on weekly level
for col in ["Sales_Units","Revenue","Lead_Time_days","Efficiency_u_per_h"]:
    df[f"WoW_pct_{col}"] = df.groupby("Product_ID")[col].pct_change(1)

# monthly/quarterly/yearly aggregates and MoM/QoQ/YoY
monthly = df.groupby(["Month","Product_ID","Product_Name","Category"]).agg({
    "Produced_Units":"sum","Good_Units":"sum","Sales_Units":"sum","Revenue":"sum","COGS":"sum","Profit":"sum",
    "Lead_Time_days":"mean","Efficiency_u_per_h":"mean","Inventory":"mean",
    "Defect_Rate":"mean","Downtime_hours":"mean","Raw_Material_Cost":"sum",
    "Employee_Satisfaction":"mean","NPS":"mean"
}).reset_index().sort_values(["Product_ID","Month"])
monthly["MoM_Sales_pct"] = monthly.groupby("Product_ID")["Sales_Units"].pct_change(1)
monthly["YoY_Sales_pct"] = monthly.groupby("Product_ID")["Sales_Units"].pct_change(12)
monthly["MoM_LeadTime_pct"] = monthly.groupby("Product_ID")["Lead_Time_days"].pct_change(1)
monthly["MoM_Eff_pct"] = monthly.groupby("Product_ID")["Efficiency_u_per_h"].pct_change(1)

quarterly = df.groupby(["Quarter","Product_ID","Product_Name","Category"]).agg({
    "Produced_Units":"sum","Sales_Units":"sum","Revenue":"sum","Lead_Time_days":"mean","Efficiency_u_per_h":"mean"
}).reset_index().sort_values(["Product_ID","Quarter"])
quarterly["QoQ_Sales_pct"] = quarterly.groupby("Product_ID")["Sales_Units"].pct_change(1)
quarterly["YoY_Q_Sales_pct"] = quarterly.groupby("Product_ID")["Sales_Units"].pct_change(4)

yearly = df.groupby(["Year","Product_ID","Product_Name","Category"]).agg({
    "Produced_Units":"sum","Sales_Units":"sum","Revenue":"sum","Lead_Time_days":"mean","Efficiency_u_per_h":"mean"
}).reset_index().sort_values(["Product_ID","Year"])
yearly["YoY_Sales_pct"] = yearly.groupby("Product_ID")["Sales_Units"].pct_change(1)

# -------- 3) summaries and flags --------
latest_year = yearly["Year"].max()
annual_last = yearly[yearly["Year"]==latest_year].copy()
annual_last["Profit_Margin"] = 0
if "Profit" in yearly.columns:
    # avoid division by zero
    annual_last["Profit_Margin"] = annual_last["Profit"] / annual_last["Revenue"].replace(0, np.nan)

top_products = annual_last.sort_values("Revenue", ascending=False).head(TOP_N_PRODUCTS)
top_ids = top_products["Product_ID"].tolist()

summary_tables = {}
for pid in top_ids:
    t = monthly[monthly["Product_ID"] == pid].sort_values("Month").copy()
    t["Rolling_3m_Sales"] = t["Sales_Units"].rolling(3, min_periods=1).mean()
    t["MoM_Drop_Flag"] = t["MoM_Sales_pct"].apply(lambda x: True if (pd.notna(x) and x < -ALERT_PCT) else False)
    summary_tables[pid] = t

company_weekly = df.groupby("Date")[["Sales_Units","Revenue"]].sum().reset_index().sort_values("Date")
company_weekly["WoW_Sales_pct"] = company_weekly["Sales_Units"].pct_change(1)
company_weekly["WoW_Revenue_pct"] = company_weekly["Revenue"].pct_change(1)

# -------- 4) anomaly detection --------
iso_features = df[["Sales_Units","Revenue","Profit","Defect_Rate","Downtime_hours","Efficiency_u_per_h"]].fillna(0)
iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
iso.fit(iso_features)
df["anomaly_score"] = iso.decision_function(iso_features)
df["is_anomaly"] = iso.predict(iso_features) == -1

# -------- 5) Forecasting with LinearRegression (per product) --------
def create_time_features_weekly(prod_week_df):
    dfw = prod_week_df.sort_values("Week_Index").reset_index(drop=True).copy()
    for lag in [1,2,3,4,13]:
        # use bfill() instead of deprecated fillna(method='bfill')
        dfw[f"lag_{lag}"] = dfw["Sales_Units"].shift(lag).bfill()
    dfw["roll_4"] = dfw["Sales_Units"].rolling(4, min_periods=1).mean()
    dfw["roll_13"] = dfw["Sales_Units"].rolling(13, min_periods=1).mean()
    dfw["month"] = dfw["Date"].dt.month
    dfw["month_sin"] = np.sin(2*np.pi*dfw["month"]/12)
    dfw["month_cos"] = np.cos(2*np.pi*dfw["month"]/12)
    for col in ["On_Promo","Marketing_Spend","Price_per_Unit","Inventory","Lead_Time_days"]:
        if col in dfw.columns:
            dfw[col] = dfw[col].fillna(0)
        else:
            dfw[col] = 0
    return dfw

forecast_results = []
forecast_dir = os.path.join(OUT_DIR, "forecasts")
os.makedirs(forecast_dir, exist_ok=True)

for pid in top_ids:
    prod_df = df[df["Product_ID"]==pid].groupby(["Week_Index","Date"]).agg({
        "Sales_Units":"sum",
        "On_Promo":"max" if "On_Promo" in df.columns else None,
        "Marketing_Spend":"sum" if "Marketing_Spend" in df.columns else None,
        "Price_per_Unit":"mean" if "Price_per_Unit" in df.columns else None,
        "Inventory":"mean" if "Inventory" in df.columns else None,
        "Lead_Time_days":"mean" if "Lead_Time_days" in df.columns else None
    }).reset_index()
    prod_df = prod_df.loc[:, ~prod_df.columns.isin([None])]
    prod_df = prod_df.sort_values("Week_Index").reset_index(drop=True)
    if len(prod_df) < 30:
        print(f"[WARN] Not enough data for forecasting {pid}")
        continue
    Xdf = create_time_features_weekly(prod_df)
    train = Xdf.iloc[:-FORECAST_H].copy()
    test = Xdf.iloc[-FORECAST_H:].copy()
    feat_cols = [c for c in Xdf.columns if c not in ["Date","Week_Index","Sales_Units"]]
    X_train = train[feat_cols].values
    y_train = train["Sales_Units"].values
    X_test = test[feat_cols].values
    y_test = test["Sales_Units"].values
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mask = y_test != 0
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask]))*100 if mask.sum()>0 else np.nan
    forecast_results.append({
        "Product_ID": pid,
        "n_weeks": len(prod_df),
        "rmse": rmse, "mae": mae, "mape_pct": mape
    })
    # save forecast csv
    future_weeks = np.arange(prod_df["Week_Index"].max()+1, prod_df["Week_Index"].max()+1+FORECAST_H)
    df_fore = pd.DataFrame({"Week_Index": future_weeks})
    df_fore["LinearReg_pred"] = model.predict(X_test)
    df_fore.to_csv(os.path.join(forecast_dir, f"forecast_linreg_{pid}.csv"), index=False)
    # plot
    plt.figure(figsize=(10,4))
    plt.plot(prod_df["Week_Index"], prod_df["Sales_Units"], label="Historical")
    plt.plot(test["Week_Index"], y_pred, label="LinearReg Forecast", marker='o')
    plt.title(f"Forecast (LinearRegression) - {pid}")
    plt.xlabel("Week_Index"); plt.ylabel("Sales Units")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(forecast_dir, f"forecast_plot_{pid}.png"))
    plt.close()

# save forecast summary
pd.DataFrame(forecast_results).to_csv(os.path.join(forecast_dir, "forecast_summary_linreg.csv"), index=False)

# -------- 6) Produce plots (one figure per file) --------
company_weekly = df.groupby("Date")[["Revenue","Sales_Units"]].sum().reset_index().sort_values("Date")
company_weekly["WoW_Sales_pct"] = company_weekly["Sales_Units"].pct_change(1)
fig, ax = plt.subplots(2,1, figsize=(14,8), gridspec_kw={"height_ratios":[3,1]}, sharex=True)
ax0, ax1 = ax
ax0.plot(company_weekly["Date"], company_weekly["Revenue"], marker='.', label="Revenue")
ax0.plot(company_weekly["Date"], company_weekly["Sales_Units"], marker='.', label="Sales Units")
ax0.set_title("Weekly Revenue & Sales Trends")
ax0.legend()
ax1.bar(company_weekly["Date"], company_weekly["WoW_Sales_pct"]*100)
ax1.axhline(0, color='black', linewidth=0.7)
ax1.set_ylabel("WoW % (sales)")
ax1.set_xlabel("Date")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "company_weekly_trends_and_wow.png"))
plt.close(fig)

# Monthly revenue & profit line (use Month as datetime directly)
monthly_kpis = df.groupby("Month").agg({"Revenue":"sum","Profit":"sum","Marketing_Spend":"sum"}).reset_index()
# ensure Month is datetime (safe)
monthly_kpis["Month"] = pd.to_datetime(monthly_kpis["Month"].astype(str))
fig = plt.figure(figsize=(12,6))
plt.plot(monthly_kpis["Month"], monthly_kpis["Revenue"], marker='o', label="Revenue")
plt.plot(monthly_kpis["Month"], monthly_kpis["Profit"], marker='o', label="Profit")
plt.title("Monthly Trends: Revenue & Profit")
plt.xlabel("Month"); plt.ylabel("Amount")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "monthly_revenue_profit.png"))
plt.close(fig)

# Quarterly Efficiency (bar) and Lead Time (line)
quarterly_prod = df.groupby("Quarter").agg({"Efficiency_u_per_h":"mean","Lead_Time_days":"mean"}).reset_index()
# convert Quarter to string for x-axis labels (safe)
quarterly_prod["Quarter_str"] = quarterly_prod["Quarter"].astype(str)
fig, ax = plt.subplots(figsize=(12,6))
ax_bar = ax
ax_bar.bar(quarterly_prod["Quarter_str"], quarterly_prod["Efficiency_u_per_h"], label="Efficiency (units/hour)")
ax2 = ax_bar.twinx()
ax2.plot(quarterly_prod["Quarter_str"], quarterly_prod["Lead_Time_days"], color='red', marker='o', label="Lead Time (days)")
ax_bar.set_title("Quarterly Efficiency and Lead Time")
ax_bar.set_xlabel("Quarter"); ax_bar.set_ylabel("Efficiency")
ax2.set_ylabel("Lead Time (days)")
ax_bar.legend(loc='upper left'); ax2.legend(loc='upper right')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "quarterly_efficiency_leadtime.png"))
plt.close(fig)

# Region performance (bar) + Acquisition rate
region_perf = df.groupby("Region").agg({"Revenue":"sum","Profit":"sum","Acquisition":"mean"}).reset_index()
fig, axes = plt.subplots(1,2, figsize=(14,6))
axes[0].bar(region_perf["Region"], region_perf["Revenue"])
axes[0].set_title("Revenue by Region")
axes[0].set_xlabel("Region"); axes[0].set_ylabel("Revenue")
axes[1].bar(region_perf["Region"], region_perf["Acquisition"])
axes[1].set_title("Average Acquisition Rate by Region")
axes[1].set_xlabel("Region"); axes[1].set_ylabel("Acquisition Rate")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "region_performance.png"))
plt.close(fig)

# Campaign pie chart (share of revenue)
campaign_rev = df.groupby("Campaign")["Revenue"].sum().reset_index()
fig = plt.figure(figsize=(8,6))
plt.pie(campaign_rev["Revenue"], labels=campaign_rev["Campaign"], autopct='%1.1f%%', startangle=140)
plt.title("Campaign revenue share")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "campaign_revenue_share_pie.png"))
plt.close(fig)

# Correlation heatmap (monthly KPIs)
corr_cols = ["Produced_Units","Sales_Units","Revenue","Profit","Defect_Rate","Downtime_hours","Raw_Material_Cost","Employee_Satisfaction","NPS"]
corr = monthly[corr_cols].corr()
fig = plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation matrix (monthly KPIs)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "correlation_matrix_monthly.png"))
plt.close(fig)

# MoM heatmap (products x months)
pivot_mom = monthly.pivot(index="Product_Name", columns="Month", values="MoM_Sales_pct").fillna(0)
fig = plt.figure(figsize=(14,7))
plt.imshow(pivot_mom, aspect='auto', cmap='bwr', vmin=-1, vmax=1)
plt.colorbar(label="MoM % (fraction)")
plt.yticks(range(len(pivot_mom.index)), pivot_mom.index)
plt.xticks(range(len(pivot_mom.columns)), [m.strftime("%Y-%m") for m in pivot_mom.columns], rotation=90)
plt.title("Heatmap: MoM % change - Products x Months")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "mom_heatmap_products_months.png"))
plt.close(fig)

# Scatter Efficiency vs Lead Time for top product
if top_ids:
    pid0 = top_ids[0]
    t0 = monthly[monthly["Product_ID"]==pid0]
    if len(t0) > 5:
        x = t0["Efficiency_u_per_h"].values
        y = t0["Lead_Time_days"].values
        coef = np.polyfit(x,y,1)
        p = np.poly1d(coef)
        fig = plt.figure(figsize=(8,5))
        plt.scatter(x,y, label="monthly points")
        plt.plot(x, p(x), color='red', label=f"fit: y={coef[0]:.3f}x+{coef[1]:.3f}")
        plt.title(f"Efficiency vs Lead Time - {pid0}")
        plt.xlabel("Efficiency (units/hour)"); plt.ylabel("Lead Time (days)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"eff_vs_leadtime_{pid0}.png"))
        plt.close(fig)

# anomalies over time (highlight)
anoms = df[df["is_anomaly"]]
anoms_week = anoms.groupby("Date")["Revenue"].sum().reset_index()
fig = plt.figure(figsize=(12,5))
plt.plot(company_weekly["Date"], company_weekly["Revenue"], label="Revenue")
if not anoms_week.empty:
    plt.scatter(anoms_week["Date"], anoms_week["Revenue"], color='red', label="Anomalous weeks")
plt.title("Weekly Revenue with Anomalous Weeks Highlighted")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "anomalies_revenue.png"))
plt.close(fig)

# inventory weeks distribution
# if Weeks_of_Inventory exists (from CSV) use it; otherwise compute approx
if "Weeks_of_Inventory" not in df.columns:
    df["Weeks_of_Inventory"] = df["Inventory"] / (df["Sales_Units"].replace(0, np.nan) + 1e-6)
fig = plt.figure(figsize=(8,4))
plt.hist(df["Weeks_of_Inventory"].replace(np.inf, np.nan).dropna(), bins=40)
plt.title("Distribution of Weeks of Inventory")
plt.xlabel("Weeks of Inventory"); plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "weeks_inventory_hist.png"))
plt.close(fig)

# -------- 7) Save tables CSVs --------
df.to_csv(os.path.join(OUT_DIR, "cosmetics_kpi_full_with_metrics.csv"), index=False)
monthly.to_csv(os.path.join(OUT_DIR, "monthly_aggregates_with_metrics.csv"), index=False)
annual_last.to_csv(os.path.join(OUT_DIR, "annual_kpis_last_year.csv"), index=False)
print("[INFO] CSV outputs saved in", OUT_DIR)

# -------- 8) Management PDF report (Persian text, KPI names in English) --------
report_path = os.path.join(OUT_DIR, "management_report.pdf")
with PdfPages(report_path) as pdf:
    # title page
    fig = plt.figure(figsize=(11,8.5)); plt.axis("off")
    plt.text(0.5,0.6,"Cosmetics/Health Products KPI Management Report", fontsize=20, ha='center')
    plt.text(0.5,0.52,"(Key metrics: Sales_Units, Revenue, Profit, Lead_Time_days, Efficiency_u_per_h)", fontsize=11, ha='center')
    plt.text(0.5,0.45,f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ha='center')
    pdf.savefig(); plt.close()

    # add a few key plots pages (company trend, top product monthly, anomalies)
    for fname in ["company_weekly_trends_and_wow.png", "monthly_revenue_profit.png", "anomalies_revenue.png"]:
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.exists(fpath):
            img = plt.imread(fpath)
            fig = plt.figure(figsize=(11,8.5)); plt.imshow(img); plt.axis("off")
            pdf.savefig(); plt.close()

    # summary table text
    fig = plt.figure(figsize=(11,8.5)); plt.axis("off")
    plt.text(0.01,0.98,"Summary: Top products by Revenue (last year)", fontsize=14, va='top')
    txt = annual_last.sort_values("Revenue", ascending=False).head(10).to_string(index=False)
    plt.text(0.01,0.92, txt, fontsize=9, family='monospace', va='top')
    pdf.savefig(); plt.close()

    # add a page with 8 actionable recommendations (Persian with KPI names in English)
    fig = plt.figure(figsize=(11,8.5)); plt.axis("off")
    recs = [
        "1) Focus marketing on the top 20% of products by sales (Pareto) — monitor Sales_Units & Revenue.",
        "2) Reduce Defect_Rate and Downtime_hours through a preventative maintenance program — increase Efficiency_u_per_h.",
        "3) Use weekly forecasts (LinearRegression) for inventory planning and raw material procurement (Raw_Material_Cost).",
        "4) Schedule promotions by evaluating the MoM and YoY effects of campaigns (Campaign).",
        "5) Reduce Weeks_of_Inventory for fast-moving SKUs and increase for seasonal ones.",
        "6) Investigate anomalies (anomaly) before making decisions; find root causes (logistics/returns/campaign).",
        "7) Automate weekly reports with key KPIs: Sales_Units, Revenue, Profit, Lead_Time_days, Efficiency_u_per_h.",
        "8) Conduct A/B testing for campaigns and evaluate CAC and Conversion (in different channels)."
    ]
    plt.text(0.01,0.98,"Recommendations (Summary):", fontsize=14, va='top')
    plt.text(0.01,0.90, "\n".join(recs), fontsize=12, va='top')
    pdf.savefig(); plt.close()

print("[INFO] Management PDF report saved at", report_path)
print("All done. Outputs in folder:", OUT_DIR)