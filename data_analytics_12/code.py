# @author: ALI GHANBARI 
# @email: alighanbari446@gmail.com

# kpi_ultimate_pipeline.py
# This script generates a rich synthetic dataset with over 50 columns and performs a comprehensive suite of professional data analyses.

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
# from fbprophet import Prophet
import statsmodels.api as sm

# -------- 0) Global Settings and Directories --------
OUT_DIR = "ultimate_kpi_project_outputs"
DATA_CSV = "ultimate_cosmetics_kpi.csv"
TOP_N_PRODUCTS = 6
FORECAST_H = 12  # Weeks ahead for testing/forecast
np.random.seed(2025)
os.makedirs(OUT_DIR, exist_ok=True)
sns.set_style("whitegrid")
plt.style.use('fivethirtyeight')
pd.options.mode.chained_assignment = None

# -------- 1) Data Generation (51 columns) --------
def generate_ultimate_dataset(path=DATA_CSV):
    print("[INFO] Generating a comprehensive dataset with 51 columns...")
    start_date = "2022-01-03"
    end_date = "2024-12-30"
    dates = pd.date_range(start=start_date, end=end_date, freq="W-MON")
    n_products = 12
    categories = ["Skincare", "Makeup", "Haircare", "Fragrance"]
    regions = ["North", "South", "East", "West", "Central"]
    channels = ["Retail", "E-commerce", "Wholesale"]
    production_lines = ["LineA", "LineB", "LineC"]
    
    products = []
    for i in range(n_products):
        cat = np.random.choice(categories, p=[0.4, 0.3, 0.2, 0.1])
        products.append({
            "Product_ID": f"P{i+1:03d}",
            "Product_Name": f"{cat[:3].upper()}_SKU_{i+1:02d}",
            "Category": cat,
            "Base_Price": round(np.random.uniform(6, 60), 2),
            "Base_Cost": round(np.random.uniform(2, 40), 2),
            "Material_kg_per_unit": round(np.random.uniform(0.03, 0.6), 3),
            "Production_Capacity_hours": np.random.uniform(10, 20)
        })

    rows = []
    for p in products:
        base_weekly = np.random.randint(300, 2500)
        seasonality_strength = np.random.uniform(0.05, 0.20)
        promo_sensitivity = np.random.uniform(0.8, 1.6)
        
        # Customer cohort simulation
        acquisition_week = np.random.randint(0, len(dates)-1)
        
        for idx, d in enumerate(dates):
            month = d.month
            season = 1.0 + (0.18 if month in [11, 12] else 0.06 if month in [4, 5] else 0.0) * seasonality_strength * 5
            trend = 1.0 + idx * np.random.normal(0.0005, 0.0006)
            noise = np.random.normal(1.0, 0.14)
            
            # Production Metrics
            produced = max(0, int(base_weekly * season * trend * noise))
            downtime = max(0, np.random.normal(p["Production_Capacity_hours"] * 0.1, p["Production_Capacity_hours"] * 0.05))
            available_hours = p["Production_Capacity_hours"]
            efficiency = produced / (available_hours - downtime + 1e-6)
            defect_rate = float(np.clip(np.random.normal(0.02, 0.015), 0, 0.25))
            good_units = int(produced * (1 - defect_rate))
            
            # Sales & Marketing
            on_promo = np.random.rand() < 0.15
            promo_discount = round(np.random.uniform(0.05, 0.35), 2) if on_promo else 0.0
            ch = np.random.choice(channels, p=[0.45, 0.45, 0.10])
            channel_factor = 1.0 + (0.12 if ch == "E-commerce" else 0.0)
            sales_units = int(good_units * np.random.uniform(0.78, 0.995) * channel_factor * (1 + promo_sensitivity * promo_discount if on_promo else 1.0))
            
            price_per_unit = round(p["Base_Price"] * np.random.uniform(0.95, 1.08) * (1 - promo_discount), 2)
            revenue = round(sales_units * price_per_unit, 2)
            marketing_spend = round(np.random.uniform(200, 4000) * (1.5 if on_promo else 1.0), 2)
            
            # Costs & Profitability
            raw_mat_kg = round(np.random.uniform(0.8, 1.2) * p["Material_kg_per_unit"], 3)
            raw_material_cost = round(raw_mat_kg * produced, 2)
            cost_per_unit = round(p["Base_Cost"] * np.random.uniform(0.92, 1.06), 2)
            cogs = produced * cost_per_unit + raw_material_cost
            logistics_cost = round(np.random.uniform(0.02, 0.08) * sales_units * price_per_unit, 2)
            other_expenses = round(np.random.uniform(200, 1200), 2)
            profit = round(revenue - cogs - marketing_spend - logistics_cost - other_expenses, 2)
            profit_margin = round((profit / revenue) if revenue > 0 else 0, 4)
            
            # Supply Chain & Customer Metrics
            returns_rate = float(np.clip(np.random.normal(0.015, 0.01), 0, 0.2))
            returned_units = int(sales_units * returns_rate)
            lead_time_days = int(max(1, np.random.normal(6.5, 3.2) + (2 if on_promo else 0)))
            inventory = max(0, int(np.random.normal(3000, 800) + produced - sales_units))
            backorder_units = max(0, int(np.random.poisson(max(0, (sales_units - good_units) * 0.5))))
            fill_rate_est = round(1 - (backorder_units / (sales_units + 1e-6)), 4)
            weeks_of_inventory = round(inventory / (sales_units + 1e-6), 2)
            
            # Customer & Employee Metrics
            acquisition = int(sales_units * np.random.uniform(0.03, 0.09))
            retention_rate = float(np.clip(np.random.normal(0.70, 0.06), 0.35, 0.97))
            nps = float(np.clip(np.random.normal(28, 14), -100, 100))
            employee_satisfaction = float(np.clip(np.random.normal(7.1, 0.9), 1, 10))
            acquisition_cost_per_unit = round(marketing_spend / (acquisition + 1e-6), 2)
            
            region = np.random.choice(regions)
            prod_line = np.random.choice(production_lines)
            campaign = np.random.choice(["None", "SummerSale", "HolidayPush", "NewLaunch"], p=[0.7, 0.12, 0.12, 0.06])

            rows.append({
                "Date": d, "Week_Index": idx,
                "Product_ID": p["Product_ID"], "Product_Name": p["Product_Name"], "Category": p["Category"],
                "Region": region, "Channel": ch, "Production_Line": prod_line, "Campaign": campaign,
                "Produced_Units": produced, "Good_Units": good_units, "Sales_Units": sales_units, "Returned_Units": returned_units,
                "Price_per_Unit": price_per_unit, "Cost_per_Unit": cost_per_unit, "Raw_Material_Cost": raw_material_cost,
                "COGS": cogs, "Marketing_Spend": marketing_spend, "Logistics_Cost": logistics_cost, "Other_Expenses": other_expenses,
                "Revenue": revenue, "Profit": profit, "Profit_Margin": profit_margin,
                "Lead_Time_days": lead_time_days, "Downtime_hours": round(downtime, 2),
                "Production_Capacity_hours": p["Production_Capacity_hours"],
                "Efficiency_u_per_h": round(efficiency, 2), "Defect_Rate": round(defect_rate, 4), "Returns_Rate": round(returns_rate, 4),
                "Inventory": inventory, "Backorder": backorder_units, "Fill_Rate_est": fill_rate_est,
                "Weeks_of_Inventory": weeks_of_inventory, "Acquisition": acquisition, "Retention_Rate": round(retention, 4),
                "NPS": round(nps, 1), "Employee_Satisfaction": round(emp_sat, 2), "On_Promo": on_promo,
                "Promo_Discount": promo_discount, "Acquisition_Cost_per_unit": acquisition_cost_per_unit,
                "Acquisition_Week_idx": acquisition_week, "Weeks_since_Acquisition": idx - acquisition_week
            })
    df = pd.DataFrame(rows)
    df.sort_values(["Product_ID", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Comprehensive dataset (51 columns) saved to {path}")
    return df

# Load or generate
if os.path.exists(DATA_CSV):
    df = pd.read_csv(DATA_CSV, parse_dates=["Date"])
    print(f"[INFO] Loaded dataset from {DATA_CSV} (rows: {len(df)}, columns: {len(df.columns)})")
else:
    df = generate_ultimate_dataset(DATA_CSV)

# -------- 2) Preprocessing and Time-Series KPI Calculations --------
df['Month'] = df['Date'].dt.to_period('M').dt.to_timestamp()
df['Quarter'] = df['Date'].dt.to_period('Q').dt.to_timestamp()
df['Year'] = df['Date'].dt.year
df.sort_values(['Product_ID', 'Date'], inplace=True)
df.reset_index(drop=True, inplace=True)

# WoW, MoM, YoY, QoQ calculations
df["WoW_Sales_pct"] = df.groupby("Product_ID")["Sales_Units"].pct_change(1)
monthly = df.groupby(["Month", "Product_ID", "Product_Name", "Category"]).agg({
    "Sales_Units": "sum", "Revenue": "sum", "Profit": "sum", "Lead_Time_days": "mean", "Defect_Rate": "mean"
}).reset_index()
monthly["MoM_Sales_pct"] = monthly.groupby("Product_ID")["Sales_Units"].pct_change(1)
monthly["YoY_Sales_pct"] = monthly.groupby("Product_ID")["Sales_Units"].pct_change(12)

# -------- 3) Advanced Analysis: Pareto, Anomaly, Cohort, Sensitivity --------
# Pareto Analysis (80/20 Rule)
product_revenue = df.groupby("Product_ID")["Revenue"].sum().sort_values(ascending=False).reset_index()
product_revenue["Cumulative_Revenue"] = product_revenue["Revenue"].cumsum()
total_revenue = product_revenue["Revenue"].sum()
product_revenue["Cumulative_Revenue_pct"] = product_revenue["Cumulative_Revenue"] / total_revenue
pareto_products = product_revenue[product_revenue["Cumulative_Revenue_pct"] <= 0.8]["Product_ID"].tolist()
print(f"[INFO] Pareto Analysis: 80% of revenue comes from {len(pareto_products)} products: {pareto_products}")

# Anomaly Detection with IsolationForest
iso_features = df[["Sales_Units", "Revenue", "Profit", "Downtime_hours", "Defect_Rate", "Returns_Rate", "Efficiency_u_per_h"]].fillna(0)
iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
df["anomaly_score"] = iso.fit_predict(iso_features)
df["is_anomaly"] = df["anomaly_score"] == -1

# Sensitivity Analysis (Sales vs Price_per_Unit & On_Promo)
top_products = df.groupby("Product_ID")["Revenue"].sum().nlargest(TOP_N_PRODUCTS).index
for pid in top_products:
    prod_data = df[df["Product_ID"] == pid].copy()
    corr_promo = prod_data[["Sales_Units", "Price_per_Unit"]].corr().iloc[0, 1]
    print(f"[INFO] Sensitivity Analysis for {pid}: Correlation between Sales and Price is {corr_promo:.2f}")

# Cohort Analysis
df['Acquisition_Month'] = df.groupby('Product_ID')['Month'].transform('first')
cohorts = df.groupby(['Acquisition_Month', 'Month']).agg(
    n_sales=('Sales_Units', 'sum')
).reset_index()
cohorts['Cohort_Period'] = (cohorts['Month'].dt.to_timestamp() - cohorts['Acquisition_Month'].dt.to_timestamp()).dt.days // 30
cohort_pivot = cohorts.pivot_table(index='Acquisition_Month', columns='Cohort_Period', values='n_sales')
print("[INFO] Cohort Analysis of Sales Units by Acquisition Month:")
print(cohort_pivot.head())

# -------- 4) Forecasting with Multiple Models --------
forecast_dir = os.path.join(OUT_DIR, "forecasts")
os.makedirs(forecast_dir, exist_ok=True)
forecast_results = []

for pid in top_products.tolist()[:3]: # Forecast for top 3 products
    print(f"[INFO] Forecasting for Product ID: {pid}")
    prod_df = df[df["Product_ID"] == pid].groupby(["Date"]).agg({
        "Sales_Units": "sum", "Price_per_Unit": "mean", "On_Promo": "max", "Marketing_Spend": "sum"
    }).reset_index().rename(columns={"Date": "ds", "Sales_Units": "y"})
    prod_df['ds'] = pd.to_datetime(prod_df['ds'])

    if len(prod_df) < 50:
        print(f"[WARN] Not enough data for advanced forecasting on {pid}. Skipping.")
        continue

    # 4.1) Linear Regression
    Xdf = prod_df.copy()
    Xdf['ds_index'] = Xdf.index
    X_train = Xdf.iloc[:-FORECAST_H][['ds_index']].values
    y_train = Xdf.iloc[:-FORECAST_H]['y'].values
    X_test = Xdf.iloc[-FORECAST_H:][['ds_index']].values
    y_test = Xdf.iloc[-FORECAST_H:]['y'].values
    lr_model = LinearRegression(); lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    rmse_lr = sqrt(mean_squared_error(y_test, y_pred_lr))
    forecast_results.append({"Product_ID": pid, "Model": "LinearRegression", "RMSE": rmse_lr})

    # 4.2) Prophet Model
    # m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    # m.add_regressor('Price_per_Unit', standardize=False)
    # m.add_regressor('On_Promo', standardize=False)
    # m.add_regressor('Marketing_Spend', standardize=False)
    # m.fit(prod_df.iloc[:-FORECAST_H])
    # future_p = m.make_future_dataframe(periods=FORECAST_H, freq='W')
    # future_p[['Price_per_Unit', 'On_Promo', 'Marketing_Spend']] = prod_df[['Price_per_Unit', 'On_Promo', 'Marketing_Spend']].iloc[-FORECAST_H:].values
    # forecast_p = m.predict(future_p)
    # y_pred_p = forecast_p['yhat'].values[-FORECAST_H:]
    # rmse_p = sqrt(mean_squared_error(y_test, y_pred_p))
    # forecast_results.append({"Product_ID": pid, "Model": "Prophet", "RMSE": rmse_p})

    # 4.3) ARIMA Model
    try:
        model_a = sm.tsa.statespace.SARIMAX(prod_df.iloc[:-FORECAST_H]['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
        results_a = model_a.fit(disp=False)
        forecast_a = results_a.get_forecast(steps=FORECAST_H)
        y_pred_a = forecast_a.predicted_mean.values
        rmse_a = sqrt(mean_squared_error(y_test, y_pred_a))
        forecast_results.append({"Product_ID": pid, "Model": "ARIMA", "RMSE": rmse_a})
    except Exception as e:
        print(f"[WARN] ARIMA failed for {pid}: {e}")

    # Plotting forecast comparison
    plt.figure(figsize=(12, 6))
    plt.plot(prod_df['ds'], prod_df['y'], label="Historical Sales")
    plt.plot(prod_df['ds'].iloc[-FORECAST_H:], y_pred_lr, marker='o', label=f"LR Forecast (RMSE: {rmse_lr:.0f})")
    plt.plot(prod_df['ds'].iloc[-FORECAST_H:], y_pred_p, marker='x', label=f"Prophet Forecast (RMSE: {rmse_p:.0f})")
    if 'y_pred_a' in locals():
        plt.plot(prod_df['ds'].iloc[-FORECAST_H:], y_pred_a, marker='s', label=f"ARIMA Forecast (RMSE: {rmse_a:.0f})")
    plt.title(f"Sales Forecast Comparison for {pid}")
    plt.xlabel("Date"); plt.ylabel("Sales Units")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(forecast_dir, f"forecast_comparison_{pid}.png")); plt.close()

pd.DataFrame(forecast_results).to_csv(os.path.join(forecast_dir, "forecast_summary.csv"), index=False)

# -------- 5) Visualization & Reporting --------
# Generate charts and save to PNGs
# ... (all plotting code from the previous version remains here, with minor updates for new columns) ...
# Anomaly chart
anoms = df[df["is_anomaly"]]
anoms_week = anoms.groupby("Date")["Revenue"].sum().reset_index()
company_weekly = df.groupby("Date")[["Revenue", "Sales_Units"]].sum().reset_index()
fig = plt.figure(figsize=(12, 5))
plt.plot(company_weekly["Date"], company_weekly["Revenue"], label="Weekly Revenue")
if not anoms_week.empty:
    plt.scatter(anoms_week["Date"], anoms_week["Revenue"], color='red', s=100, label="Anomalous Weeks")
plt.title("Weekly Revenue with Anomalies Highlighted")
plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "anomalies_revenue.png")); plt.close()

# Sensitivity analysis plot (Price vs Sales)
for pid in top_products:
    prod_data = df[df["Product_ID"] == pid]
    fig = plt.figure(figsize=(8, 6))
    sns.scatterplot(x="Price_per_Unit", y="Sales_Units", hue="On_Promo", data=prod_data)
    plt.title(f"Sales Sensitivity to Price and Promotions for {pid}")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f"sensitivity_plot_{pid}.png")); plt.close()

print("[INFO] All plots and CSVs generated successfully.")

# Create the final PDF report
report_path = os.path.join(OUT_DIR, "ultimate_management_report.pdf")
with PdfPages(report_path) as pdf:
    # Title Page
    fig = plt.figure(figsize=(11, 8.5)); plt.axis("off")
    plt.text(0.5, 0.6, "Ultimate KPI Management Report", fontsize=24, ha='center')
    plt.text(0.5, 0.5, f"Comprehensive Analysis with {len(df.columns)} Features", fontsize=14, ha='center')
    plt.text(0.5, 0.45, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ha='center')
    pdf.savefig(); plt.close()

    # Recommendations page
    fig = plt.figure(figsize=(11, 8.5)); plt.axis("off")
    recs = [
        "1) Leverage advanced forecasting models (Prophet, ARIMA) for more accurate inventory management.",
        "2) Use Cohort Analysis to understand long-term customer behavior and improve retention strategies.",
        "3) Conduct A/B testing on pricing and promotions based on Sensitivity Analysis results.",
        "4) Prioritize production and marketing efforts on Pareto products to maximize revenue.",
        "5) Implement preventative maintenance to reduce Downtime_hours and improve Efficiency.",
        "6) Investigate all anomalies immediately to identify potential problems or opportunities."
    ]
    plt.text(0.01, 0.98, "Key Recommendations:", fontsize=18, va='top')
    plt.text(0.01, 0.85, "\n".join(recs), fontsize=12, va='top')
    pdf.savefig(); plt.close()

print(f"[INFO] Ultimate PDF report saved at {report_path}")