# @author: ALI GHANBARI 
# @email: alighanbari446@gmail.com

# cosmetics_kpi_time_analysis.py
"""
نسخه حرفه‌ای برای محاسبات Time-series KPI ها:
شامل: WoW, MoM, QoQ, YoY برای Sales, Revenue, Lead_Time_days, Efficiency_u_per_h
و تولید جداول مدیریتی و نمودارها.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---------- تنظیمات ----------
OUT_DIR = "cosmetics_kpi_outputs"
DATA_CSV = "cosmetics_kpi_rich.csv"  # اگر ندارید، این اسکریپت دیتاست نمونه می‌سازد
TOP_N_PRODUCTS = 6  # چند محصول برتر را برای جداول تفصیلی بررسی کنیم
PCT_CHANGE_ALERT = 0.10  # اگر افت > 10% باشه به عنوان هشدار پرچم زده می‌شود

os.makedirs(OUT_DIR, exist_ok=True)
pd.options.mode.chained_assignment = None
np.random.seed(123)

# ---------- 1) بارگذاری یا تولید دیتاست نمونه (اگر فایل موجود نبود) ----------
def generate_sample_dataset(path=DATA_CSV, years=3):
    """تولید دیتاست مصنوعی غنی مشابه آنچه قبلاً داشتیم"""
    start_date = "2022-01-03"
    end_date = "2024-12-30"
    dates = pd.date_range(start=start_date, end=end_date, freq="W-MON")
    categories = ["Skincare", "Makeup", "Haircare", "Fragrance"]
    channels = ["Retail","E-commerce","Wholesale"]
    regions = ["North","South","East","West","Central"]
    products = []
    n_products = 12
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
        base = np.random.randint(300,2500)
        for d in dates:
            week_idx = (d - pd.to_datetime(dates[0])).days // 7
            month = d.month
            season = 1.0 + (0.18 if month in [11,12] else 0.06 if month in [4,5] else 0.0)
            noise = np.random.normal(1.0, 0.14)
            produced = max(0, int(base * season * noise))
            downtime = max(0, np.random.normal(6, 2.5))
            avail_hours = 7*24*0.18
            efficiency = produced / (avail_hours - downtime + 1e-6)
            defect = float(np.clip(np.random.normal(0.02, 0.015), 0, 0.25))
            good = int(produced * (1-defect))
            on_promo = np.random.rand() < 0.15
            promo_discount = round(np.random.uniform(0.05,0.35), 2) if on_promo else 0.0
            ch = np.random.choice(channels, p=[0.45,0.45,0.10])
            sales = int(good * np.random.uniform(0.78,0.995) * (1 + 0.8*promo_discount if on_promo else 1.0))
            price = round(p["Base_Price"] * np.random.uniform(0.95,1.08) * (1 - promo_discount),2)
            raw_cost = round(np.random.uniform(0.8,1.2) * p["Material_kg_per_unit"],3) * produced
            cost_unit = round(p["Base_Cost"] * np.random.uniform(0.92,1.06), 2)
            cogs = produced * cost_unit + raw_cost * 0.1
            marketing = round(np.random.uniform(200,4000) * (1.5 if on_promo else 1.0), 2)
            logistics = round(np.random.uniform(0.02,0.08) * sales * price, 2)
            other = round(np.random.uniform(200,1200), 2)
            revenue = round(sales * price, 2)
            profit = round(revenue - cogs - marketing - logistics - other, 2)
            returns_rate = float(np.clip(np.random.normal(0.015,0.01), 0, 0.2))
            returned = int(sales * returns_rate)
            lead_time = int(max(1, np.random.normal(6.5, 3.2) + (2 if on_promo else 0)))
            inventory = max(0, int(np.random.normal(3000, 800) + produced - sales))
            backorder = max(0, int(np.random.poisson(max(0, (sales-good) * 0.5))))
            acquisition = int(sales * np.random.uniform(0.03,0.09))
            retention = float(np.clip(np.random.normal(0.70, 0.06), 0.35, 0.97))
            nps = float(np.clip(np.random.normal(28, 14), -100, 100))
            emp_sat = float(np.clip(np.random.normal(7.1, 0.9), 1, 10))
            region = np.random.choice(regions)
            campaign = np.random.choice(["None","SummerSale","HolidayPush","NewLaunch"], p=[0.7,0.12,0.12,0.06])
            rows.append({
                "Date": d, "Week_Index": week_idx,
                "Product_ID": p["Product_ID"], "Product_Name": p["Product_Name"], "Category": p["Category"],
                "Region": region, "Channel": ch, "Campaign": campaign,
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
    print(f"[INFO] دیتاست نمونه در {path} ذخیره شد.")
    return df

if os.path.exists(DATA_CSV):
    df = pd.read_csv(DATA_CSV, parse_dates=["Date"])
    print(f"[INFO] دیتاست از {DATA_CSV} بارگذاری شد. تعداد ردیف: {len(df)}")
else:
    df = generate_sample_dataset(DATA_CSV)

# ---------- 2) فرمت زمان و ستون‌های زمان‌بندی ----------
df['Month'] = df['Date'].dt.to_period('M').dt.to_timestamp()
df['Quarter'] = df['Date'].dt.to_period('Q').dt.to_timestamp()
df['Year'] = df['Date'].dt.year
# برای محاسبات WoW لازم است ایندکس بر اساس Date و Product_ID مرتب باشد:
df.sort_values(['Product_ID','Date'], inplace=True)
df.reset_index(drop=True, inplace=True)

# ---------- 3) محاسبه KPIهای پایه و اهداف ----------
# Target (هفته‌ای) به صورت متوسط تاریخی (سنتتیک)
weekly_target = df.groupby("Product_ID")["Sales_Units"].mean().reset_index().rename(columns={"Sales_Units":"Weekly_Target"})
weekly_target["Weekly_Target"] = (weekly_target["Weekly_Target"] * 1.05).round().astype(int)
df = df.merge(weekly_target, on="Product_ID", how="left")
df["Target_Achieved"] = df["Sales_Units"] / (df["Weekly_Target"] + 1e-9)

# Rolling و متریک‌های پایه
df["Rolling_4w_Sales"] = df.groupby("Product_ID")["Sales_Units"].transform(lambda x: x.rolling(4, min_periods=1).mean())
df["Rolling_13w_Sales"] = df.groupby("Product_ID")["Sales_Units"].transform(lambda x: x.rolling(13, min_periods=1).mean())

# ---------- 4) محاسبات WoW, MoM, QoQ, YoY ----------
# --- 4.1 Week-over-Week (WoW) برای Sales, Revenue, Lead_Time_days, Efficiency ---
def compute_wow(df_in, group_cols, value_col, new_col_name):
    """محاسبه WoW درصدی: (this - prev)/prev در سطح group_cols (معمولاً Product_ID)"""
    df_tmp = df_in.sort_values(group_cols + ["Date"]).copy()
    df_tmp[new_col_name] = df_tmp.groupby(group_cols)[value_col].pct_change(1)  # lag 1 week
    return df_tmp[[*group_cols, "Date", value_col, new_col_name]]

for col in ["Sales_Units", "Revenue", "Lead_Time_days", "Efficiency_u_per_h"]:
    df[f"WoW_pct_{col}"] = df.groupby("Product_ID")[col].pct_change(1)

# --- 4.2 Month-over-Month (MoM) (بر اساس تجمیع ماهانه) ---
monthly = df.groupby(["Month","Product_ID","Product_Name","Category"]).agg({
    "Produced_Units":"sum","Good_Units":"sum","Sales_Units":"sum","Revenue":"sum","COGS":"sum","Profit":"sum",
    "Lead_Time_days":"mean","Efficiency_u_per_h":"mean","Inventory":"mean"
}).reset_index().sort_values(["Product_ID","Month"])
monthly["MoM_Sales_pct"] = monthly.groupby("Product_ID")["Sales_Units"].pct_change(1)
monthly["YoY_Sales_pct"] = monthly.groupby("Product_ID")["Sales_Units"].pct_change(12)  # 12 months
monthly["MoM_LeadTime_pct"] = monthly.groupby("Product_ID")["Lead_Time_days"].pct_change(1)
monthly["MoM_Eff_pct"] = monthly.groupby("Product_ID")["Efficiency_u_per_h"].pct_change(1)

# --- 4.3 Quarter-over-Quarter (QoQ) (بر اساس تجمیع فصلی) ---
quarterly = df.groupby(["Quarter","Product_ID","Product_Name","Category"]).agg({
    "Produced_Units":"sum","Sales_Units":"sum","Revenue":"sum","Lead_Time_days":"mean","Efficiency_u_per_h":"mean"
}).reset_index().sort_values(["Product_ID","Quarter"])
quarterly["QoQ_Sales_pct"] = quarterly.groupby("Product_ID")["Sales_Units"].pct_change(1)
quarterly["YoY_Q_Sales_pct"] = quarterly.groupby("Product_ID")["Sales_Units"].pct_change(4)  # 4 quarters = 1 year

# --- 4.4 Year-over-Year (YoY) (بر اساس سالانه) ---
yearly = df.groupby(["Year","Product_ID","Product_Name","Category"]).agg({
    "Produced_Units":"sum","Sales_Units":"sum","Revenue":"sum","Lead_Time_days":"mean","Efficiency_u_per_h":"mean"
}).reset_index().sort_values(["Product_ID","Year"])
yearly["YoY_Sales_pct"] = yearly.groupby("Product_ID")["Sales_Units"].pct_change(1)

# ---------- 5) تولید جداول مدیریتی خلاصه (Summary tables) ----------
# Top products by annual revenue (آخرین سال موجود)
latest_year = yearly["Year"].max()
annual_last = yearly[yearly["Year"]==latest_year].copy()
annual_last["Profit_Margin"] = annual_last["Revenue"].replace(0, np.nan)
# find top N products
top_products = annual_last.sort_values("Revenue", ascending=False).head(TOP_N_PRODUCTS)
top_ids = top_products["Product_ID"].tolist()

# Create a summary table including MoM and YoY for sales for top products (monthly)
summary_tables = {}
for pid in top_ids:
    t_monthly = monthly[monthly["Product_ID"]==pid].copy().sort_values("Month")
    # add rolling 3-month avg
    t_monthly["Rolling_3m_Sales"] = t_monthly["Sales_Units"].rolling(3, min_periods=1).mean()
    # flag significant drops (MoM drop > threshold)
    t_monthly["MoM_Drop_Flag"] = t_monthly["MoM_Sales_pct"].apply(lambda x: True if (pd.notna(x) and x < -PCT_CHANGE_ALERT) else False)
    summary_tables[pid] = t_monthly

# Company-level WoW summary (last 12 weeks)
company_weekly = df.groupby("Date").agg({"Sales_Units":"sum","Revenue":"sum","Profit":"sum"}).reset_index().sort_values("Date")
company_weekly["WoW_Sales_pct"] = company_weekly["Sales_Units"].pct_change(1)
company_weekly["WoW_Revenue_pct"] = company_weekly["Revenue"].pct_change(1)

# ---------- 6) نمودارهای حرفه‌ای برای گزارش مدیریتی ----------
def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {path}")
    return path

# A) Trend: Company weekly Sales & Revenue (with WoW bar below)
fig, ax = plt.subplots(2,1, figsize=(12,7), gridspec_kw={"height_ratios":[3,1]}, sharex=True)
ax0 = ax[0]
ax1 = ax[1]
ax0.plot(company_weekly["Date"], company_weekly["Sales_Units"], marker='.', label="واحدهای فروش (هفتگی)")
ax0.plot(company_weekly["Date"], company_weekly["Revenue"], marker='.', label="درآمد (هفتگی)")
ax0.set_title("روند هفتگی فروش و درآمد (Company-level)")
ax0.legend()
# WoW bar
ax1.bar(company_weekly["Date"], company_weekly["WoW_Sales_pct"]*100)
ax1.axhline(0, color='black', linewidth=0.7)
ax1.set_ylabel("درصد تغییر WoW (%)")
ax1.set_xlabel("تاریخ")
fig.tight_layout()
save_fig(fig, "company_weekly_trends_and_wow.png")

# B) For each top product: trend Sales + MoM% as bar (monthly)
for pid in top_ids:
    t = summary_tables[pid]
    fig, ax = plt.subplots(2,1, figsize=(12,6), gridspec_kw={"height_ratios":[3,1]}, sharex=True)
    ax[0].plot(t["Month"], t["Sales_Units"], marker='o', label="فروش ماهانه")
    ax[0].plot(t["Month"], t["Rolling_3m_Sales"], linestyle='--', label="میانگین متحرک 3 ماهه")
    ax[0].set_title(f"روند فروش ماهانه - محصول {pid} — {t['Product_Name'].iloc[0]}")
    ax[0].legend()
    ax[1].bar(t["Month"], t["MoM_Sales_pct"]*100)
    ax[1].axhline(0, color='black', linewidth=0.7)
    ax[1].set_ylabel("MoM %")
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    save_fig(fig, f"product_{pid}_monthly_sales_and_mom.png")

# C) Lead Time trend (monthly) and YoY comparison for top products
for pid in top_ids:
    t = monthly[monthly["Product_ID"]==pid].copy().sort_values("Month")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(t["Month"], t["Lead_Time_days"], marker='o', label="میانگین Lead Time (روز)")
    ax.set_title(f"روند Lead Time ماهانه - محصول {pid}")
    ax.set_ylabel("روز")
    ax.set_xlabel("ماه")
    fig.autofmt_xdate(rotation=45)
    save_fig(fig, f"product_{pid}_leadtime_trend.png")

# D) Efficiency trend and QoQ % (quarterly) for top products
for pid in top_ids:
    tq = quarterly[quarterly["Product_ID"]==pid].sort_values("Quarter")
    fig, ax = plt.subplots(2,1, figsize=(10,6), gridspec_kw={"height_ratios":[3,1]}, sharex=True)
    ax[0].plot(tq["Quarter"], tq["Efficiency_u_per_h"], marker='o', label="میانگین بهره‌وری (units/hour)")
    ax[0].set_title(f"بهره‌وری (Efficiency) فصلی - محصول {pid}")
    ax[1].bar(tq["Quarter"], tq["QoQ_Sales_pct"]*100)
    ax[1].axhline(0, color='black', linewidth=0.7)
    ax[1].set_ylabel("QoQ % (فروش)")
    fig.autofmt_xdate(rotation=45)
    save_fig(fig, f"product_{pid}_efficiency_and_qoq.png")

# E) Scatter: Efficiency vs Lead Time (monthly) با خط رگرسیون ساده (برای یکی از محصولات)
pid0 = top_ids[0]
t0 = monthly[monthly["Product_ID"]==pid0]
if len(t0) > 5:
    x = t0["Efficiency_u_per_h"].values
    y = t0["Lead_Time_days"].values
    # fit linear
    coef = np.polyfit(x,y,1)
    p = np.poly1d(coef)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(x,y, label="داده‌های ماهانه")
    ax.plot(x, p(x), color='red', label=f"رگرسیون خطی: y={coef[0]:.3f}x+{coef[1]:.3f}")
    ax.set_title(f"بهره‌وری vs Lead Time - محصول {pid0}")
    ax.set_xlabel("Efficiency (units/hour)")
    ax.set_ylabel("Lead Time (days)")
    ax.legend()
    save_fig(fig, f"product_{pid0}_eff_vs_leadtime.png")

# F) Heatmap (matrix) of MoM% for products x months (visual overview)
pivot_mom = monthly.pivot(index="Product_Name", columns="Month", values="MoM_Sales_pct").fillna(0)
fig, ax = plt.subplots(figsize=(12,6))
im = ax.imshow(pivot_mom, aspect='auto', cmap='bwr', vmin=-1, vmax=1)
plt.colorbar(im, label="MoM % (fraction)")
ax.set_yticks(range(len(pivot_mom.index))); ax.set_yticklabels(pivot_mom.index)
ax.set_xticks(range(len(pivot_mom.columns))); ax.set_xticklabels([m.strftime("%Y-%m") for m in pivot_mom.columns], rotation=90)
ax.set_title("Heatmap: MoM درصد تغییر فروش (محصولات × ماه‌ها)")
fig.tight_layout()
save_fig(fig, "mom_heatmap_products_months.png")

# ---------- 7) جداول خروجی برای مدیران ----------
# الف) جدول خلاصه ماهانه برای محصولات برتر (CSV)
for pid, t in summary_tables.items():
    out = os.path.join(OUT_DIR, f"summary_monthly_{pid}.csv")
    t.to_csv(out, index=False)
    print(f"[SAVE] {out}")

# ب) جدول company weekly و WoW
company_weekly_out = os.path.join(OUT_DIR, "company_weekly_wow.csv")
company_weekly.to_csv(company_weekly_out, index=False)
print(f"[SAVE] {company_weekly_out}")

# ج) جدول monthly aggregate کل شرکت
monthly_agg = monthly.groupby("Month").agg({"Sales_Units":"sum","Revenue":"sum","Produced_Units":"sum"}).reset_index()
monthly_agg["MoM_Sales_pct"] = monthly_agg["Sales_Units"].pct_change(1)
monthly_agg.to_csv(os.path.join(OUT_DIR, "monthly_aggregate_company.csv"), index=False)
print(f"[SAVE] {os.path.join(OUT_DIR, 'monthly_aggregate_company.csv')}")

# ---------- 8) گزارش خلاصه (متن ساده) با نکات مدیریتی خودکار ----------
# مثال: بررسی افت‌های قابل توجه در Lead Time و Efficiency در ماه اخیر برای محصولات برتر
alerts = []
for pid, t in summary_tables.items():
    if len(t) >= 2:
        last = t.iloc[-1]
        prev = t.iloc[-2]
        # MoM lead time and efficiency
        lt_pct = last["Lead_Time_days"] / (prev["Lead_Time_days"] + 1e-9) - 1
        eff_pct = last["Efficiency_u_per_h"] / (prev["Efficiency_u_per_h"] + 1e-9) - 1
        if lt_pct > PCT_CHANGE_ALERT:
            alerts.append(f"هشدار: افزایش Lead Time برای محصول {pid} در ماه اخیر: {lt_pct*100:.1f}%")
        if eff_pct < -PCT_CHANGE_ALERT:
            alerts.append(f"هشدار: کاهش Efficiency برای محصول {pid} در ماه اخیر: {eff_pct*100:.1f}%")
# ذخیره
with open(os.path.join(OUT_DIR, "management_alerts.txt"), "w", encoding="utf-8") as f:
    if alerts:
        f.write("\n".join(alerts))
    else:
        f.write("هشدار مهمی یافت نشد.")
print(f"[SAVE] {os.path.join(OUT_DIR, 'management_alerts.txt')}")

# ---------- 9) ذخیره نهایی دیتاست با همه متریک‌ها ----------
df_out_path = os.path.join(OUT_DIR, "cosmetics_kpi_with_time_metrics.csv")
df.to_csv(df_out_path, index=False)
print(f"[SAVE] {df_out_path}")

print("\nتمام خروجی‌ها در پوشه:", OUT_DIR)
print("فایل‌های کلیدی: company_weekly_trends_and_wow.png, product_*_monthly_sales_and_mom.png, product_*_leadtime_trend.png, product_*_efficiency_and_qoq.png, mom_heatmap_products_months.png")
