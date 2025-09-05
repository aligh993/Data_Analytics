# @author: ALI GHANBARI 
# @email: alighanbari446@gmail.com

#  pip install pandas numpy matplotlib scikit-learn lightgbm prophet matplotlib-venn statsmodels


# cosmetics_forecast_pipeline.py
# -------------------------------------------------------
# اجرای کامل: دیتا -> ویژگی -> Prophet & LightGBM -> مقایسه -> PDF گزارش
# توضیحات و عناوین (تا حد ممکن) به فارسی هستند.
# -------------------------------------------------------

import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Try imports for Prophet and LightGBM, set flags if available
use_prophet = False
use_lgb = False
try:
    try:
        from prophet import Prophet
    except Exception:
        from fbprophet import Prophet
    use_prophet = True
except Exception:
    print("Prophet نصب نیست — برای استفاده از Prophet نصب کنید: pip install prophet (یا fbprophet)")
try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    use_lgb = True
except Exception:
    print("LightGBM نصب نیست — برای بهترین نتایج نصب کنید: pip install lightgbm")
    # we'll fallback to RandomForest if LGBM not available

# ---------------------------
# 1) بارگذاری دیتاست یا تولید دیتاست نمونه
# ---------------------------
DATA_CSV = "cosmetics_kpi_rich.csv"

def generate_sample_dataset(path=DATA_CSV, seed=42):
    """تولید دیتاست مصنوعی غنی (3 سال، چند SKU و ستون‌های عملیاتی)"""
    np.random.seed(seed)
    start_date = "2022-01-03"
    end_date = "2024-12-30"
    dates = pd.date_range(start=start_date, end=end_date, freq="W-MON")
    products = []
    categories = ["Skincare", "Makeup", "Haircare", "Fragrance"]
    regions = ["North","South","East","West","Central"]
    channels = ["Retail","E-commerce","Wholesale"]
    production_lines = ["LineA","LineB","LineC"]
    n_products = 12
    for i in range(n_products):
        p = {
            "Product_ID": f"P{i+1:03d}",
            "Product_Name": f"{np.random.choice(categories)[:3].upper()}_SKU_{i+1:02d}",
            "Category": np.random.choice(categories),
            "Base_Price": round(np.random.uniform(6,60),2),
            "Base_Cost": round(np.random.uniform(2,40),2),
            "Material_kg_per_unit": round(np.random.uniform(0.03,0.6),3)
        }
        products.append(p)
    rows=[]
    for p in products:
        base = np.random.randint(300,2500)
        for d in dates:
            week_index = (d - pd.to_datetime(start_date)).days // 7
            month = d.month
            season = 1.0 + (0.18 if month in [11,12] else 0.06 if month in [4,5] else 0.0)
            noise = np.random.normal(1.0,0.14)
            produced = max(0,int(base*season*noise))
            downtime = max(0,np.random.normal(6,2.5))
            available_hours = 7*24*0.18
            efficiency = produced/(available_hours-downtime+1e-6)
            defect = float(np.clip(np.random.normal(0.02,0.015),0,0.25))
            good = int(produced*(1-defect))
            on_promo = np.random.rand()<0.15
            promo_discount = round(np.random.uniform(0.05,0.35),2) if on_promo else 0.0
            ch = np.random.choice(channels,p=[0.45,0.45,0.10])
            channel_factor = 1.0 + (0.12 if ch=="E-commerce" else 0.0)
            sales = int(good*np.random.uniform(0.78,0.995)*channel_factor*(1+0.8*promo_discount if on_promo else 1.0))
            price = round(p["Base_Price"]*np.random.uniform(0.95,1.08)*(1-promo_discount),2)
            material_cost = round(np.random.uniform(0.8,1.2)*p["Material_kg_per_unit"],3)*produced
            cost_unit = round(p["Base_Cost"]*np.random.uniform(0.92,1.06),2)
            cogs = produced*cost_unit + material_cost*0.1
            marketing = round(np.random.uniform(200,4000)*(1.5 if on_promo else 1.0),2)
            logistics = round(np.random.uniform(0.02,0.08)*sales*price,2)
            other = round(np.random.uniform(200,1200),2)
            revenue = round(sales*price,2)
            profit = round(revenue - cogs - marketing - logistics - other,2)
            returns_rate = float(np.clip(np.random.normal(0.015,0.01),0,0.2))
            returned = int(sales*returns_rate)
            lead_time = int(max(1,np.random.normal(6.5,3.2) + (2 if on_promo else 0)))
            inventory = max(0,int(np.random.normal(3000,800) + produced - sales))
            backorder = max(0,int(np.random.poisson(max(0,(sales-good)*0.5))))
            acquisition = int(sales*np.random.uniform(0.03,0.09))
            retention = float(np.clip(np.random.normal(0.70,0.06),0.35,0.97))
            nps = float(np.clip(np.random.normal(28,14),-100,100))
            emp_sat = float(np.clip(np.random.normal(7.1,0.9),1,10))
            region = np.random.choice(regions)
            prod_line = np.random.choice(production_lines)
            campaign = np.random.choice(["None","SummerSale","HolidayPush","NewLaunch"],p=[0.7,0.12,0.12,0.06])
            rows.append({
                "Date": d, "Week_Index": week_index,
                "Product_ID": p["Product_ID"], "Product_Name": p["Product_Name"], "Category": p["Category"],
                "Region": region, "Channel": ch, "Production_Line": prod_line, "Campaign": campaign,
                "Produced_Units": produced, "Good_Units": good, "Sales_Units": sales, "Returned_Units": returned,
                "Price_per_Unit": price, "Cost_per_Unit": cost_unit, "Raw_Material_Cost": round(material_cost,2),
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
    print(f"دیتاست نمونه ایجاد و در {path} ذخیره شد.")
    return df

if not os.path.exists(DATA_CSV):
    df = generate_sample_dataset(DATA_CSV)
else:
    df = pd.read_csv(DATA_CSV, parse_dates=["Date"])
    print(f"دیتاست از {DATA_CSV} بارگذاری شد. (ردیف‌ها: {len(df)})")

# ---------------------------
# 2) پیش‌پردازش و KPI ها
# ---------------------------
df['Month'] = df['Date'].dt.to_period('M').dt.to_timestamp()
df['Year'] = df['Date'].dt.year
df['Quarter'] = df['Date'].dt.to_period('Q').dt.to_timestamp()

# Weekly target (synthetic) و محاسبات Rolling, MoM, YoY
targets = df.groupby("Product_ID")["Sales_Units"].mean().reset_index().rename(columns={"Sales_Units":"Weekly_Target"})
targets["Weekly_Target"] = (targets["Weekly_Target"]*1.05).round().astype(int)
df = df.merge(targets, on="Product_ID", how="left")
df["Target_Achieved"] = df["Sales_Units"] / (df["Weekly_Target"] + 1e-9)
df["Rolling_4w_Sales"] = df.groupby("Product_ID")["Sales_Units"].transform(lambda x: x.rolling(4, min_periods=1).mean())
df["Rolling_13w_Sales"] = df.groupby("Product_ID")["Sales_Units"].transform(lambda x: x.rolling(13, min_periods=1).mean())
df["MoM_Sales_pct"] = df.groupby("Product_ID")["Sales_Units"].transform(lambda x: x.pct_change(4))
df["YoY_Sales_pct"] = df.groupby("Product_ID")["Sales_Units"].transform(lambda x: x.pct_change(52))
df["Weeks_of_Inventory"] = df["Inventory"] / (df["Sales_Units"].replace(0, np.nan) + 1e-9)
df["Fill_Rate_est"] = 1 - (df["Backorder"] / (df["Sales_Units"].replace(0, np.nan) + 1e-9))

# ---------------------------
# 3) انتخاب محصولات Top برای پیش‌بینی (بر اساس Revenue)
# ---------------------------
annual_rev = df.groupby("Product_ID").agg({"Revenue":"sum","Product_Name":"first"}).reset_index()
TOP_K = 3
top_prods = annual_rev.sort_values("Revenue", ascending=False).head(TOP_K)["Product_ID"].tolist()
print("محصولات منتخب برای پیش‌بینی (Product_ID):", top_prods)

# Map product english->persian (ساده و جزئی)
def translate_name(en_name):
    # تغییرات جزئی برای خوانایی فارسی
    return en_name.replace("SKI","پوست").replace("MAK","آرایش").replace("HAI","مو").replace("FRA","عطر").replace("_SKU_"," - محصول شماره ")

prod_name_map = {}
for pid in top_prods:
    en = df.loc[df["Product_ID"]==pid, "Product_Name"].iloc[0]
    prod_name_map[pid] = {"en": en, "fa": translate_name(en)}

# ---------------------------
# 4) توابع کمکی برای فیچرینگ و ارزیابی
# ---------------------------
def create_lag_features(series, lags=[1,2,3,4,13], roll_windows=[4,13]):
    df_l = pd.DataFrame({"y": series.values})
    for lag in lags:
        df_l[f"lag_{lag}"] = series.shift(lag).fillna(method="bfill").values
    for w in roll_windows:
        df_l[f"roll_{w}"] = series.rolling(w, min_periods=1).mean().values
    return df_l

def evaluate_forecast(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    # sklearn's mape may warn on zeros; handle
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred)
    except Exception:
        mape = np.nan
    return {"mse": float(mse), "mae": float(mae), "mape": float(mape)}

# ---------------------------
# 5) مدل LightGBM با TimeSeriesSplit + GridSearch
# ---------------------------
def train_lgb_cv(X, y, n_splits=5):
    """برگرداندن مدل آموزش‌دیده LGBM با TimeSeriesSplit CV و بهترین پارامتر"""
    if not use_lgb:
        return None, None
    tscv = TimeSeriesSplit(n_splits=n_splits)
    param_grid = {
        "num_leaves":[31,63],
        "n_estimators":[100,300],
        "learning_rate":[0.05,0.1]
    }
    model = LGBMRegressor(random_state=42)
    gs = GridSearchCV(model, param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1, verbose=0)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_

# ---------------------------
# 6) Prophet training (با regressors اختیاری)
# ---------------------------
def train_prophet(train_df, regressors=None):
    """train_df must have Date and y (sales). regressors: dict of name->series aligned with train_df"""
    if not use_prophet:
        return None
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    if regressors:
        for r in regressors:
            m.add_regressor(r)
    m.fit(train_df)
    return m

# ---------------------------
# 7) اجرای پیش‌بینی برای هر محصول و مقایسه مدل‌ها
# ---------------------------
H = 12  # horizon in weeks (holdout)
comparison_records = []
forecast_plots = []

for pid in top_prods:
    prod_df = df[df["Product_ID"]==pid].sort_values("Week_Index").copy()
    weekly = prod_df.groupby("Week_Index").agg({"Date":"first","Sales_Units":"sum"}).reset_index()
    if len(weekly) < 40:
        print(f"{pid} دادهٔ کافی ندارد، رد می‌شود.")
        continue
    # train/test split: last H weeks test
    train = weekly.iloc[:-H].copy()
    test = weekly.iloc[-H:].copy()
    # --- Prophet ---
    prophet_result = {"pred": None, "metrics": None}
    if use_prophet:
        # prepare prophet dataframe
        df_prop = train.rename(columns={"Date":"ds","Sales_Units":"y"})[["ds","y"]]
        # include regressors if available (On_Promo, Marketing_Spend aggregated weekly)
        # Build weekly regressors from prod_df
        weekly_extra = prod_df.groupby("Week_Index").agg({"On_Promo":"max","Marketing_Spend":"sum"}).reindex(weekly["Week_Index"]).fillna(0).reset_index()
        df_prop["On_Promo"] = weekly_extra.iloc[:len(train)]["On_Promo"].astype(int).values
        df_prop["Marketing_Spend"] = weekly_extra.iloc[:len(train)]["Marketing_Spend"].values
        try:
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m.add_regressor("On_Promo")
            m.add_regressor("Marketing_Spend")
            m.fit(df_prop)
            future = m.make_future_dataframe(periods=H, freq='W-MON')
            # attach regressors for future
            fut_extra = weekly_extra["On_Promo"].values[:len(future)]
            future["On_Promo"] = fut_extra
            future["Marketing_Spend"] = weekly_extra["Marketing_Spend"].values[:len(future)]
            fcst = m.predict(future)
            prophet_pred = fcst.tail(H)["yhat"].values
            prophet_metrics = evaluate_forecast(test["Sales_Units"].values, prophet_pred)
            prophet_result = {"pred": prophet_pred, "metrics": prophet_metrics}
        except Exception as e:
            prophet_result = {"pred": None, "metrics": None}
    # --- LightGBM / RF ---
    lgb_result = {"pred": None, "metrics": None, "model_info": None}
    # create features from weekly['Sales_Units']
    feat_df = create_lag_features(weekly["Sales_Units"])
    # Add month as cyclical features
    dates = weekly["Date"].dt
    months = weekly["Date"].dt.month.values
    feat_df["month_sin"] = np.sin(2*np.pi*(months/12))
    feat_df["month_cos"] = np.cos(2*np.pi*(months/12))
    # train/test
    X = feat_df.iloc[:-H].values
    y = weekly["Sales_Units"].iloc[:-H].values
    X_test = feat_df.iloc[-H:].values
    # scale if needed
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xt = scaler.transform(X_test)
    if use_lgb:
        try:
            best_model, best_params = train_lgb_cv(X, y, n_splits=4)
            pred_lgb = best_model.predict(X_test)
            lgb_metrics = evaluate_forecast(weekly["Sales_Units"].iloc[-H:].values, pred_lgb)
            lgb_result = {"pred": pred_lgb, "metrics": lgb_metrics, "model_info": best_params}
        except Exception as e:
            # fallback to RandomForest
            rf = RandomForestRegressor(n_estimators=200, random_state=42)
            rf.fit(Xs, y)
            pred_rf = rf.predict(Xt)
            rf_metrics = evaluate_forecast(weekly["Sales_Units"].iloc[-H:].values, pred_rf)
            lgb_result = {"pred": pred_rf, "metrics": rf_metrics, "model_info": {"fallback":"RandomForest"}}
    else:
        # use RandomForest fallback
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(Xs, y)
        pred_rf = rf.predict(Xt)
        rf_metrics = evaluate_forecast(weekly["Sales_Units"].iloc[-H:].values, pred_rf)
        lgb_result = {"pred": pred_rf, "metrics": rf_metrics, "model_info": {"fallback":"RandomForest"}}

    # ذخیره نتایج مقایسه
    rec = {
        "Product_ID": pid,
        "Product_Name_en": prod_name_map[pid]["en"] if pid in prod_name_map else weekly["Product_Name"].iloc[0] if "Product_Name" in weekly.columns else pid,
        "Product_Name_fa": prod_name_map.get(pid, {}).get("fa", prod_name_map.get(pid,{}).get("en","")),
        "Prophet_mse": prophet_result["metrics"]["mse"] if prophet_result["metrics"] else np.nan,
        "Prophet_mape": prophet_result["metrics"]["mape"] if prophet_result["metrics"] else np.nan,
        "LGB_mse": lgb_result["metrics"]["mse"] if lgb_result["metrics"] else np.nan,
        "LGB_mape": lgb_result["metrics"]["mape"] if lgb_result["metrics"] else np.nan,
        "LGB_info": lgb_result["model_info"]
    }
    comparison_records.append(rec)

    # نمودار مقایسه: تاریخی vs پیش‌بینی‌ها
    fig, ax = plt.subplots(figsize=(11,4))
    ax.plot(weekly["Week_Index"], weekly["Sales_Units"], label="مقادیر تاریخی", marker='.', linewidth=1)
    test_idx = weekly["Week_Index"].iloc[-H:].values
    if prophet_result["pred"] is not None:
        ax.plot(test_idx, prophet_result["pred"], label="پیش‌بینی Prophet", marker='o')
    if lgb_result["pred"] is not None:
        ax.plot(test_idx, lgb_result["pred"], label="پیش‌بینی LightGBM/RF", marker='o')
    ax.set_title(f"مقایسه پیش‌بینی - {prod_name_map[pid]['fa']} — {prod_name_map[pid]['en']}")
    ax.set_xlabel("Week Index"); ax.set_ylabel("Sales Units")
    ax.legend(); plt.tight_layout()
    figpath = f"forecast_comp_{pid}.png"
    fig.savefig(figpath); forecast_plots.append(figpath)
    plt.close(fig)

# DataFrame مقایسه مدل‌ها
comp_df = pd.DataFrame(comparison_records)
comp_csv = "forecast_model_comparison.csv"
comp_df.to_csv(comp_csv, index=False)
print("مقایسه مدل‌ها ذخیره شد:", comp_csv)

# ---------------------------
# 8) ساخت PDF گزارش جامع
# ---------------------------
report_pdf = "cosmetics_forecast_report_full.pdf"
with PdfPages(report_pdf) as pdf:
    # صفحه عنوان
    fig=plt.figure(figsize=(11,8.5)); plt.axis('off')
    plt.text(0.5,0.6,"گزارش پیش‌بینی و مقایسه مدل‌ها — پروژه KPI محصولات بهداشتی و آرایشی", ha='center', fontsize=18)
    plt.text(0.5,0.5,f"تاریخ: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", ha='center')
    plt.text(0.5,0.38,"مدل‌ها: Prophet (سری‌زمانی با regressors) و LightGBM (با TimeSeries CV). در صورت عدم نصب LightGBM از RandomForest استفاده شد.", ha='center', fontsize=10)
    pdf.savefig(); plt.close()

    # صفحه جدول مقایسه
    fig=plt.figure(figsize=(11,8.5)); plt.axis('off')
    plt.title("خلاصه مقایسه مدل‌ها (Top products)", fontsize=14)
    txt = comp_df.round(3).to_string(index=False)
    plt.text(0.01,0.98,"جدول مقایسه (مقادیر خطا: MSE و MAPE)", fontsize=12, va='top')
    plt.text(0.01,0.88,txt, fontsize=9, va='top', family='monospace')
    pdf.savefig(); plt.close()

    # نمودارهای هر محصول
    for p in forecast_plots:
        try:
            img = plt.imread(p)
            fig = plt.figure(figsize=(11,8.5)); plt.imshow(img); plt.axis('off'); pdf.savefig(); plt.close()
        except Exception:
            pass

    # ماتریس همبستگی KPIهای ماهانه (اختیاری)
    try:
        monthly = df.groupby(["Month","Product_ID"]).agg({"Produced_Units":"sum","Sales_Units":"sum","Revenue":"sum","Profit":"sum","Defect_Rate":"mean","Downtime_hours":"sum","Raw_Material_Cost":"sum","Employee_Satisfaction":"mean","NPS":"mean"}).reset_index()
        corr_cols = ["Produced_Units","Sales_Units","Revenue","Profit","Defect_Rate","Downtime_hours","Raw_Material_Cost","Employee_Satisfaction","NPS"]
        corr = monthly[corr_cols].corr()
        fig = plt.figure(figsize=(11,8.5)); plt.title("ماتریس همبستگی KPIهای ماهیانه", fontsize=14)
        plt.imshow(corr); plt.colorbar(); plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha='right'); plt.yticks(range(len(corr_cols)), corr_cols)
        pdf.savefig(); plt.close()
    except Exception:
        pass

    # توصیه‌های عملی (8 مورد)
    fig=plt.figure(figsize=(11,8.5)); plt.axis('off')
    recs = [
        "1. تمرکز بودجه تبلیغاتی بر ۲۰٪ محصولات پرفروش (قانون پارتو).",
        "2. کاهش Defect Rate با نگهداری پیشگیرانه و مانیتورینگ OEE خطوط تولید.",
        "3. استفاده از پیش‌بینی‌های Prophet/LightGBM برای برنامه‌ریزی خرید مواد اولیه و ظرفیت تولید.",
        "4. تیونینگ LightGBM با TimeSeries CV و هایپرپارامترها برای افزایش دقت.",
        "5. اضافه کردن regressors (promotions, marketing_spend, holidays) به مدل‌های سری‌زمانی.",
        "6. اتوماسیون تولید پیش‌بینی هفتگی و داشبورد برای مدیران با KPI کلیدی.",
        "7. آنالیز هفته‌های غیرعادی (آنومالی) قبل از تصمیم‌گیری عملیاتی.",
        "8. آزمون A/B برای سنجش اثر کمپین‌ها و اعتبارسنجی causal impact."
    ]
    plt.text(0.01,0.99,"توصیه‌های عملی (خلاصه)", fontsize=14, va='top')
    plt.text(0.01,0.9,"\n".join(recs), fontsize=12, va='top')
    pdf.savefig(); plt.close()

    # توضیح ستون‌ها (فارسی)
    fig=plt.figure(figsize=(11,8.5)); plt.axis('off')
    plt.text(0.01,0.98,"توضیح ستون‌ها (خلاصه)", fontsize=14, va='top')
    col_desc = {
        "Date":"تاریخ ردیف (هفته)",
        "Product_ID":"شناسه محصول",
        "Product_Name":"نام محصول (انگلیسی)",
        "Category":"دسته‌بندی محصول (مثلاً Skincare -> پوست)",
        "Region":"منطقه",
        "Channel":"کانال فروش (Retail/E-commerce/Wholesale)",
        "Produced_Units":"تعداد تولید شده در هفته",
        "Good_Units":"تعداد سالم تولید شده",
        "Sales_Units":"تعداد فروخته شده در هفته",
        "Returned_Units":"تعداد برگشتی",
        "Price_per_Unit":"قیمت فروش هر واحد",
        "Cost_per_Unit":"هزینه واحد",
        "Raw_Material_Cost":"هزینه مواد خام مصرفی (هفته‌ای)",
        "COGS":"هزینه کالای فروخته شده",
        "Marketing_Spend":"هزینه بازاریابی",
        "Logistics_Cost":"هزینه لجستیک",
        "Other_Expenses":"سایر هزینه‌ها",
        "Revenue":"درآمد",
        "Profit":"سود",
        "Profit_Margin":"حاشیه سود",
        "Lead_Time_days":"زمان تأمین (روز)",
        "Downtime_hours":"ساعات از کار افتادگی خط تولید",
        "Efficiency_u_per_h":"بهره‌وری (تعداد در ساعت موثر)",
        "Defect_Rate":"نرخ نقص تولید",
        "Inventory":"موجودی",
        "Backorder":"سفارشات معوق",
        "Acquisition":"تعداد مشتریان جدید جذب شده",
        "Retention_Rate":"نرخ نگهداری مشتری",
        "NPS":"شاخص NPS",
        "Employee_Satisfaction":"رضایت کارکنان (1-10)",
        "On_Promo":"آیا محصول در هفته پروموشن بوده؟",
        "Promo_Discount":"میزان تخفیف پروموشن (درصد)"
    }
    # نمایش در چند ستون برای خوانایی
    desc_lines = "\n".join([f"{k}: {v}" for k,v in col_desc.items()])
    plt.text(0.01,0.9, desc_lines, fontsize=10, va='top', family='monospace')
    pdf.savefig(); plt.close()

print("گزارش PDF ذخیره شد:", report_pdf)
print("مقایسه مدل‌ها (CSV):", comp_csv)
print("نمودارهای مقایسه در فایل‌های PNG تولید شدند (نام‌هایی مانند forecast_comp_P001.png).")

# پایان اسکریپت
