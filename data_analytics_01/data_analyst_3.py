# @author: ALI GHANBARI 
# @email: alighanbari446@gmail.com

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the generated dataset
try:
    df = pd.read_csv('hygiene_products_kpis.csv')
except FileNotFoundError:
    print("فایل دیتاست یافت نشد. لطفا کد تولید دیتا را اجرا کنید.")
    exit()

# Data Cleaning & Preparation
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.to_period('M')
df['Year'] = df['Date'].dt.year
df['Quarter'] = df['Date'].dt.quarter
df['Week'] = df['Date'].dt.to_period('W')

# Set a professional style for plots
sns.set_style("whitegrid")
plt.style.use('fivethirtyeight')

# --- بخش اول: تحلیل‌های توصیفی ---
print("--- تحلیل‌های توصیفی (Descriptive Analysis) ---")
# Summary statistics for key KPIs
kpi_summary = df[['Revenue', 'Profit', 'Acquisition_Rate', 'Efficiency_Rate', 'Lead_Time_Days', 'Customer_Satisfaction_Score']].describe()
print("آمار توصیفی KPIهای اصلی:\n", kpi_summary)

# --- بخش دوم: تحلیل‌های زمان‌محور (Time-based Analysis) ---
print("\n--- تحلیل‌های زمان‌محور ---")

# 1. روند سود و درآمد ماهانه (Month-over-Month & Year-over-Year)
monthly_kpis = df.groupby('Month').agg({
    'Revenue': 'sum',
    'Profit': 'sum',
    'Acquisition_Rate': 'mean',
    'Website_Traffic': 'sum'
}).reset_index()
monthly_kpis['Month'] = monthly_kpis['Month'].dt.to_timestamp()

plt.figure(figsize=(15, 7))
sns.lineplot(x='Month', y='Revenue', data=monthly_kpis, label='درآمد کل', marker='o')
sns.lineplot(x='Month', y='Profit', data=monthly_kpis, label='سود کل', marker='o')
plt.title('روند ماهانه درآمد و سود شرکت')
plt.xlabel('ماه')
plt.ylabel('مقدار')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 2. روند بهره‌وری تولید و Lead Time فصلی
quarterly_production = df.groupby('Quarter').agg({
    'Efficiency_Rate': 'mean',
    'Lead_Time_Days': 'mean'
}).reset_index()

fig, ax1 = plt.subplots(figsize=(12, 6))
sns.barplot(x='Quarter', y='Efficiency_Rate', data=quarterly_production, color='b', ax=ax1)
ax2 = ax1.twinx()
sns.lineplot(x='Quarter', y='Lead_Time_Days', data=quarterly_production, color='r', marker='o', ax=ax2)
ax1.set_title('میانگین بهره‌وری و Lead Time فصلی')
ax1.set_xlabel('فصل')
ax1.set_ylabel('میانگین بهره‌وری', color='b')
ax2.set_ylabel('میانگین Lead Time (روز)', color='r')
plt.show()


# --- بخش سوم: تحلیل‌های مقایسه‌ای و سگمنت‌بندی (Comparative & Segmentation Analysis) ---
print("\n--- تحلیل‌های مقایسه‌ای ---")

# 3. مقایسه عملکرد بر اساس منطقه جغرافیایی
region_performance = df.groupby('Region').agg({
    'Revenue': 'sum',
    'Profit': 'sum',
    'Acquisition_Rate': 'mean'
}).reset_index()

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.barplot(x='Region', y='Revenue', data=region_performance)
plt.title('درآمد کل بر اساس منطقه')
plt.xlabel('منطقه')
plt.ylabel('درآمد')

plt.subplot(1, 2, 2)
sns.barplot(x='Region', y='Acquisition_Rate', data=region_performance)
plt.title('میانگین نرخ جذب مشتری بر اساس منطقه')
plt.xlabel('منطقه')
plt.ylabel('نرخ جذب')
plt.tight_layout()
plt.show()


# 4. مقایسه عملکرد کمپین‌ها
campaign_kpis = df.groupby('Campaign').agg({
    'Revenue': 'sum',
    'Acquisition_Rate': 'mean',
    'Customer_Satisfaction_Score': 'mean'
}).reset_index()

plt.figure(figsize=(10, 6))
plt.pie(campaign_kpis['Revenue'], labels=campaign_kpis['Campaign'], autopct='%1.1f%%', startangle=140)
plt.title('سهم هر کمپین از کل درآمد')
plt.show()


# --- بخش چهارم: تحلیل ارتباطی (Correlation Analysis) ---
print("\n--- تحلیل ارتباطی ---")

# 5. ارتباط بین ترافیک وب‌سایت و نرخ جذب مشتری
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Website_Traffic', y='Acquisition_Rate', data=df)
plt.title('رابطه بین ترافیک وب‌سایت و نرخ جذب مشتری')
plt.xlabel('ترافیک وب‌سایت')
plt.ylabel('نرخ جذب مشتری')
plt.show()


# 6. Heatmap برای ماتریس همبستگی
correlation_matrix = df[['Revenue', 'Profit', 'Website_Traffic', 'Acquisition_Rate', 'Retention_Rate', 'Customer_Satisfaction_Score']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('ماتریس همبستگی بین KPIهای کلیدی')
plt.show()


# --- بخش پنجم: تحلیل پیش‌بینی (Predictive Analysis) ---
print("\n--- تحلیل پیش‌بینی ---")
# این بخش نیاز به یک مدل یادگیری ماشین دارد که در اینجا یک مثال ساده ارائه می‌شود.
from sklearn.linear_model import LinearRegression

# آماده‌سازی داده برای مدل رگرسیون
X = df[['Website_Traffic', 'Raw_Material_Cost']]
y = df['Revenue']

model = LinearRegression()
model.fit(X, y)
print(f"مدل رگرسیون ساده برای پیش‌بینی درآمد ساخته شد.")
print(f"ضریب (Coefficient) ترافیک وب‌سایت: {model.coef_[0]:.2f}")
print(f"ضریب هزینه مواد اولیه: {model.coef_[1]:.2f}")
print("این ضرایب نشان می‌دهند که هر واحد افزایش در این متغیرها، چه تأثیری بر درآمد دارد.")

# --- نتیجه‌گیری و توصیه‌های تجاری ---
print("\n--- خلاصه‌ی نتایج و توصیه‌های تجاری ---")
print("این تحلیل‌های جامع، دیدگاه‌های ارزشمندی را در مورد عملکرد تولیدی، بازاریابی و فروش شرکت ارائه می‌دهند و می‌توانند مبنای تصمیم‌گیری‌های استراتژیک قرار گیرند.")

# %%
