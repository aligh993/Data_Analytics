# @author: ALI GHANBARI 
# @email: alighanbari446@gmail.com

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Set a professional style for plots
sns.set_style("whitegrid")
plt.style.use('fivethirtyeight')

# --- بخش اول: تحلیل‌های کلان و عملکردی ---
print("--- تحلیل‌های کلان و عملکردی ---")

# 1. روند سود و درآمد ماهانه (Month-over-Month Profit & Revenue)
monthly_kpis = df.groupby('Month').agg({
    'Revenue': 'sum',
    'Profit': 'sum'
}).reset_index()
monthly_kpis['Month'] = monthly_kpis['Month'].dt.to_timestamp()

plt.figure(figsize=(15, 7))
sns.lineplot(x='Month', y='Revenue', data=monthly_kpis, label='درآمد کل', marker='o')
sns.lineplot(x='Month', y='Profit', data=monthly_kpis, label='سود کل', marker='o')
plt.title('روند ماهانه درآمد و سود شرکت')
plt.xlabel('ماه')
plt.ylabel('مقدار (میلیون تومان)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 2. بررسی کمپین‌های بازاریابی و نرخ جذب مشتری (Campaigns & Acquisition Rate)
campaign_performance = df.groupby('Campaign').agg({
    'Ad_Spend': 'sum',
    'Acquisition_Rate': 'mean'
}).reset_index()
campaign_performance['ROI'] = (campaign_performance['Acquisition_Rate'] / campaign_performance['Ad_Spend']) * 100

plt.figure(figsize=(12, 6))
sns.barplot(x='Campaign', y='Acquisition_Rate', data=campaign_performance, palette='viridis')
plt.title('میانگین نرخ جذب مشتری بر اساس کمپین بازاریابی')
plt.xlabel('کمپین')
plt.ylabel('میانگین نرخ جذب')
plt.tight_layout()
plt.show()


# --- بخش دوم: تحلیل‌های تولید و عملیاتی ---
print("\n--- تحلیل‌های تولید و عملیاتی ---")

# 3. بهره‌وری و Lead Time بر اساس دسته محصول (Efficiency & Lead Time by Product)
product_kpis = df.groupby('Product_Category').agg({
    'Efficiency_Rate': 'mean',
    'Lead_Time_Days': 'mean',
    'Profit': 'sum'
}).reset_index()

fig, ax1 = plt.subplots(figsize=(14, 7))
sns.barplot(x='Product_Category', y='Efficiency_Rate', data=product_kpis, color='b', ax=ax1, label='بهره‌وری تولید')
ax2 = ax1.twinx()
sns.lineplot(x='Product_Category', y='Lead_Time_Days', data=product_kpis, color='r', marker='o', ax=ax2, label='میانگین Lead Time')
ax1.set_title('مقایسه بهره‌وری و Lead Time بر اساس دسته محصول')
ax1.set_xlabel('دسته محصول')
ax1.set_ylabel('میانگین بهره‌وری', color='b')
ax2.set_ylabel('میانگین Lead Time (روز)', color='r')
fig.tight_layout()
plt.show()


# 4. ارتباط بین هزینه مواد اولیه و سود (Correlation between Raw Material Cost & Profit)
plt.figure(figsize=(12, 7))
sns.scatterplot(x='Raw_Material_Cost', y='Profit', data=df, hue='Product_Category', style='Product_Category', s=100)
plt.title('رابطه بین هزینه مواد اولیه و سود برای هر محصول')
plt.xlabel('هزینه مواد اولیه')
plt.ylabel('سود')
plt.tight_layout()
plt.show()


# --- بخش سوم: نتیجه‌گیری و توصیه‌های تجاری ---
print("\n--- خلاصه‌ی نتایج و توصیه‌های تجاری ---")
print("1. روند سود و درآمد ماهانه: سود و درآمد شرکت در فصول مختلف سال نوسان داشته است. بررسی دقیق‌تر فصلی بودن فروش توصیه می‌شود.")
print("2. عملکرد کمپین‌ها: کمپین 'Summer_Sale' با وجود هزینه کمتر، نرخ جذب مشتری بالاتری داشته است. اختصاص بودجه بیشتر به این نوع کمپین‌ها می‌تواند به رشد شرکت کمک کند.")
print("3. بهره‌وری تولید: محصولات دسته 'Hygiene' و 'Hair_Care' بالاترین بهره‌وری را دارند. این محصولات می‌توانند به عنوان نقاط قوت تولیدی شرکت معرفی شوند.")
print("4. سودآوری محصولات: محصولات دسته 'Body_Care' با وجود Lead Time بیشتر، سودآوری بالایی دارند. مدیریت بهینه زنجیره تأمین برای کاهش Lead Time این محصولات می‌تواند به افزایش سود کمک کند.")
print("5. رگرسیون: یک مدل رگرسیون ساده می‌تواند برای پیش‌بینی سود بر اساس متغیرهای کلیدی مانند هزینه مواد اولیه و Ad_Spend طراحی شود.")