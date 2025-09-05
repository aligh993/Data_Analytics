# @author: ALI GHANBARI 
# @email: alighanbari446@gmail.com

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. خواندن دیتاست
try:
    df = pd.read_csv('project_data.csv')
except FileNotFoundError:
    print("فایل 'project_data.csv' یافت نشد. لطفا فایل را در مسیر صحیح قرار دهید.")
    exit()

# تبدیل ستون تاریخ به فرمت Datetime
df['Date'] = pd.to_datetime(df['Date'])

# 2. تحلیل‌های کلیدی
print("--- تحلیل‌های کلیدی ---")
# سود کل در کل دوره
total_profit = (df['Profit_Per_Unit'] * df['Sales_Volume']).sum()
print(f"سود کل در این دوره: {total_profit:,.2f}")

# میانگین بهره‌وری تولید بر اساس دسته محصول
avg_efficiency_by_product = df.groupby('Product_Category')['Efficiency_Rate'].mean().reset_index()
print("\nمیانگین بهره‌وری تولید بر اساس دسته محصول:\n", avg_efficiency_by_product)

# ارتباط بین Lead Time و بهره‌وری
correlation = df[['Lead_Time_Days', 'Efficiency_Rate']].corr()
print("\nماتریس همبستگی بین Lead Time و بهره‌وری:\n", correlation)

# 3. ویژوال‌سازی داده‌ها
sns.set_style("whitegrid")
plt.style.use('fivethirtyeight')

# نمودار خطی روند سود در طول زمان
plt.figure(figsize=(12, 6))
df_monthly_profit = df.groupby(df['Date'].dt.to_period('M'))[['Revenue_Per_Unit', 'Sales_Volume']].apply(lambda x: (x['Revenue_Per_Unit'] * x['Sales_Volume']).sum()).reset_index(name='Total_Profit')
df_monthly_profit['Date'] = df_monthly_profit['Date'].dt.to_timestamp()
sns.lineplot(x='Date', y='Total_Profit', data=df_monthly_profit)
plt.title('روند سود کلی ماه به ماه')
plt.xlabel('تاریخ')
plt.ylabel('سود')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# نمودار میله‌ای برای مقایسه بهره‌وری بر اساس کانال فروش
plt.figure(figsize=(10, 6))
sns.barplot(x='Sales_Channel', y='Efficiency_Rate', data=df)
plt.title('مقایسه بهره‌وری تولید بر اساس کانال فروش')
plt.xlabel('کانال فروش')
plt.ylabel('میانگین نرخ بهره‌وری')
plt.show()


# نمودار پراکندگی برای بررسی ارتباط بین هزینه مواد اولیه و سود
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Raw_Material_Cost', y='Profit_Per_Unit', data=df, hue='Product_Category')
plt.title('رابطه بین هزینه مواد اولیه و سود')
plt.xlabel('هزینه مواد اولیه')
plt.ylabel('سود به ازای هر واحد')
plt.show()



'''
project_data.csv:

Date,Product_ID,Product_Category,Lead_Time_Days,Efficiency_Rate,Raw_Material_Cost,Production_Volume,Sales_Volume,Revenue_Per_Unit,Profit_Per_Unit,Sales_Channel,Region,Campaign,Acquisition_Rate,Retention_Rate,Churn_Rate
2024-01-01,P001,Electronics,5,0.85,50,100,80,120,70,Online,North,Campaign_A,0.15,0.85,0.15
2024-01-01,P002,Furniture,12,0.78,150,50,45,300,150,Store,South,Campaign_B,0.10,0.90,0.10
2024-01-02,P001,Electronics,6,0.88,52,110,95,125,73,Online,North,Campaign_A,0.16,0.84,0.16
2024-01-02,P002,Furniture,11,0.80,148,55,50,300,152,Store,South,Campaign_B,0.11,0.89,0.11
2024-01-03,P001,Electronics,5,0.90,48,120,100,130,82,Online,East,Campaign_C,0.18,0.82,0.18
2024-01-03,P002,Furniture,10,0.82,145,60,55,310,165,Store,West,Campaign_D,0.09,0.91,0.09
...

'''

