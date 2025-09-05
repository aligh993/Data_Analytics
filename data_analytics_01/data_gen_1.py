# @author: ALI GHANBARI 
# @email: alighanbari446@gmail.com

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def create_random_data(num_records):
    """
    این تابع یک دیتاست تصادفی با ستون‌های مشخص شده می‌سازد.
    :param num_records: تعداد رکوردهای مورد نظر برای دیتاست.
    :return: یک دیتافریم (DataFrame) پانداس.
    """
    # ایجاد یک بازه‌ی زمانی
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_records)
    dates = pd.to_datetime([start_date + timedelta(days=x) for x in range(num_records)])

    # ستون‌های ابعادی
    regions = ['North', 'South', 'East', 'West']
    campaigns = ['Campaign_A', 'Campaign_B', 'Campaign_C']
    product_categories = ['Electronics', 'Furniture', 'Apparel']
    customer_segments = ['New', 'Returning', 'VIP']
    sales_channels = ['Online', 'Store', 'Phone']

    # تولید داده‌های تصادفی
    data = {
        'Date': dates,
        'Month': dates.month,
        'Year': dates.year,
        'Quarter': dates.quarter,
        
        'Acquisition_Rate': np.random.uniform(0.05, 0.25, num_records),
        'Retention_Rate': np.random.uniform(0.70, 0.95, num_records),
        'Churn_Rate': np.random.uniform(0.05, 0.30, num_records),
        'Customer_Satisfaction_Score': np.random.randint(60, 100, num_records),
        'Website_Traffic': np.random.randint(1000, 50000, num_records),
        
        'Region': np.random.choice(regions, num_records),
        'Campaign': np.random.choice(campaigns, num_records),
        'Product_Category': np.random.choice(product_categories, num_records),
        'Customer_Segment': np.random.choice(customer_segments, num_records),
        'Sales_Channel': np.random.choice(sales_channels, num_records)
    }

    df = pd.DataFrame(data)

    # ایجاد ستون‌های وابسته (Revenue, Cost, Profit)
    df['Cost'] = np.random.randint(50, 500, num_records)
    df['Revenue'] = df['Cost'] * np.random.uniform(1.2, 2.5, num_records)
    df['Profit'] = df['Revenue'] - df['Cost']
    
    return df

# استفاده از تابع برای ایجاد دیتاست با 1000 رکورد
df_project = create_random_data(1000)

# نمایش 5 ردیف اول دیتاست
print(df_project.head())

# ذخیره‌ی دیتاست در یک فایل CSV
file_name = 'project_kpi_data.csv'
df_project.to_csv(file_name, index=False)
print(f"\nدیتاست با موفقیت در فایل '{file_name}' ذخیره شد.")