# @author: ALI GHANBARI 
# @email: alighanbari446@gmail.com

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_data(num_records):
    # Defining possible values for categorical columns
    products = ['Shampoo', 'Soap', 'Hand_Sanitizer', 'Dish_Soap']
    categories = ['Hair_Care', 'Body_Care', 'Hygiene']
    regions = ['North', 'South', 'East', 'West']
    campaigns = ['Summer_Sale', 'New_Product_Launch', 'Hygiene_Awareness']
    customer_segments = ['New', 'Returning', 'VIP']
    sales_channels = ['Online', 'Retail_Store', 'Wholesale']

    # Generating random data
    data = {
        'Date': [datetime.today() - timedelta(days=random.randint(1, 365)) for _ in range(num_records)],
        'Order_ID': [f'ORD-{i:04}' for i in range(1, num_records + 1)],
        'Customer_ID': [f'CUST-{random.randint(1000, 9999)}' for _ in range(num_records)],
        'Product_Name': [random.choice(products) for _ in range(num_records)],
        'Product_Category': [random.choice(categories) for _ in range(num_records)],
        'Region': [random.choice(regions) for _ in range(num_records)],
        'Campaign': [random.choice(campaigns) for _ in range(num_records)],
        'Sales_Channel': [random.choice(sales_channels) for _ in range(num_records)],
        'Customer_Segment': [random.choice(customer_segments) for _ in range(num_records)],
        
        'Website_Traffic': np.random.randint(500, 5000, num_records),
        'Acquisition_Rate': np.random.uniform(0.05, 0.20, num_records),
        'Retention_Rate': np.random.uniform(0.70, 0.95, num_records),
        'Churn_Rate': 1 - np.random.uniform(0.70, 0.95, num_records),
        'Customer_Satisfaction_Score': np.random.randint(1, 5, num_records),
        
        'Lead_Time_Days': np.random.randint(2, 20, num_records),
        'Efficiency_Rate': np.random.uniform(0.75, 0.95, num_records),
        'Raw_Material_Cost': np.random.uniform(1, 5, num_records),
        'Ad_Spend': np.random.randint(500, 5000, num_records),
        
        'Units_Produced': np.random.randint(100, 1000, num_records),
        'Units_Sold': np.random.randint(50, 900, num_records),
    }
    
    df = pd.DataFrame(data)
    df['Revenue'] = df['Units_Sold'] * np.random.uniform(5, 20, num_records)
    df['Cost'] = (df['Units_Produced'] * df['Raw_Material_Cost']) + df['Ad_Spend']
    df['Profit'] = df['Revenue'] - df['Cost']
    
    return df

# Generate 1000 records
df = generate_data(1000)
# Save to CSV
df.to_csv('hygiene_products_kpis.csv', index=False)
print("دیتاست با موفقیت ایجاد و در فایل hygiene_products_kpis.csv ذخیره شد.")