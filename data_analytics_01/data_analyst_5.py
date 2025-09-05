# @author: ALI GHANBARI 
# @email: alighanbari446@gmail.com

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Load the dataset
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

# --- بخش اول: تحلیل‌های استاتیک و ویژوال‌سازی (Matplotlib & Seaborn) ---
print("--- تحلیل‌های استاتیک و ویژوال‌سازی ---")

# Set a professional style for plots
sns.set_style("whitegrid")
plt.style.use('fivethirtyeight')

# تحلیل‌های توصیفی (Descriptive Analysis)
kpi_summary = df[['Revenue', 'Profit', 'Acquisition_Rate', 'Efficiency_Rate', 'Lead_Time_Days', 'Customer_Satisfaction_Score']].describe()
print("آمار توصیفی KPIهای اصلی:\n", kpi_summary)

# 1. روند سود و درآمد ماهانه
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



# --- بخش دوم: تحلیل‌های پیشرفته (Advanced Analytics) ---
print("\n--- تحلیل‌های پیشرفته ---")

# A/B Testing
print("\n--- تحلیل A/B Testing ---")
campaign_a = df[df['Campaign'] == 'Summer_Sale']['Acquisition_Rate']
campaign_b = df[df['Campaign'] == 'New_Product_Launch']['Acquisition_Rate']
t_stat, p_value = stats.ttest_ind(campaign_a, campaign_b, equal_var=False, nan_policy='omit')
print(f"مقدار p-value در تست A/B: {p_value:.4f}")
if p_value < 0.05:
    print("نتیجه: تفاوت در نرخ جذب مشتری بین دو کمپین **از نظر آماری معنادار** است.")
else:
    print("نتیجه: تفاوت معناداری بین نرخ جذب مشتری دو کمپین وجود ندارد.")

# خوشه‌بندی مشتریان (Clustering)
print("\n--- تحلیل خوشه‌بندی مشتریان ---")
df_for_clustering = df[['Revenue', 'Retention_Rate', 'Customer_Satisfaction_Score']].fillna(0)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Customer_Cluster'] = kmeans.fit_predict(df_for_clustering)
print("مشتریان به 3 خوشه تقسیم شدند.")

# رگرسیون پیشرفته (Random Forest)
print("\n--- تحلیل رگرسیون پیشرفته ---")
X = df[['Website_Traffic', 'Ad_Spend', 'Raw_Material_Cost', 'Efficiency_Rate']]
y = df['Profit']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
print("مدل Random Forest Regressor برای پیش‌بینی سود آموزش داده شد.")


# --- بخش سوم: داشبورد تعاملی با Plotly و Dash ---
print("\n--- راه‌اندازی داشبورد تعاملی ---")
print("برای مشاهده داشبورد، مرورگر خود را باز کرده و به آدرس http://127.0.0.1:8050/ بروید.")

app = Dash(__name__)

app.layout = html.Div([
    html.H1("داشبورد تحلیل عملکرد شرکت مواد بهداشتی و شوینده 📊", style={'text-align': 'center'}),
    
    html.Div([
        html.Div([
            html.H3("فیلتر بر اساس دسته محصول"),
            dcc.Dropdown(
                id='product-category-dropdown',
                options=[{'label': i, 'value': i} for i in df['Product_Category'].unique()],
                value=df['Product_Category'].unique()[0],
                clearable=False
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3("فیلتر بر اساس منطقه"),
            dcc.Dropdown(
                id='region-dropdown',
                options=[{'label': i, 'value': i} for i in df['Region'].unique()],
                value=df['Region'].unique()[0],
                clearable=False
            ),
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
    ]),
    
    dcc.Graph(id='monthly-profit-graph'),
    dcc.Graph(id='product-kpis-graph'),
    
    html.Div([
        html.H3("تحلیل خوشه بندی مشتریان"),
        dcc.Graph(id='customer-clusters-graph')
    ]),
])

@app.callback(
    [Output('monthly-profit-graph', 'figure'),
     Output('product-kpis-graph', 'figure'),
     Output('customer-clusters-graph', 'figure')],
    [Input('product-category-dropdown', 'value'),
     Input('region-dropdown', 'value')]
)
def update_graphs(selected_category, selected_region):
    filtered_df = df[(df['Product_Category'] == selected_category) & (df['Region'] == selected_region)]
    
    monthly_kpis = filtered_df.groupby('Month').agg({'Revenue': 'sum', 'Profit': 'sum'}).reset_index()
    monthly_kpis['Month'] = monthly_kpis['Month'].dt.to_timestamp()
    fig_profit = px.line(monthly_kpis, x='Month', y=['Revenue', 'Profit'], 
                         title=f'روند ماهانه درآمد و سود برای {selected_category} در منطقه {selected_region}')

    production_kpis = filtered_df.groupby('Month').agg({'Efficiency_Rate': 'mean', 'Lead_Time_Days': 'mean'}).reset_index()
    production_kpis['Month'] = production_kpis['Month'].dt.to_timestamp()
    fig_production = px.line(production_kpis, x='Month', y=['Efficiency_Rate', 'Lead_Time_Days'], 
                              title=f'روند بهره‌وری و Lead Time برای {selected_category} در منطقه {selected_region}')

    customer_clusters_fig = px.scatter_3d(df, x='Revenue', y='Retention_Rate', z='Customer_Satisfaction_Score',
                                         color='Customer_Cluster',
                                         title='خوشه بندی مشتریان بر اساس سودآوری، حفظ و رضایت')
    
    return fig_profit, fig_production, customer_clusters_fig

if __name__ == '__main__':
    app.run_server(debug=True)
# %%
