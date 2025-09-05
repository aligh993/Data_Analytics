# @author: ALI GHANBARI 
# @email: alighanbari446@gmail.com

#%%
import pandas as pd
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

# --- تحلیل‌های پیشرفته ---

# 1. تحلیل A/B Testing
print("--- تحلیل A/B Testing ---")
campaign_a = df[df['Campaign'] == 'Summer_Sale']['Acquisition_Rate']
campaign_b = df[df['Campaign'] == 'New_Product_Launch']['Acquisition_Rate']

# Perform a t-test to check for statistical significance
t_stat, p_value = stats.ttest_ind(campaign_a, campaign_b, equal_var=False, nan_policy='omit')
print(f"مقدار p-value در تست A/B: {p_value:.4f}")
if p_value < 0.05:
    print("نتیجه: تفاوت در نرخ جذب مشتری بین دو کمپین **از نظر آماری معنادار** است.")
else:
    print("نتیجه: تفاوت معناداری بین نرخ جذب مشتری دو کمپین وجود ندارد.")

# 2. تحلیل خوشه‌بندی (Clustering) مشتریان
print("\n--- تحلیل خوشه‌بندی مشتریان ---")
# Select features for clustering
features = df[['Revenue', 'Retention_Rate', 'Customer_Satisfaction_Score']].dropna()
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Customer_Cluster'] = kmeans.fit_predict(df[['Revenue', 'Retention_Rate', 'Customer_Satisfaction_Score']].fillna(0))
print("مشتریان به 3 خوشه تقسیم شدند.")

# 3. تحلیل رگرسیون پیشرفته (پیش‌بینی سود)
print("\n--- تحلیل رگرسیون پیشرفته ---")
# Prepare data for Random Forest Regressor
X = df[['Website_Traffic', 'Ad_Spend', 'Raw_Material_Cost', 'Efficiency_Rate']]
y = df['Profit']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
print("مدل Random Forest Regressor برای پیش‌بینی سود آموزش داده شد.")

# --- ساخت داشبورد تعاملی با Dash ---
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
    # Filter data based on dropdown selections
    filtered_df = df[(df['Product_Category'] == selected_category) & (df['Region'] == selected_region)]
    
    # Graph 1: Monthly Profit and Revenue
    monthly_kpis = filtered_df.groupby('Month').agg({'Revenue': 'sum', 'Profit': 'sum'}).reset_index()
    monthly_kpis['Month'] = monthly_kpis['Month'].dt.to_timestamp()
    fig_profit = px.line(monthly_kpis, x='Month', y=['Revenue', 'Profit'], 
                         title=f'روند ماهانه درآمد و سود برای {selected_category} در منطقه {selected_region}')

    # Graph 2: Efficiency and Lead Time
    production_kpis = filtered_df.groupby('Month').agg({'Efficiency_Rate': 'mean', 'Lead_Time_Days': 'mean'}).reset_index()
    production_kpis['Month'] = production_kpis['Month'].dt.to_timestamp()
    fig_production = px.line(production_kpis, x='Month', y=['Efficiency_Rate', 'Lead_Time_Days'], 
                              title=f'روند بهره‌وری و Lead Time برای {selected_category} در منطقه {selected_region}')

    # Graph 3: Customer Clusters
    customer_clusters_fig = px.scatter_3d(df, x='Revenue', y='Retention_Rate', z='Customer_Satisfaction_Score',
                                         color='Customer_Cluster',
                                         title='خوشه بندی مشتریان بر اساس سودآوری، حفظ و رضایت')
    
    return fig_profit, fig_production, customer_clusters_fig

if __name__ == '__main__':
    app.run_server(debug=True)
# %%
