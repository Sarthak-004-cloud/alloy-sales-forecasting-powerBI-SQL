
# Project 2: Python Module for Alloy Sales Forecasting and Analysis
# This notebook performs EDA and business-driven insights
# from `alloy_analysis_view.csv`, exported from SQL

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('alloy_analysis_view.csv')

#  Basic Checks
df.info()
df.head()
df.describe()

# Date Cleanup and Feature Engineering
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
df['Month'] = df['Date'].dt.to_period('M').dt.to_timestamp()
df['Year'] = df['Date'].dt.year

# Create Revenue Column
df['Revenue'] = df['Quantity_kg'] * df['UnitPrice_INR']


# Sector-wise Margin & Quantity Analysis
sector_summary = df.groupby('Sector').agg({
    'Quantity_kg': 'sum',
    'TotalMargin': 'sum',
    'UnitPrice_INR': 'mean',
    'Margin_INR_per_kg': 'mean'
}).sort_values('TotalMargin', ascending=False)

print("\n Sector-wise Summary:")
print(sector_summary)


# Insight
top_sector = sector_summary['TotalMargin'].idxmax()
print(f" Highest total margin comes from: {top_sector}")

 # Plot Sector vs Total Margin
plt.figure(figsize=(10,6))
sns.barplot(data=sector_summary.reset_index(), x='Sector', y='TotalMargin', palette='viridis')
plt.title('Total Margin by Sector')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Monthly Revenue Trend
monthly = df.groupby('Month').agg({
    'Revenue': 'sum',
    'TotalMargin': 'sum',
    'Quantity_kg': 'sum'
}).reset_index()

#  Insight
worst_month = monthly.sort_values('Revenue').iloc[0]['Month']
print(f" Lowest revenue month in dataset: {worst_month}")

plt.figure(figsize=(12,5))
sns.lineplot(data=monthly, x='Month', y='Revenue', marker='o')
plt.title('Monthly Revenue Trend')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Region-Wise Revenue Summary
region_summary = df.groupby('Region').agg({
    'Revenue': 'sum',
    'TotalMargin': 'sum',
    'Quantity_kg': 'sum'
}).sort_values('Revenue', ascending=False)

print("\n Region-wise Revenue Summary:")
print(region_summary)

plt.figure(figsize=(10,5))
sns.barplot(data=region_summary.reset_index(), x='Region', y='Revenue', palette='magma')
plt.title('Revenue by Region')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#Industry Cycle Impact
cycle_impact = df.groupby('IndustryCycle').agg({
    'Revenue': 'mean',
    'TotalMargin': 'mean'
}).sort_values('Revenue', ascending=False)

print("\n Revenue & Margin by Industry Cycle:")
print(cycle_impact)

plt.figure(figsize=(6,4))
sns.heatmap(cycle_impact, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Revenue & Margin by Industry Cycle')
plt.tight_layout()
plt.show()


# Price Elasticity Check (Correlation)
print("\n Price–Quantity Correlation by Sector:")
for sector in df['Sector'].unique():
    sub = df[df['Sector'] == sector]
    corr = sub['UnitPrice_INR'].corr(sub['Quantity_kg'])
    print(f"Sector: {sector:<15} → Correlation: {corr:.2f}")

# Profit Efficiency (Margin/Revenue Ratio)
sector_summary['Margin_Per_Revenue'] = sector_summary['TotalMargin'] / (sector_summary['Quantity_kg'] * sector_summary['UnitPrice_INR'])
sector_summary['Margin_Per_Revenue'] = sector_summary['Margin_Per_Revenue'].round(2)

print("\n Profit Efficiency by Sector:")
print(sector_summary[['Margin_Per_Revenue']])

# Export to CSV
monthly.to_csv('monthly_summary.csv', index=False)
sector_summary.to_csv('sector_summary.csv')
region_summary.to_csv('region_summary.csv')

print("\n Exported all summaries successfully.")

#Month-over-Month Revenue Growth
monthly['Revenue_Growth_%'] = monthly['Revenue'].pct_change() * 100
monthly['Revenue_Growth_%'] = monthly['Revenue_Growth_%'].round(2)
print("\n Month-over-Month Revenue Growth:")
print(monthly[['Month', 'Revenue', 'Revenue_Growth_%']].tail())

# Margin Trend by Sector
margin_trend = df.groupby(['Month', 'Sector'])['TotalMargin'].sum().reset_index()
plt.figure(figsize=(12,6))
sns.lineplot(data=margin_trend, x='Month', y='TotalMargin', hue='Sector', marker='o')
plt.title('Margin Trend by Sector')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Outlier Detection in Margin
Q1 = df['Margin_INR_per_kg'].quantile(0.25)
Q3 = df['Margin_INR_per_kg'].quantile(0.75)
IQR = Q3 - Q1
df['OutlierMarginFlag'] = ((df['Margin_INR_per_kg'] < (Q1 - 1.5 * IQR)) | (df['Margin_INR_per_kg'] > (Q3 + 1.5 * IQR)))
print("\n Outlier Margin Flag Counts:")
print(df['OutlierMarginFlag'].value_counts())


# Supplier Performance Analysis
supplier_summary = df.groupby('Supplier').agg({
    'Revenue': 'sum',
    'TotalMargin': 'sum',
    'Quantity_kg': 'sum',
    'BaseCost_INR': 'mean'
})
supplier_summary['Margin_Per_Kg'] = supplier_summary['TotalMargin'] / supplier_summary['Quantity_kg']
print("\n Supplier-wise Performance Summary:")
print(supplier_summary.round(2))

# Supplier Share by Alloy Type
supplier_alloy = df.groupby(['Supplier', 'Alloy'])['Quantity_kg'].sum().unstack().fillna(0)
print("\n Supplier Share by Alloy:")
print(supplier_alloy.head())

# Export to CSV
monthly.to_csv('monthly_summary.csv', index=False)
sector_summary.to_csv('sector_summary.csv')
region_summary.to_csv('region_summary.csv')
supplier_summary.to_csv('supplier_summary.csv')

print("\n Exported all summaries successfully.")





