#all imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

#Load data and allow the display of all columns
desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 12)
data = pd.read_csv("walmart.csv", parse_dates=[0], infer_datetime_format=True)


#Make sure that there are no empty values
print(data.isnull().sum())


#Categorize store number and holiday flag so they are not considered as integers
data["Holiday_Flag"] = data["Holiday_Flag"].astype('category')
data["Store"] = data["Store"].astype('category')


#Separate date to use in sale trend
data['Date'] = pd.to_datetime(data['Date'])
data['Year']=data['Date'].dt.year
data['Month']=data['Date'].dt.month
data['Day']=data['Date'].dt.day
data['MonthYear_value'] = data['Date'].dt.to_period('M')    

#Print data
print(data)

#Visualize quantitive data 
q = ['Temperature', 'CPI', 'Fuel_Price', 'Unemployment']
for i in range(len(q)):
    sns.displot(data[q[i]], kde=True)
    
plt.tight_layout()
plt.show()
    


#Find which store has highest sales

sns.color_palette("crest", as_cmap=True)
fig,ax=plt.subplots(figsize=(12,8))
sns.barplot(x='Store', y='Weekly_Sales',data=data,palette = "crest")

top_sales = data.groupby("Store")["Weekly_Sales"].sum().reset_index(name="Total Sales")
top_sales = top_sales.sort_values(by="Total Sales", ascending = False , ignore_index = True)
print(top_sales.head(5))


#Find which store has the highest standard deviation
max_deviation = data.groupby("Store")["Weekly_Sales"].std().reset_index(name="Sales_Variance")
print(max_deviation.sort_values(by="Sales_Variance", ascending = False, ignore_index = True).head(5))



#Find trends over the year specifically holidays
sales_year = data.groupby(['MonthYear_value']).sum()
sales_year[['Weekly_Sales']].plot(figsize=(16, 6), subplots=True)


#Monthly Sales

plt.figure(figsize=(14,7), dpi=80)
plt.bar(data['Month'], data['Weekly_Sales'])
plt.xlabel('Months')
plt.ylabel('Weekly_Sales')
plt.title('Monthwise Sales')


#Unemployment rate per store
sns.color_palette("dark")
fig,ax=plt.subplots(figsize=(12,8))
sns.barplot(data=data, x= 'Store', y='Unemployment', palette = "mako")

#Temperature and sales
fig,ax=plt.subplots(figsize=(12,8))
sns.scatterplot(data=data, x=data['Temperature'], y=data['Weekly_Sales'])


#Holiday Season effect
sns.barplot(data=data, x=data['Holiday_Flag'], y=data['Weekly_Sales'])








