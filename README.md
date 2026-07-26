# Ex.No: 1B   CONVERSION OF NON STATIONARY TO STATIONARY DATA                

# Date: 26/7/2026

### AIM:
To perform regular differncing,seasonal adjustment and log transformatio on international airline passenger data
### ALGORITHM:
1. Import the required packages like pandas and numpy
2. Read the data using the pandas
3. Perform the data preprocessing if needed and apply regular differncing,seasonal adjustment,log transformation.
4. Plot the data according to need, before and after regular differncing,seasonal adjustment,log transformation.
5. Display the overall results.
### PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from statsmodels.tsa.seasonal import seasonal_decompose

data = pd.read_csv('/content/Superstore.csv', encoding='latin1')

data['Order Date'] = pd.to_datetime(data['Order Date'], dayfirst=True)

# Aggregate 'Profit' by 'Order Date' to ensure a unique time series index
ts_data = data.groupby('Order Date')['Profit'].sum().to_frame()

# Filter for positive profits
ts_data_positive_profit = ts_data[ts_data['Profit'] > 0].copy()

ts_data_positive_profit['passengers_diff'] = ts_data_positive_profit['Profit'] - ts_data_positive_profit['Profit'].shift(1)

result_profit = seasonal_decompose(ts_data_positive_profit['Profit'], model='additive', period=12)
ts_data_positive_profit['passengers_sea_diff'] = result_profit.resid

ts_data_positive_profit['passengers_log'] = np.log(ts_data_positive_profit['Profit'])
ts_data_positive_profit['passengers_log_diff'] = ts_data_positive_profit['passengers_log'] - ts_data_positive_profit['passengers_log'].shift(1)

result_log_diff = seasonal_decompose(
    ts_data_positive_profit['passengers_log_diff'].dropna(),
    model='additive',
    period=12
)
ts_data_positive_profit['passengers_log_seasonal_diff'] = result_log_diff.resid

plt.figure(figsize=(16, 16))

plt.subplot(6, 1, 1)
plt.plot(ts_data_positive_profit['Profit'], label='Original')
plt.legend(loc='best')
plt.title('Original Data')
plt.xlabel('Year')
plt.ylabel('No of passengers')

plt.subplot(6, 1, 2)
plt.plot(ts_data_positive_profit['passengers_diff'], label='Regular Difference')
plt.legend(loc='best')
plt.title('Regular Differencing')
plt.xlabel('Year')
plt.ylabel('Differenced No of passengers')

plt.subplot(6, 1, 3)
plt.plot(ts_data_positive_profit['passengers_sea_diff'], label='Seasonal Adjustment')
plt.legend(loc='best')
plt.title('Seasonal Adjustment')
plt.xlabel('Year')
plt.ylabel('Seasonally adjusted No of passengers')

plt.subplot(6, 1, 4)
plt.plot(ts_data_positive_profit['passengers_log'], label='Log Transformation')
plt.legend(loc='best')
plt.title('Log Transformation')
plt.xlabel('Year')
plt.ylabel('Log(No of passengers)')

plt.subplot(6, 1, 5)
plt.plot(
    ts_data_positive_profit['passengers_log_diff'],
    label='Log Transformation and Regular Differencing'
)
plt.legend(loc='best')
plt.title('Log Transformation and Regular Differencing')
plt.xlabel('Year')
plt.ylabel('RDiff(Log(No of passengers))')

plt.subplot(6, 1, 6)
plt.plot(
    ts_data_positive_profit['passengers_log_seasonal_diff'],
    label='Log Transformation and Regular Differencing and Seasonal Differencing'
)
plt.legend(loc='best')
plt.title('Log Transformation and Regular Differencing and Seasonal Differencing')
plt.xlabel('Year')
plt.ylabel('SDiff(RDiff(Log(No of passengers)))')

plt.tight_layout()
plt.show()

ts_data_positive_profit.plot(kind='line')


```


### OUTPUT:

<img width="1671" height="271" alt="image" src="https://github.com/user-attachments/assets/d30f2108-4a26-4524-a441-d30a6c913b48" />


REGULAR DIFFERENCING:
<img width="1676" height="263" alt="image" src="https://github.com/user-attachments/assets/94125bc2-3cae-4f3d-acff-88cea12351bc" />


SEASONAL ADJUSTMENT:

<img width="1667" height="291" alt="image" src="https://github.com/user-attachments/assets/950a8ff9-8cc4-4ea9-be41-3862e100cc18" />

LOG TRANSFORMATION:

<img width="1676" height="268" alt="image" src="https://github.com/user-attachments/assets/ff9c01d8-e100-4a97-a8b9-220ced0bba48" />

<img width="1682" height="537" alt="image" src="https://github.com/user-attachments/assets/67466431-45c3-4c0d-933c-3da0e5310afe" />
<img width="641" height="491" alt="image" src="https://github.com/user-attachments/assets/c5c1284d-506f-4840-be8e-3279ce19f67f" />

### RESULT:
Thus we have created the python code for the conversion of non stationary to stationary data on international airline passenger
data.
