
import pandas as pd
import numpy as np
import socket
import struct
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

fraud_data = pd.read_csv('./data/Fraud_Data.csv')
ip_address_data = pd.read_csv('./data/IpAddress_to_Country.csv')

missing_values = fraud_data.isnull().sum()
print("Missing Values:\n", missing_values)

fraud_data_cleaned = fraud_data.dropna()

# Remove duplicate rows
fraud_data_cleaned.drop_duplicates(inplace=True)
# Convert data types as necessary
fraud_data_cleaned['purchase_time'] = pd.to_datetime(fraud_data_cleaned['purchase_time'])
fraud_data_cleaned['ip_address'] = fraud_data_cleaned['ip_address'].astype(str)

# Univariate Analysis - Histogram of purchase values
plt.figure(figsize=(10, 6))
sns.histplot(fraud_data_cleaned['purchase_value'], bins=30, kde=True)
plt.title('Distribution of Purchase Values')
plt.xlabel('Purchase Value')
plt.ylabel('Frequency')
plt.show()
# Bivariate Analysis - Purchase value vs. Fraud Class
plt.figure(figsize=(10, 6))
sns.boxplot(x='class', y='purchase_value', data=fraud_data_cleaned)
plt.title('Purchase Value by Fraud Class')
plt.xlabel('Fraud Class')
plt.ylabel('Purchase Value')
plt.show()

import pandas as pd
import socket
import struct
import numpy as np

# Function to check if an IP address is valid
def is_valid_ip(ip):
    try:
        socket.inet_aton(ip)
        return True
    except socket.error:
        return False

# Convert IP Addresses to Integer Format
def ip_to_int(ip):
    if is_valid_ip(ip):  # Check if the IP is valid
        return struct.unpack("!I", socket.inet_aton(ip))[0]
    else:
        return np.nan  # Return NaN for invalid IP addresses

# Assume fraud_data_cleaned is already defined and contains an 'ip_address' column
fraud_data_cleaned['ip_as_int'] = fraud_data_cleaned['ip_address'].apply(ip_to_int)

# Check for NaN values in the new 'ip_as_int' column
invalid_ips = fraud_data_cleaned[fraud_data_cleaned['ip_as_int'].isna()]

# Debugging: Print columns and a sample of the DataFrame
print("Columns in fraud_data_cleaned:", fraud_data_cleaned.columns.tolist())
print("Sample of fraud_data_cleaned:")
print(fraud_data_cleaned.head())

# Debugging: Check the columns in ip_address_data
print("Columns in ip_address_data:", ip_address_data.columns.tolist())
print("Sample of ip_address_data:")
print(ip_address_data.head())

# Check if 'ip_as_int' exists in both DataFrames
if 'ip_as_int' not in fraud_data_cleaned.columns:
    print("Error: 'ip_as_int' column not found in fraud_data_cleaned.")
else:
    if 'ip_as_int' not in ip_address_data.columns:
        print("Error: 'ip_as_int' column not found in ip_address_data.")
    else:
        # Merge with IpAddress_to_Country dataset
        merged_data = fraud_data_cleaned.merge(ip_address_data, on='ip_as_int', how='left')
        print("Merge successful! Sample of merged data:")
        print(merged_data.head())

# Print invalid IPs for debugging if needed
if not invalid_ips.empty:
    print("Invalid IP Addresses:")
    print(invalid_ips[['ip_address']])

# Feature Engineering
# Assuming you have a 'user_id' column
transaction_frequency = fraud_data_cleaned.groupby('user_id').size().reset_index(name='transaction_frequency')

# Merge frequency data back into the main dataset
fraud_data_cleaned = fraud_data_cleaned.merge(transaction_frequency, on='user_id', how='left')

# Calculate transaction velocity
fraud_data_cleaned['transaction_velocity'] = fraud_data_cleaned['transaction_frequency'] / ((fraud_data_cleaned['purchase_time'] - fraud_data_cleaned['purchase_time'].min()).dt.days + 1)

# Time-Based Features
fraud_data_cleaned['hour_of_day'] = fraud_data_cleaned['purchase_time'].dt.hour
fraud_data_cleaned['day_of_week'] = fraud_data_cleaned['purchase_time'].dt.dayofweek

# Normalization and Scaling
scaler = StandardScaler()
fraud_data_cleaned[['purchase_value', 'transaction_velocity']] = scaler.fit_transform(fraud_data_cleaned[['purchase_value', 'transaction_velocity']])

# Encode Categorical Features
fraud_data_encoded = pd.get_dummies(fraud_data_cleaned, columns=['source', 'browser', 'sex', 'day_of_week'], drop_first=True)

# Save the cleaned data
fraud_data_encoded.to_csv('./data/Cleaned_Fraud_Data.csv', index=False)