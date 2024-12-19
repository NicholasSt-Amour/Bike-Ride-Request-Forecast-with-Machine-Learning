import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sb 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn import metrics 
from sklearn.svm import SVC 
from xgboost import XGBRegressor 
from sklearn.linear_model import LinearRegression, Lasso, Ridge 
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
  
import warnings 
warnings.filterwarnings('ignore')

df = pd.read_csv('train.csv') 
#print(df.head()) # displays the first five rows of the data set
#print(df.shape) # displays a tuple with the number of rows and columns
#df.info() #displays key info about the data set, like the data type of each column 

for i in df.columns:
    df = df.rename(columns = {i:i.replace("+AF8-","_")}) #cleaning the dataframe

#print(df.head()) #rechecking the modified data frame

print(df.columns) #checking the status of the columns after the modification
print(df['pickup_time'])

#making the times more readable by first converting to datetime type and then splitting each
#component into year, month, day, hour, minute, day_of_week

df['pickup_time'] = pd.to_datetime(df['pickup_time'])
df['drop_time'] = pd.to_datetime(df['pickup_time'])

df['pickup_year'] = df['pickup_time'].dt.year
df['pickup_month'] = df['pickup_time'].dt.month    # Extract month
df['pickup_day'] = df['pickup_time'].dt.day        # Extract day
df['pickup_hour'] = df['pickup_time'].dt.hour      # Extract hour
df['pickup_minute'] = df['pickup_time'].dt.minute  # Extract minute
df['pickup_day_of_week'] = df['pickup_time'].dt.dayofweek  # Extract day of the week (Mon=0, Sun=6)

df['drop_year'] = df['drop_time'].dt.year
df['drop_month'] = df['drop_time'].dt.month    # Extract month
df['drop_day'] = df['drop_time'].dt.day        # Extract day
df['drop_hour'] = df['drop_time'].dt.hour      # Extract hour
df['drop_minute'] = df['drop_time'].dt.minute  # Extract minute
df['drop_day_of_week'] = df['drop_time'].dt.dayofweek  # Extract day of the week (Mon=0, Sun=6)

print(df.columns)

# next step: check which features should be discarded
# df_select = df.loc[...]
