# %%
import pandas as pd

data = pd.read_csv('./datasets/mod_03_topic_05_weather_data.csv.gz')

# %%
data.info()
data.head()
data.shape

# %%

data.isna().mean().sort_values(ascending=False)

# %%

data = data[data.columns[data.isna().mean().lt(0.35)]]

#Remove rows from the dataset where 'RainTomorrow' column has NaN values
data = data.dropna(subset=['RainTomorrow'])

# %%
import numpy as np

data_num = data.select_dtypes(include=np.number)
data_cat = data.select_dtypes(include='object')

# %%

data_cat['Date'] = pd.to_datetime(data_cat['Date'])
data_cat[['Year', 'Month']] = data_cat['Date'].apply(
    lambda x: pd.Series([x.year, x.month]))

data_num['Year'] = data_cat['Year']
data_cat['Month'] = data_cat['Month'].astype(str)

data_cat.drop('Date', axis=1, inplace=True)
data_cat.drop('Year', axis=1, inplace=True)
    
# %%

max_year = data_num['Year'].max()

train_index = data_num['Year'] < max_year
test_index = data_num['Year'] == max_year

X_train_num = data_num[train_index].drop(columns='Year')
X_train_cat = data_cat[train_index]
y_train = data['RainTomorrow'][train_index]

X_test_num = data_num[test_index].drop(columns='Year')
X_test_cat = data_cat[test_index]
y_test = data['RainTomorrow'][test_index]

# %%
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer().set_output(transform='pandas')

X_train_num = num_imputer.fit_transform(X_train_num)
X_test_num = num_imputer.transform(X_test_num)

pd.concat([X_train_num, X_test_num]).isna().sum()

# %%
cat_imputer = SimpleImputer(strategy='most_frequent').set_output(transform='pandas')

X_train_cat = cat_imputer.fit_transform(X_train_cat)
X_test_cat = cat_imputer.transform(X_test_cat)

pd.concat([X_train_cat, X_test_cat]).isna().sum()

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().set_output(transform='pandas')

X_train_num = scaler.fit_transform(X_train_num)
X_test_num = scaler.transform(X_test_num)

# %%
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    drop='if_binary', 
    sparse_output=False).set_output(transform='pandas')

X_train_cat = encoder.fit_transform(X_train_cat)
X_test_cat = encoder.transform(X_test_cat)

# %%

X_train = pd.concat([X_train_num, X_train_cat], axis=1)
X_test = pd.concat([X_test_num, X_test_cat], axis=1)

# %%

y_train.value_counts(normalize=True)
y_train.isna().sum()

# %%
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(
    class_weight='balanced',
    solver='liblinear',
    random_state=42)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# %%
from sklearn.metrics import classification_report

report_dict = classification_report(y_test, y_pred, output_dict=True)

# Dictionary to DataFrame
report_df = pd.DataFrame(report_dict).transpose()

print(report_df)
