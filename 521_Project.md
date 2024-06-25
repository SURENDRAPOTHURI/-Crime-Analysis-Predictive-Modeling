## Overview of the Project

Possible use of the Philadelphia Crime Data dataset is to develop a machine learning model that can predict the type of offense based on the latitude and longitude of the incident, as well as the date, year, and month when the incident occurred.

To do this, we could use supervised learning algorithms, such as logistic regression, decision trees, or random forests.






## Identify and describe the data you used

The Data is collection of crime data for the city of Philadelphia, Pennsylvania, USA. The data spans from January 2006 to June 2017 and includes over 2.2 million rows.

The data includes information on various types of crimes committed in Philadelphia, such as assault, burglary, homicide, theft, and vandalism. Each row in the dataset represents a reported crime incident and includes information such as the date and time of the incident, the type of crime, the location of the incident, and the outcome of the incident (e.g., an arrest was made or not).

Additionally, it includes information on the police district, sector, and beat where the incident occurred.

The purpose of this dataset is to provide researchers, analysts, and interested parties with access to a comprehensive set of crime data for the city of Philadelphia. This data can be used to perform various analyses, including crime pattern detection, trend analysis, and forecasting.

#### Columns

Dc_Dist             : The police district in which the incident occurred.


Psa                 : The police service area in which the incident occurred.


Dispatch_Date_Time  : The date and time that the incident was dispatched to police officers.


Dispatch_Date       : The date that the incident was dispatched to police officers.


Dispatch_Time       : The time that the incident was dispatched to police officers.


Hour                : The hour of the day that the incident occurred (in 24-hour format).


Dc_Key              : A unique identifier for the incident.


Location_Block      : The block address where the incident occurred.


UCR_General         : The Uniform Crime Reporting (UCR) code for the general category of the incident.


Text_General_Code   : A text description of the UCR code for the incident.


Police_Districts    : A mapping of the police districts to larger geographic areas.


Month               : The month of the year that the incident occurred.


Lon                 : The longitude of the location of the incident.


Lat                 : The latitude of the location of the incident.

## Intention

Once the model is trained and validated, we could use it to make predictions on new data by inputting the year, month, day, latitude, and longitude of a location, and getting the predicted type of offense in return. 

This could help law enforcement agencies to allocate resources more effectively, such as by increasing police patrols in areas with high risk of crime, and help individuals to take precautions and avoid dangerous situations.

## Analytic Approach

### 1. Data Preprocessing

##### 1. Understanding Data Set and Shape of the dataset


df = pd.read_csv('data/crime.csv')

pd.set_option('display.max_columns',None)

df.head()

df.shape



#####  2. Checking Data Types of the Attributes 

df.info()

#####  3. Exploring Categorical Attributes 

cat_feature = [feature for feature in df.columns if df[feature].dtype == 'object']

print("Number of Categorical Features are : ",len(cat_feature))

print(cat_feature)

df[cat_feature][:5]

##### 4. Exploring Numerical Attributes 

num_feature = [feature for feature in df.columns if df[feature].dtype != 'object']

print("Number of Numerical Features are : ",len(num_feature))

print(num_feature)

df[num_feature][:3]

##### 5. Statistical Summary of the dataset - Descriptive Statistics

df.describe()

##### 6. Checking Missing Values (NaN) in Dataset 

df.isnull().sum()

df = df.dropna()

df.isnull().sum()

df['Dispatch_Date_Time'] = pd.to_datetime(df['Dispatch_Date_Time'])

df = df.sort_values(by='Dispatch_Date_Time', ascending=True)

df['Crime_Year'] = df['Dispatch_Date_Time'].dt.year

df['Crime_Month'] = df['Dispatch_Date_Time'].dt.month

df['Crime_Day'] = df['Dispatch_Date_Time'].dt.day

df.info()

df.head()

feature_nan = [feature for feature in df.columns if df[feature].isnull().sum()>1]

for feature in feature_nan:

    print('{} : {} % Missing values'.format(feature,np.around(df[feature].isnull().mean(),4)))

##### 7. Checking the Distribution of Dist (Target Variable)

df['Dc_Dist'].value_counts(normalize = True)

### 2. EDA

1. To visualize numbers of incidents happen based on month, created bargraph using Crime_Month column. 

2. To visualize numbers of incidents happen based on year, created bargraph using Crime_Year column. 

3. To visualize numbers of incidents happen based on Hour, created bargraph using Hour column. 

4. To visualize numbers of incidents happen based on Type of crime, created bargraph using Text_General_Code column.

### 3. Feature Engineering

<b> Selected only first 13 types for machine learning model prediction. </b>

filtered_df = df[df['Text_General_Code'].isin(['All Other Offenses', 'Other Assaults', 'Thefts', 'Vandalism/Criminal Mischief', 'Theft from Vehicle','Narcotic / Drug Law Violations', 'Fraud','Recovered Stolen Motor Vehicle','Burglary Residential','Aggravated Assault No Firearm','DRIVING UNDER THE INFLUENCE','Robbery No Firearm','Motor Vehicle Theft', ])]

filtered_df['Text_General_Code'].unique()

mycol = filtered_df[["Text_General_Code"]]
for i in mycol:
    cleanup_nums = {i: {"All Other Offenses": 0, "Other Assaults": 1, "Thefts": 2, "Vandalism/Criminal Mischief":3, "Theft from Vehicle":4,"Narcotic / Drug Law Violations":5,"Fraud":6, "Recovered Stolen Motor Vehicle":7,"Burglary Residential":8, "Aggravated Assault No Firearm":9,"DRIVING UNDER THE INFLUENCE":10, "Robbery No Firearm":11, "Motor Vehicle Theft":12}}
    df1 = filtered_df.replace(cleanup_nums)

### 4. Machine Learning Model

feed = df1[['Lon', 'Lat', 'Crime_Year', 'Crime_Month', 
           'Crime_Day', 'Text_General_Code']]

##### Taking all independent variable columns
df_train_x = feed.drop('Text_General_Code',axis = 1)

###### Target variable column
df_train_y = feed['Text_General_Code']

x_train, x_test, y_train, y_test = train_test_split(df_train_x, df_train_y, test_size=0.20, random_state=42)

##### LogisticRegression

from sklearn.preprocessing import StandardScaler

s=StandardScaler()

s.fit(x_train)

x_train=s.transform(x_train)

x_test=s.transform(x_test)






lg = LogisticRegression()

lg.fit(x_train, y_train)

predictions_lg = lg.predict(x_test)

lg.score(x_test,y_test)

##### DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier

tree=DecisionTreeClassifier()

tree.fit(x_train,y_train)

tree.score(x_test,y_test)


##### RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

r=RandomForestClassifier()

r.fit(x_train,y_train)

r.score(x_test,y_test)

## Limitations

The project is constrained by a limited number of features available to use for predicting machine learning models. This means that the models may not be able to fully capture the complexity of the underlying data and may therefore have reduced accuracy or predictive power.

The data is not evenly distributed, it can cause problems when training machine learning models.
 
The success of any machine learning model is highly dependent on the quality and quantity of features used in training the model, and with fewer features available, it may be difficult to achieve optimal performance.


```python

```
