#!/usr/bin/env python
# coding: utf-8

# In[101]:


import pandas as pd

# loading the dataset and storing it in a variable - df
# limiting the dataset entries to 100,000 records  - nrows set at 100000

df = pd.read_csv(r'C:\Users\vince\Downloads\accident-dataset\dft-road-casualty-statistics-accident-last-5-years.csv',
                 nrows=100000)
df


# In[102]:


print(df.shape) # getting the number of rows and columns present in the dataset (rows, columns)


# In[106]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Visualising the full data using heatmap
#figsize represents the dimensional size of the figure
#cmap represents the form and colour mapping

plt.figure(figsize=(7,7))
sns.heatmap(df.corr(), cmap='YlGnBu')
plt.show()


# In[107]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#dropping several columns and assigning new set of columns to a variable - dff
dff = df.drop(columns=['accident_index','accident_reference','location_easting_osgr','location_northing_osgr',
                     'police_force','number_of_vehicles','number_of_casualties',
                     'local_authority_district','local_authority_ons_district','local_authority_highway',
                     'first_road_class','first_road_number','second_road_class','second_road_number',
                     'special_conditions_at_site','carriageway_hazards','urban_or_rural_area',
                     'did_police_officer_attend_scene_of_accident','trunk_road_flag','lsoa_of_accident_location',
                     'day_of_week','time','speed_limit','pedestrian_crossing_human_control'])


# In[108]:


#Visualising the new reformed data(dff) prior to dropping several columns data using heatmap
#figsize represents the dimensional size of the figure
#cmap represents the form and colour mapping
#annotation is set to true to display values on each respective blocks

plt.figure(figsize=(10,10))
sns.heatmap(dff.corr(),annot=True, cmap='YlGnBu')
plt.show()


# In[34]:


df.info() #represents the data information. Involves the title of the columns, the non-null count and data type


# In[35]:


# fill in the data records with blank/empty cells

df.fillna(df.min(), inplace=True)
df


# In[47]:


from sklearn.preprocessing import LabelEncoder

le  =  LabelEncoder() # Storing the label encoder function in a variable - le

# all data records and entries are required to have manchine readable values
# le.fit_transform() transforms data columns of type 'object' into column values of type - int

df['accident_index'] = le.fit_transform(df['accident_index']) 
df['accident_reference'] = le.fit_transform(df['accident_reference'])
df['date'] = le.fit_transform(df['date'])
df['time'] = le.fit_transform(df['time'])
df['local_authority_ons_district'] = le.fit_transform(df['local_authority_ons_district'])
df['local_authority_highway'] = le.fit_transform(df['local_authority_highway'])
df['lsoa_of_accident_location'] = le.fit_transform(df['lsoa_of_accident_location'])


# In[8]:


df.info() # confirming data type transformation and blank filling procedure


# In[48]:


# The describe() method returns description of the data in the DataFrame.(the minimum value(min), standard deviation(std),...
# the number of not-empty values(count), The average (mean) value, 25% - The 25% percentile*, max - the maximum value, etc.)

df.describe()


# In[49]:


df


# In[50]:


# Assigning and initializing the column values present in the df dataset to variable X
# this represents the INDEPENDENT variable to be utilised in this research process

X = df.drop(columns=['accident_index','accident_reference','location_easting_osgr','location_northing_osgr',
                     'police_force','accident_severity','number_of_vehicles','number_of_casualties',
                     'local_authority_district','local_authority_ons_district','local_authority_highway',
                     'first_road_class','first_road_number','second_road_class','second_road_number',
                     'special_conditions_at_site','carriageway_hazards','urban_or_rural_area',
                     'did_police_officer_attend_scene_of_accident','trunk_road_flag','lsoa_of_accident_location',
                     'day_of_week','time','speed_limit','pedestrian_crossing_human_control'])
X


# In[51]:


X.shape # getting the number of rows and columns present in X (rows, columns)


# In[41]:


df.shape


# In[42]:


df


# In[52]:


# Assigning and initializing the 'accident_severity' column present in the df dataset to variable Y
# this represents the DEPENDENT variable to be utilised in this research process
y = df['accident_severity']
y


# In[103]:


y.shape # getting the number of rows and columns present in y (rows, columns)


# In[54]:


from sklearn.preprocessing import MinMaxScaler

# MinMaxScaler() is a preprocessing step to improve the performance of the algorithm
scalar = MinMaxScaler()
X_scaler = scalar.fit_transform(X)


# In[55]:


from sklearn.model_selection import train_test_split

# The train_test_split() method is used to split our data into train and test sets.
# The dataframe gets divided into X_train, X_test, y_train, and y_test.
# test and training sizes are set to 30% and 70% respectively

X_train, X_test, y_train, y_test =train_test_split(X_scaler, y, test_size=0.3, random_state=0)


# In[127]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create Decision Tree classifer object and storing it in a varibale - model
model = DecisionTreeClassifier(max_depth=5)

#X_train and y_train sets are used for training and fitting the model
model.fit(X_train, y_train)

#predict() function enables us to predict the labels of the data values on the basis of the trained model.
#this prediction is stored in a variable - pred
pred = model.predict(X_test)

#accuracy_score(y_test, pred) counts all the indexes where an element of y_test equals to an element of pred...
# and then divide it with the total number of elements in the list
print('accuracy_score: ' , accuracy_score(y_test, pred))


# In[128]:


cnf_matrix = metrics.confusion_matrix(y_test, pred)
cnf_matrix


# In[129]:


mat = metrics.confusion_matrix(y_test, pred)
sns.heatmap(mat, annot=True, fmt="d")


# In[124]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Create GaussianNB object and storing it in a varibale - model
model = GaussianNB()

#X_train and y_train sets are used for training and fitting the model
model.fit(X_train, y_train)

#predict() function enables us to predict the labels of the data values on the basis of the trained model.
#this prediction is stored in a variable - pred
pred = model.predict(X_test)

#accuracy_score(y_test, pred) counts all the indexes where an element of y_test equals to an element of pred...
# and then divide it with the total number of elements in the list
print('accuracy_score: ' , accuracy_score(y_test, pred))


# In[125]:


cnf_matrix = metrics.confusion_matrix(y_test, pred)
cnf_matrix


# In[126]:


mat = metrics.confusion_matrix(y_test, pred)
sns.heatmap(mat, annot=True, fmt="d")


# In[121]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Create KNeighborsClassifier object and storing it in a varibale - model
model = KNeighborsClassifier()

#X_train and y_train sets are used for training and fitting the model
model.fit(X_train, y_train)

#predict() function enables us to predict the labels of the data values on the basis of the trained model.
#this prediction is stored in a variable - pred
pred = model.predict(X_test)

#accuracy_score(y_test, pred) counts all the indexes where an element of y_test equals to an element of pred...
# and then divide it with the total number of elements in the list
print('accuracy_score: ' , accuracy_score(y_test, pred))


# In[122]:


cnf_matrix = metrics.confusion_matrix(y_test, pred)
cnf_matrix


# In[123]:


mat = metrics.confusion_matrix(y_test, pred)
sns.heatmap(mat, annot=True, fmt="d")


# In[130]:


from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn import metrics

# Create LinearSVC object and storing it in a varibale - model
model = LinearSVC()

#X_train and y_train sets are used for training and fitting the model
model.fit(X_train, y_train)

#predict() function enables us to predict the labels of the data values on the basis of the trained model.
#this prediction is stored in a variable - pred
pred = model.predict(X_test)

#accuracy_score(y_test, pred) counts all the indexes where an element of y_test equals to an element of pred...
# and then divide it with the total number of elements in the list
print('accuracy_score: ' , accuracy_score(y_test, pred))


# In[131]:


from sklearn import metrics

#predict() function enables us to predict the labels of the data values on the basis of the trained model.
#this prediction is stored in a variable - pred
cnf_matrix = metrics.confusion_matrix(y_test, pred)
cnf_matrix


# In[132]:


mat = metrics.confusion_matrix(y_test, pred)
sns.heatmap(mat, annot=True, fmt="d")


# In[88]:


#histogram representation of some columns
df.hist(column = ['accident_year', 'longitude', 'latitude', 'accident_severity', 'number_of_vehicles',
                  'number_of_casualties', 'road_type','weather_conditions'], figsize=(20, 10))


# In[90]:


#histogram representation of all columns
# figsize represents the size of the figure
df.hist(figsize=(20, 20))

