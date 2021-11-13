#!/usr/bin/env python
# coding: utf-8

# # GRIP JUNE @THE SPARK FOUNDATION(COMPANY)

# # DATA SCIENCE AND BUSINESS ANALYTICS

# # AUTHOR : PAVAN RS

# # TASK 1: PREDICTION USING SUPERVISED ML

# OBJECTIVE: TO DETERMINE THE PERCENTAGE OF STUDENT BASED ON THE HOURS OF STUDY

# # Importing all required Libraries and Data Set

# In[8]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')

url = "http://bit.ly/w-data"
df = pd.read_csv(url)


# In[9]:


df


# # Checking for Null Values

# In[10]:


df.isnull().sum()


# # Visualizing the Data Set

# In[11]:


#ploting the above data
df.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# By looking at the graph we can conclude that there is a positive linear relation the number of hours studied and percentage of score.

# # Preparing the Data 

# In[12]:


#Splitting the data attributes into input and output
X = df.iloc[:, :-1].values  
y = df.iloc[:, 1].values  


# In[13]:


#here we will import train_test_split method from sklearn and split the DataSet into train and test DataSets
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0) 


# # Training the Algorithm

# In[17]:


#importing the regression model
from sklearn.linear_model import LinearRegression  
lg = LinearRegression()  
lg.fit(X_train, y_train) 


# In[18]:


lg.intercept_


# In[19]:


lg.coef_


# # Plotting the regression line

# In[15]:


line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# # Making Predictions

# In[20]:


print(X_test)

#prediction the scores
y_pred = lg.predict(X_test) 


# In[21]:


#predicted output
y_pred


# #Comparing Actual vs Predicted

# In[23]:


comp_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
comp_df 


# # Predicting the score of student based on hours studied

# As per our question here our aim is to predict the score of student if he/she studied for 9.25 

# In[30]:


hours = 9.25
own_pred = lg.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# #Accuracy of the model

# In[33]:


np.round(lg.score(X_test,y_test)*100,2)


# # Evaluating the model

# In[37]:


#Evaluating using Mean Absolute Error
from sklearn import metrics  
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred)) 
 
#Evalluating using R square Error
print("r2 score:",metrics.r2_score(y_test,y_pred))




