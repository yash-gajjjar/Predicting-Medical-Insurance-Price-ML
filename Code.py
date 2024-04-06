#!/usr/bin/env python
# coding: utf-8

# In[223]:


## Downloading the Data


# In[224]:


get_ipython().system('pip install pandas-profiling --quiet')


# In[225]:


medical_charges_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'


# In[226]:


from urllib.request import urlretrieve


# In[227]:


urlretrieve(medical_charges_url, 'medical.csv')


# In[228]:


import pandas as pd
medical_df = pd.read_csv('medical.csv')
medical_df


# In[229]:


medical_df.info()


# In[230]:


medical_df.describe()


# In[231]:


get_ipython().system('pip install jovian --quiet')
import jovian
jovian.commit()


# In[232]:


## Exploratory Analysis and Visualization


# In[233]:


get_ipython().system('pip install plotly matplotlib seaborn --quiet')


# In[234]:


import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[235]:


sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# In[236]:


### Age

#Age is a numeric column. The minimum age in the dataset is 18 and the maximum age is 64. Thus, we can visualize the distribution of age using a histogram with 47 bins (one for each year) and a box plot.


# In[237]:


medical_df.age.describe()


# In[238]:


fig = px.histogram(medical_df, 
                   x='age', 
                   marginal='box', 
                   nbins=47, 
                   title='Distribution of Age')
fig.update_layout(bargap=0.1)
fig.show()


# In[239]:


### Body Mass Index

#Let's look at the distribution of BMI (Body Mass Index) of customers, using a histogram and box plot.


# In[240]:


fig = px.histogram(medical_df, 
                   x='bmi', 
                   marginal='box', 
                   color_discrete_sequence=['red'], 
                   title='Distribution of BMI (Body Mass Index)')
fig.update_layout(bargap=0.1)
fig.show()


# In[241]:


### Charges

#Let's visualize the distribution of "charges" i.e. the annual medical charges for customers. This is the column we're trying to predict. Let's also use the categorical column "smoker" to distinguish the charges for smokers and non-smokers.


# In[242]:


fig = px.histogram(medical_df, 
                   x='charges', 
                   marginal='box', 
                   color='smoker', 
                   color_discrete_sequence=['green', 'grey'], 
                   title='Annual Medical Charges')
fig.update_layout(bargap=0.1)
fig.show()


# In[243]:


### Smoker

#Let's visualize the distribution of the "smoker" column (containing values "yes" and "no") using a histogram.


# In[244]:


medical_df.smoker.value_counts()


# In[245]:


px.histogram(medical_df, x='smoker', color='sex', title='Smoker')


# In[246]:


### Age and Charges

#Let's visualize the relationship between "age" and "charges" using a scatter plot. Each point in the scatter plot represents one customer. We'll also use values in the "smoker" column to color the points.


# In[247]:


fig = px.scatter(medical_df, 
                 x='age', 
                 y='charges', 
                 color='smoker', 
                 opacity=0.8, 
                 hover_data=['sex'], 
                 title='Age vs. Charges')
fig.update_traces(marker_size=5)
fig.show()


# In[248]:


#We can make the following observations from the above chart:

#The general trend seems to be that medical charges increase with age, as we might expect. However, there is significant variation at every age, and it's clear that age alone cannot be used to accurately determine medical charges.


#We can see three "clusters" of points, each of which seems to form a line with an increasing slope:

    # 1.The first and the largest cluster consists primary of presumably "healthy non-smokers" who have relatively low medical charges compared to others
     
    # 2. The second cluster contains a mix of smokers and non-smokers. It's possible that these are actually two distinct but overlapping clusters: "non-smokers with medical issues" and "smokers without major medical issues".
     
    # 3. The final cluster consists exclusively of smokers, presumably smokers with major medical issues that are possibly related to or worsened by smoking.


# In[249]:


fig = px.scatter(medical_df, 
                 x='bmi', 
                 y='charges', 
                 color='smoker', 
                 opacity=0.8, 
                 hover_data=['sex'], 
                 title='BMI vs. Charges')
fig.update_traces(marker_size=5)
fig.show()


# In[250]:


### Correlation


# In[251]:


medical_df.charges.corr(medical_df.age)


# In[252]:


medical_df.charges.corr(medical_df.bmi)


# In[253]:


smoker_values = {'no': 0, 'yes': 1}
smoker_numeric = medical_df.smoker.map(smoker_values)
medical_df.charges.corr(smoker_numeric)


# In[254]:


medical_df.corr()


# In[255]:


sns.heatmap(medical_df.corr(), cmap='Reds', annot=True)
plt.title('Correlation Matrix');


# In[256]:


jovian.commit()


# In[257]:


## Linear Regression using a Single Feature


# In[258]:


non_smoker_df = medical_df[medical_df.smoker == 'no']


# In[259]:


plt.title('Age vs. Charges')
sns.scatterplot(data=non_smoker_df, x='age', y='charges', alpha=0.7, s=15);


# In[260]:


### Model
#linear regression
#charges=w×age+b


# In[261]:


def estimate_charges(age, w, b):
    return w * age + b


# In[262]:


w = 50
b = 100
ages = non_smoker_df.age
estimated_charges = estimate_charges(ages, w, b)


# In[263]:


plt.plot(ages, estimated_charges, 'r-o');
plt.xlabel('Age');
plt.ylabel('Estimated Charges');


# In[264]:


target = non_smoker_df.charges

plt.plot(ages, estimated_charges, 'r', alpha=0.9);
plt.scatter(ages, target, s=8,alpha=0.8);
plt.xlabel('Age');
plt.ylabel('Charges')
plt.legend(['Estimate', 'Actual']);


# In[265]:


def try_parameters(w, b):
    ages = non_smoker_df.age
    target = non_smoker_df.charges
    
    estimated_charges = estimate_charges(ages, w, b)
    
    plt.plot(ages, estimated_charges, 'r', alpha=0.9);
    plt.scatter(ages, target, s=8,alpha=0.8);
    plt.xlabel('Age');
    plt.ylabel('Charges')
    plt.legend(['Estimate', 'Actual']);


# In[266]:


try_parameters(60, 200)


# In[267]:


try_parameters(400, 5000)


# In[268]:


### Loss/Cost Function


#Let's define a function to compute the RMSE.


# In[269]:


get_ipython().system('pip install numpy --quiet')
import numpy as np


# In[270]:


def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets - predictions)))


# In[271]:


w = 50
b = 100
try_parameters(w, b)


# In[272]:


targets = non_smoker_df['charges']
predicted = estimate_charges(non_smoker_df.age, w, b)


# In[273]:


rmse(targets, predicted)


# In[274]:


def try_parameters(w, b):
    ages = non_smoker_df.age
    target = non_smoker_df.charges
    predictions = estimate_charges(ages, w, b)
    
    plt.plot(ages, predictions, 'r', alpha=0.9);
    plt.scatter(ages, target, s=8,alpha=0.8);
    plt.xlabel('Age');
    plt.ylabel('Charges')
    plt.legend(['Prediction', 'Actual']);
    
    loss = rmse(target, predictions)
    print("RMSE Loss: ", loss)


# In[275]:


try_parameters(50, 100)


# In[276]:


### Linear Regression using Scikit-learn


# In[277]:


get_ipython().system('pip install scikit-learn --quiet')

from sklearn.linear_model import LinearRegression

model = LinearRegression()


# In[278]:


inputs = non_smoker_df[['age']]
targets = non_smoker_df.charges
print('inputs.shape :', inputs.shape)
print('targes.shape :', targets.shape)


# In[279]:


model.fit(inputs, targets)


# In[280]:


model.predict(np.array([[23], 
                        [37], 
                        [61]]))


# In[281]:


predictions = model.predict(inputs)


# In[282]:


predictions


# In[283]:


rmse(targets, predictions)


# In[284]:


# w
model.coef_


# In[285]:


# b
model.intercept_


# In[286]:


try_parameters(model.coef_, model.intercept_)


# In[287]:


# Create inputs and targets
inputs, targets = non_smoker_df[['age']], non_smoker_df['charges']

# Create and train the model
model = LinearRegression().fit(inputs, targets)

# Generate predictions
predictions = model.predict(inputs)

# Compute loss to evalute the model
loss = rmse(targets, predictions)
print('Loss:', loss)


# In[288]:


## Linear Regression using Multiple Features

#So far, we've used on the "age" feature to estimate "charges". Adding another feature like "bmi" is fairly straightforward. We simply assume the following relationship:

#charges = w_1 \times age + w_2 \times bmi + b


# In[289]:


# Create inputs and targets
inputs, targets = non_smoker_df[['age', 'bmi']], non_smoker_df['charges']

# Create and train the model
model = LinearRegression().fit(inputs, targets)

# Generate predictions
predictions = model.predict(inputs)

# Compute loss to evalute the model
loss = rmse(targets, predictions)
print('Loss:', loss)


# In[290]:


non_smoker_df.charges.corr(non_smoker_df.bmi)


# In[291]:


fig = px.scatter(non_smoker_df, x='bmi', y='charges', title='BMI vs. Charges')
fig.update_traces(marker_size=5)
fig.show()


# In[292]:


fig = px.scatter_3d(non_smoker_df, x='age', y='bmi', z='charges')
fig.update_traces(marker_size=3, marker_opacity=0.5)
fig.show()


# In[293]:


model.coef_, model.intercept_


# In[294]:


non_smoker_df.charges.corr(non_smoker_df.children)


# In[295]:


fig = px.strip(non_smoker_df, x='children', y='charges', title= "Children vs. Charges")
fig.update_traces(marker_size=4, marker_opacity=0.7)
fig.show()


# In[296]:


# Create inputs and targets
inputs, targets = non_smoker_df[['age', 'bmi', 'children']], non_smoker_df['charges']

# Create and train the model
model = LinearRegression().fit(inputs, targets)

# Generate predictions
predictions = model.predict(inputs)

# Compute loss to evalute the model
loss = rmse(targets, predictions)
print('Loss:', loss)


# In[297]:


# Create inputs and targets
inputs, targets = medical_df[['age', 'bmi', 'children']], medical_df['charges']

# Create and train the model
model = LinearRegression().fit(inputs, targets)

# Generate predictions
predictions = model.predict(inputs)

# Compute loss to evalute the model
loss = rmse(targets, predictions)
print('Loss:', loss)


# In[298]:


## Using Categorical Features for Machine Learning

#Binary Categories
#The "smoker" category has just two values "yes" and "no". Let's create a new column "smoker_code" containing 0 for "no" and 1 for "yes".


# In[299]:


sns.barplot(data=medical_df, x='smoker', y='charges');


# In[300]:


smoker_codes = {'no': 0, 'yes': 1}
medical_df['smoker_code'] = medical_df.smoker.map(smoker_codes)
medical_df.charges.corr(medical_df.smoker_code)


# In[301]:


medical_df


# In[302]:


# Create inputs and targets
inputs, targets = medical_df[['age', 'bmi', 'children', 'smoker_code']], medical_df['charges']

# Create and train the model
model = LinearRegression().fit(inputs, targets)

# Generate predictions
predictions = model.predict(inputs)

# Compute loss to evalute the model
loss = rmse(targets, predictions)
print('Loss:', loss)


# In[303]:


sns.barplot(data=medical_df, x='sex', y='charges')


# In[304]:


sex_codes = {'female': 0, 'male': 1}
medical_df['sex_code'] = medical_df.sex.map(sex_codes)
medical_df.charges.corr(medical_df.sex_code)


# In[305]:


# Create inputs and targets
inputs, targets = medical_df[['age', 'bmi', 'children', 'smoker_code', 'sex_code']], medical_df['charges']

# Create and train the model
model = LinearRegression().fit(inputs, targets)

# Generate predictions
predictions = model.predict(inputs)

# Compute loss to evalute the model
loss = rmse(targets, predictions)
print('Loss:', loss)


# In[306]:


### One-hot Encoding


# In[307]:


sns.barplot(data=medical_df, x='region', y='charges');


# In[308]:


from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
enc.fit(medical_df[['region']])
enc.categories_


# In[309]:


one_hot = enc.transform(medical_df[['region']]).toarray()
one_hot


# In[310]:


medical_df[['northeast', 'northwest', 'southeast', 'southwest']] = one_hot


# In[311]:


medical_df


# In[312]:


# Create inputs and targets
input_cols = ['age', 'bmi', 'children', 'smoker_code', 'sex_code', 'northeast', 'northwest', 'southeast', 'southwest']
inputs, targets = medical_df[input_cols], medical_df['charges']

# Create and train the model
model = LinearRegression().fit(inputs, targets)

# Generate predictions
predictions = model.predict(inputs)

# Compute loss to evalute the model
loss = rmse(targets, predictions)
print('Loss:', loss)


# In[313]:


###Model Improvements

### Feature Scaling


# In[314]:


model.coef_


# In[315]:


model.intercept_


# In[316]:


weights_df = pd.DataFrame({
    'feature': np.append(input_cols, 1),
    'weight': np.append(model.coef_, model.intercept_)
})
weights_df


# In[317]:


medical_df


# In[318]:


from sklearn.preprocessing import StandardScaler
numeric_cols = ['age', 'bmi', 'children'] 
scaler = StandardScaler()
scaler.fit(medical_df[numeric_cols])


# In[319]:


scaler.mean_


# In[320]:


scaler.var_


# In[321]:


scaled_inputs = scaler.transform(medical_df[numeric_cols])
scaled_inputs


# In[322]:


cat_cols = ['smoker_code', 'sex_code', 'northeast', 'northwest', 'southeast', 'southwest']
categorical_data = medical_df[cat_cols].values


# In[323]:


inputs = np.concatenate((scaled_inputs, categorical_data), axis=1)
targets = medical_df.charges

# Create and train the model
model = LinearRegression().fit(inputs, targets)

# Generate predictions
predictions = model.predict(inputs)

# Compute loss to evalute the model
loss = rmse(targets, predictions)
print('Loss:', loss)


# In[324]:


weights_df = pd.DataFrame({
    'feature': np.append(numeric_cols + cat_cols, 1),
    'weight': np.append(model.coef_, model.intercept_)
})


# In[325]:


weights_df


# In[326]:


### Creating a Test Set


# In[327]:


from sklearn.model_selection import train_test_split


# In[328]:


inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.1)


# In[329]:


# Create and train the model
model = LinearRegression().fit(inputs_train, targets_train)

# Generate predictions
predictions_test = model.predict(inputs_test)

# Compute loss to evalute the model
loss = rmse(targets_test, predictions_test)
print('Test Loss:', loss)


# In[330]:


# Generate predictions
predictions_train = model.predict(inputs_train)

# Compute loss to evalute the model
loss = rmse(targets_train, predictions_train)
print('Training Loss:', loss)


# In[331]:


jovian.commit()


# In[ ]:




