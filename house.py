# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 10:59:47 2021

@author: GbolahanOlumade
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("minihomeprices.csv")

df.info()

df.describe()

# Descriptiion of our data set

df.describe().style.background_gradient(cmap='CMRmap')


#  to know how many null values

df.isna().sum()

# fill null values with median value

df['bedrooms'] = df['bedrooms'].fillna( df['bedrooms'].mean() )

df['bedrooms'] = df['bedrooms'].fillna( df['bedrooms'].mean())

#  here we can use inplace=True as well.  both are valid for update data frame

df.head()

# barploat show

plt.figure(figsize=(10, 7))
plt.title("Bedroom wise price increase.")

sns.barplot('bedrooms', 'price', data=df)
plt.xlabel('Bedrooms', )
plt.ylabel('Price')
plt.show()

plt.figure(figsize=(10, 5))

sns.scatterplot('bedrooms', 'price',data=df)
plt.title("Price vs Bedroom Scatter plot")

plt.xlabel("House Bedrooms")
plt.ylabel('House Price')
plt.show()


plt.figure(figsize=(10, 7))

sns.lmplot(x="bedrooms", y="price", data=df);
plt.title("Price and bedroom wise line plot")
plt.show()


from sklearn.linear_model import LinearRegression

ln = LinearRegression()

X = df.drop(['price'], axis=1)
y = df['price']

df['bedrooms'] = df['bedrooms'].astype('int64')

df.info()

print(X)
print("_" * 25)
print(y)

ln.fit(X, y)

ln.predict([[4000, 2, 50]])

ln.coef_

ln.intercept_

#  know score 
score = ln.score( X, y )

print(score * 100)










