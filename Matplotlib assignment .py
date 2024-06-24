#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# In[ ]:


housing = fetch_california_housing()
X = housing.data  # Features
y = housing.target  # Target


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[4]:


model = Sequential([
    Dense(50, activation='relu', input_shape=[X_train.shape[1]]),  # First hidden layer
    Dense(50, activation='relu'),  # Second hidden layer
    Dense(50, activation='relu'),  # Third hidden layer
    Dense(1)  # Output layer
])


# In[5]:


model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on test set: {mse}")


# In[6]:


print("\nExample predictions:")
for i in range(5):
    print(f"Prediction: {y_pred[i][0]:.2f}, Actual: {y_test[i]:.2f}")


# In[7]:


plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

