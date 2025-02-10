import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import pickle
import numpy as np

# Load the data
data = pd.read_csv('sonar.all.data', header=None, delimiter=" ")
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]

# Label encoding
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.3f}")

# Create a dictionary with the model and training data
data_dict = {'model': model, 'X_train': X_train, 'y_train': y_train}

# Save the dictionary to a pickle file
with open('logistic_model.pkl', 'wb') as file:
    pickle.dump(data_dict, file)
