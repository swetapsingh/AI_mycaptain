import numpy as np
from sklearn.datasets import load_sample_image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load sample images from scikit-learn (toy dataset)
china = load_sample_image("china.jpg")
flower = load_sample_image("flower.jpg")

# Combine the two images into a single dataset
data = np.array([china, flower])
n_samples, height, width, channels = data.shape

# Reshape the data into a 2D array (n_samples x n_features)
X = data.reshape((n_samples, -1))
y = np.array([0, 1])  # Labels for the two classes (China and Flower)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=0)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy}")
