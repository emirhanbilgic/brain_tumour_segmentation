import cv2
import numpy as np
from sklearn.model_selection import train_test_split

#load data, dataloaders can be used for this purpose.
X = []  # Images
y = []  # Masks

#split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
