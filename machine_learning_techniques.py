from sklearn.cluster import KMeans
import skfuzzy as fuzz
import cv2 
import numpy as np

def adaptive_thresholding(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

#apply the method
segmented = [adaptive_thresholding(img) for img in X_val]

def otsu_thresholding(img):
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

segmented = [otsu_thresholding(img) for img in X_val]

def k_means_segmentation(img, n_clusters=2):
    img_reshaped = img.reshape((-1, 1))
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(img_reshaped)
    segmented = kmeans.labels_.reshape(img.shape)
    return (segmented == 1).astype(np.uint8) * 255  # Assuming tumor is the second cluster

segmented = [k_means_segmentation(img) for img in X_val]


def fuzzy_c_means_segmentation(img, n_clusters=2):
    img_reshaped = img.reshape((-1, 1))
    _, u, _, _, _, _, _ = fuzz.cluster.cmeans(img_reshaped.T, n_clusters, 2, error=0.005, maxiter=1000)
    segmented = u[1].reshape(img.shape)
    return (segmented > 0.5).astype(np.uint8) * 255

# Apply the method
segmented = [fuzzy_c_means_segmentation(img) for img in X_val]
