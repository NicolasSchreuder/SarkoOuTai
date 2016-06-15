__author__ = 'Nicolas Schreuder, Sholom Schechtman, Pierre Foret'

import os
from sklearn.decomposition import PCA
import cv2
from skimage.feature import daisy
import numpy as np
from sklearn.cross_validation import train_test_split
import sklearn.metrics as skmetrics
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import joblib

# We set the path from where we get the data from
data_path=os.path.join(os.getcwd(), 'ReconSarko')
os.chdir(data_path)

# We create the train matrix by associating to each face image its daisy descriptor
face_folders = os.listdir(os.getcwd())
X, y, id_target = [], [], 1
for face_folder in face_folders:
    if face_folder != '.DS_Store':
        os.chdir(face_folder+'/processed_faces')
        faces_list = os.listdir(os.getcwd())
        faces_list.remove('.DS_Store')
        for face_path in faces_list:
            face = cv2.imread(face_path, 0)
            descriptor = daisy(face, step=2)
            flat_descriptor = np.asarray(descriptor).reshape(-1)
            X.append(flat_descriptor)
            y.append(id_target)
        id_target -= 1
        os.chdir(data_path)

# We train/test split the matrix of daisy descriptors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# We apply PCA to the train and test matrix in order to reduce their size
pca = PCA(n_components=400)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("Beginning of the learning phase")
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf0 = GridSearchCV(SVC(kernel='rbf', class_weight='auto', probability=False), param_grid)
clf = clf0.fit(X_train_pca, y_train)

print("Best estimator found by grid search:")
print(clf.best_estimator_)

clf.fit(X_train_pca, y_train)
y_pred = clf.predict(X_test_pca)
print('Percentage of accurate predictions :', skmetrics.precision_score(y_pred, y_test))


# We save the classifier and the pca parameters in pickle files
a=joblib.dump(clf, 'ClassifierSarko')
b=joblib.dump(pca, 'PCASarko')
print('Classifier and PCA saved')
