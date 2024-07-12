from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.svm import SVC
import numpy as np
import joblib
import time
import cv2
import sys
import os

#            prefix            name histo    name label      suffix                 model path
#Opt = ["../model_test/","final_histo","final_label",   "_1_8_ror_noOther.npy",      "../models/"]
Opt = ["../data/processed_Data/set_11_", "Data" , "Labels",   ".npy",      "../models_Supervised/"]

extention = Opt[2]
histo = np.load(Opt[0] + Opt[1] + Opt[3])
label = np.load(Opt[0] + Opt[2] + Opt[3])

print(histo.shape,label.shape)
unique_labels, counts = np.unique(label, return_counts=True)
print("Label distribution:", dict(zip(unique_labels, counts)))



X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(histo, label, test_size=0.3)




model_path = Opt[4]
if not os.path.exists(model_path):
    os.makedirs(model_path)
    
# Save SVM models with different parameters
#for C_val in [0.01, 0.1, 1.0, 10, 100]:
for C_val in [ 1.0, 5, 10 ]:
    #for kernel_val in ['linear', 'poly', 'rbf', 'sigmoid']:
    for kernel_val in ['poly','rbf']:
        model_svm = SVC(C=C_val, kernel=kernel_val,class_weight='balanced')
        #model_svm = SVC(C=C_val, kernel=kernel_val)
        model_svm.fit(X_train, y_train)
        accuracy = model_svm.score(X_test, y_test)
        print(f"SVM with C={C_val}, kernel={kernel_val} - Accuracy: {accuracy}")
        joblib.dump(model_svm, f"{model_path}/svm_C_{C_val}_kernel_{kernel_val}.pkl")


import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Save Logistic Regression models with different max_iter values
#for i in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
if True:
    for i in [100, 1000, 10000, 100000]:
        model_logreg = LogisticRegression(max_iter=i,class_weight='balanced')
        #model_logreg = LogisticRegression(max_iter=i)
        model_logreg.fit(X_train, y_train)
        accuracy = model_logreg.score(X_test, y_test)  
        print(f"Logistic Regression with max_iter={i} - Accuracy: {accuracy}")
        joblib.dump(model_logreg, f"{model_path}/logreg_max_iter_{i}.pkl")



# Save KNN models with different n_neighbors values
if True:
    for i in [1,3,8]:
        model_knn = KNeighborsClassifier(n_neighbors=i,weights='uniform')
        #model_knn = KNeighborsClassifier(n_neighbors=i)
        model_knn.fit(X_train, y_train)
        accuracy = model_knn.score(X_test, y_test)  
        print(f"KNN with n_neighbors={i} - Accuracy: {accuracy}")
        joblib.dump(model_knn, f"{model_path}/knn_neighbors_{i}.pkl")
