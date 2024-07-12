from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.svm import SVC
from joblib import load
import numpy as np
import time
import cv2
import sys
import os

#                   prefix       name histo    name label             suffix                 model directory 
#Opt = ["../model_test/test1_", "histos", "labels", "_1_8_ror.npy", "../models/"]
Opt = ["../data/processed_Data/set_2_", "Data" , "Labels",   ".npy",    "../models_Supervised/"]


histo = np.load(Opt[0] + Opt[1] + Opt[3])
label = np.load(Opt[0] + Opt[2] + Opt[3])

print(histo.shape,label.shape)
unique_labels, counts = np.unique(label, return_counts=True)
print("Label distribution:", dict(zip(unique_labels, counts)))


model_path = Opt[4]

# List all model files in the model directory
model_files = [f for f in os.listdir(model_path) if f.endswith('.pkl')]

best_accuracy = 0
best_model_file = None

# Evaluate each model
for model_file in model_files:
    model_filename = os.path.join(model_path, model_file)
    loaded_model = load(model_filename)
    
    # Predict using the loaded model
    predictions = loaded_model.predict(histo)
    
    # Calculate accuracy
    accuracy = accuracy_score(label, predictions)
    print(f"Accuracy with model {model_file}:", accuracy)
    
    # Update best model if current model is better
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_file = model_file
    

# Print the best model and its accuracy
if best_model_file is not None:
    print(f"\nBest model: {best_model_file}")
    print(f"Best accuracy: {best_accuracy}")
    # Compute confusion matrix for the best model
    best_model = load(os.path.join(model_path, best_model_file))
    predictions = best_model.predict(histo)
    cm = confusion_matrix(label, predictions)

    if True:
        # write the results in real_results.txt
        with open("real_results.txt", 'a') as file:
            file.write('\n')
            file.write(f'Best model: {best_model_file}\n')
            file.write(f'Best accuracy: {best_accuracy}\n\n')
            
            file.write("Confusion Matrix:\n")
            for row in cm:
                file.write(" ".join(map(str, row)) + "\n")

    
    
    
    print("\nConfusion Matrix:")
    print(cm)
    if True:
        # Plot confusion matrix
        plt.imshow(cm, cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
else:
    print("No models found.")
