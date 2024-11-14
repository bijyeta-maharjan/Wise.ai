# %% [markdown]
# # SVM Classification
# 
# ### This file has the final classification using SVM
# 
# 1. The data consisting of 12 features is separated into train and test in the ratio 60:40. 
# 2. This data is scaled and fed into the SVM. 
# 3. The poly kernel for our algorithm was chosen because a linear kernel was not able to capture the complexity of the data. 
# 4. C is chosen to be 170 and it depicts how close to the margin are the data points. 
# 5. Once the SVM is trained, it is tested against the test set and parameters like accuracy and error rate are obtained.
# 

# %% [markdown]
# Import dependencies

# %%
import numpy as np 
import pandas as pd
import joblib
import librosa
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_classif

# Load the dataset
df = pd.read_excel(r'/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/Audio_Final_Data.xlsx')
df.dataframeName = 'Audio_Final_Data.csv'
df.head(5)

# Define features and target
X = df.iloc[:, 0:12]
y = df.iloc[:, -1].values  # Ensure y is a 1D array

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Feature selection using RFE
svc = SVC(kernel="linear")
rfe = RFE(estimator=svc, n_features_to_select=10, step=1)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

# Define a pipeline to streamline preprocessing, feature selection, and model training
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_classif, k=20)),
    ('svm', SVC())
])

# Define the parameter grid
param_grid = {
    'feature_selection__k': [5, 10, 20],  # Number of features to select
    'svm__C': [0.1, 1, 10, 100, 200, 500],
    'svm__kernel': ['poly', 'rbf', 'sigmoid'],
    'svm__degree': [2, 3, 4, 5],  # Only relevant for poly kernel
    'svm__gamma': ['scale', 'auto']
}

# Initialize GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
# Fit the model
grid_search.fit(X_train_rfe, y_train)

# Print the best parameters and the corresponding accuracy
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation accuracy: {:.2f}".format(grid_search.best_score_))

# Evaluate on the test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test_rfe, y_test)
print("Test set accuracy: {:.2f}".format(test_accuracy))

# Save the model
joblib.dump(best_model, 'svm_fall_detection_model.pkl')
print("Model saved as svm_fall_detection_model.pkl")

# Function to extract features from an audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, mono=True)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Function to load the model and make predictions
def predict_fall(audio_file):
    # Load the trained model
    model = joblib.load('svm_fall_detection_model.pkl')
    
    # Extract features from the audio file
    features = extract_features(audio_file).reshape(1, -1)  # Ensure features shape
    
    # Transform the features using the pipeline's steps
    prediction = model.predict(features)
    return prediction


# Example usage
audio_file = input("Enter the path to the audio file: ")
prediction = predict_fall(audio_file)
print("Fall detected" if prediction == 1 else "No fall detected")

