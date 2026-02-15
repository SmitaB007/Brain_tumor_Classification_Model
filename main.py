import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = []
labels = []
img_size=64

for category in ["Normal","tumor"]:
    folder= os.path.join("dataset",category)
    label=1 if category == "tumor" else 0
    for img_name in  os.listdir(folder):
        img_path=os.path.join(folder,img_name)
        img=cv2.imread(img_path)

        if img is not None:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (img_size, img_size))
            vector = resized.flatten()

            data.append(vector)
            labels.append(label)

X = np.array(data)
y = np.array(labels)

# print("Data shape:", X.shape)


X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.3,random_state=42
)

scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
joblib.dump(model, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
# print("Accuracy:", accuracy*100)

# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
