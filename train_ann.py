# train_ann.py - trains MLPClassifier on 3 classes, saves artifacts
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

CSV = "bolt_dataset.csv"
if not os.path.exists(CSV):
    raise RuntimeError("Dataset bolt_dataset.csv not found. Run make_dataset.py first.")

df = pd.read_csv(CSV)
feat_cols = [c for c in df.columns if c.startswith("f")]
X = df[feat_cols].values
y_raw = df["label"].values

# encode labels (Safe, Risk, Failure)
le = LabelEncoder().fit(y_raw)
y = le.transform(y_raw)
print("Classes:", list(le.classes_))

# scale features
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# train/test split (stratify by class)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# MLP with class weight handling via sample_weight
clf = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', solver='adam',
                    max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# save models folder if missing
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/ann_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(le, "models/label_encoder.joblib")
print("Models saved in models/ folder")

# evaluate
y_pred = clf.predict(X_test)
print("Classification report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

