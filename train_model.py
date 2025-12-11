import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("dataset.csv")

X = df[["E","yield","uts","d_o","t","L","rider","F_ax","F_tr","sa","sb","svm","delta","SF","Nf"]]
y = df["label"]

le = LabelEncoder()
y_enc = le.fit_transform(y)
sc = StandardScaler()
X_s = sc.fit_transform(X)

Xtr,Xts,ytr,yts = train_test_split(X_s,y_enc,test_size=0.2,random_state=42)
rf = RandomForestClassifier(n_estimators=200,max_depth=10,random_state=42)
rf.fit(Xtr,ytr)

print(classification_report(yts,rf.predict(Xts),target_names=le.classes_))

joblib.dump(rf,"rf_model.joblib")
joblib.dump(sc,"scaler.joblib")
joblib.dump(le,"encoder.joblib")
print("✅ Model trained and saved!")
