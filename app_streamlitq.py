import streamlit as st
import joblib, numpy as np, math

rf = joblib.load("rf_model.joblib")
sc = joblib.load("scaler.joblib")
le = joblib.load("encoder.joblib")

st.title("🚴 Bicycle Frame Safety BOT")

E=st.selectbox("Material",["Steel","Al6061","Chromoly"])
if E=="Steel": E_MPa,y,u=(210000,250,400)
elif E=="Al6061": E_MPa,y,u=(69000,150,310)
else: E_MPa,y,u=(210000,450,650)

d=st.slider("Outer diameter (mm)",20,45,30)
t=st.slider("Thickness (mm)",1,3,2)
L=st.slider("Tube length (mm)",200,700,350)
r=st.slider("Rider weight (kg)",55,110,75)
g=9.81
Fa=r*g*st.slider("Axial load fraction",0.0,1.0,0.3)
Ft=r*g*st.slider("Transverse load fraction",0.0,1.0,0.5)

def areaI(d,t): di=d-2*t; return math.pi*(d**2-di**2)/4, math.pi*(d**4-di**4)/64
A,I=areaI(d,t)
sa=Fa/A; M=Ft*L; sb=M*(d/2)/I; svm=(sa**2+3*sb**2)**0.5
E=E_MPa; delta=Ft*L**3/(3*E*I)
SF=y/svm; Nf=(abs(sb)/1e3)**(-10)

x=[E_MPa,y,u,d,t,L,r,Fa,Ft,sa,sb,svm,delta,SF,Nf]
xs=sc.transform([x])
pred=rf.predict(xs)[0]
label=le.inverse_transform([pred])[0]

if st.button("Predict Safety"):
    st.subheader(f"Result: {label}")
    st.write(f"Safety Factor: {SF:.2f}, Von Mises: {svm:.1f} MPa, Deflection: {delta:.3f} mm")
