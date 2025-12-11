import numpy as np
import pandas as pd
import math, random

def area_outer_inner(d_o, t):
    d_i = d_o - 2*t
    A = math.pi*(d_o**2 - d_i**2)/4.0
    I = math.pi*(d_o**4 - d_i**4)/64.0
    return A, I

def compute_stresses(d_o, t, L, F_axial, F_trans):
    A, I = area_outer_inner(d_o, t)
    sigma_axial = F_axial / A
    M = F_trans * L
    c = d_o/2
    sigma_bending = M * c / I
    sigma_vm = math.sqrt(sigma_axial**2 + 3*sigma_bending**2)
    E = 210000  # MPa (steel)
    delta = F_trans * L**3 / (3 * E * I)
    return sigma_axial, sigma_bending, sigma_vm, delta

def estimate_fatigue_life(sigma_a, sigma_m, sigma_uts=400):
    A = 1e3; b = -0.1
    denom = max(1e-6, 1 - sigma_m/sigma_uts)
    sigma_eff = sigma_a / denom
    N = (sigma_eff/A)**(1/b)
    return N

def make_dataset(n=3000):
    mats = [("Steel",210000,250,400),("Al6061",69000,150,310),("Chromoly",210000,450,650)]
    rows = []
    for _ in range(n):
        m,E,y,u = random.choice(mats)
        d_o = np.random.uniform(20,45)
        t = np.random.uniform(1,3)
        L = np.random.uniform(200,700)
        rider = np.random.uniform(55,110)
        g=9.81
        F_axial = np.random.uniform(0,rider*g)
        F_trans = np.random.uniform(0,rider*g)
        sa,sb,svm,delta = compute_stresses(d_o,t,L,F_axial,F_trans)
        SF = y/svm
        Nf = estimate_fatigue_life(sb,sa,u)
        label="Safe" if SF>2 and Nf>1e6 else "Risk" if SF>1 else "Failure"
        rows.append([m,E,y,u,d_o,t,L,rider,F_axial,F_trans,sa,sb,svm,delta,SF,Nf,label])
    df=pd.DataFrame(rows,columns=["mat","E","yield","uts","d_o","t","L","rider","F_ax","F_tr","sa","sb","svm","delta","SF","Nf","label"])
    df.to_csv("dataset.csv",index=False)
    print("✅ Saved dataset.csv with",len(df),"rows")

if __name__=="__main__":
    make_dataset()
