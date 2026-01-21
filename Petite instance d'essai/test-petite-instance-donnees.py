import pandas as pd
import pyomo.environ as pyo
from itertools import product
from pyomo.environ import value
import numpy as np
import math

model = pyo.ConcreteModel()


# 1. SETS

T=[0, 1, 2, 3, 4, 5]  # horizon temporel
A = list(pd.read_csv("AREAS_instance.csv", sep=";")["AREAS"])  # zones
I=['ORANGE','FREE MOBILE', 'BOUYGUES TELECOM','SFR']  # opérateurs
τ='ORANGE'  # notre opérateur
O=['o3G-FREE MOBILE', 'o3G-BOUYGUES TELECOM', 'o3G-SFR','o3G-ORANGE','o4G-FREE MOBILE','o4G-BOUYGUES TELECOM','o4G-SFR','o4G-ORANGE','o5G-FREE MOBILE','o5G-BOUYGUES TELECOM','o5G-SFR','o5G-ORANGE']  
O_i = {"ORANGE" : ['o3G-ORANGE' , 'o4G-ORANGE', 'o5G-ORANGE'] , 'FREE MOBILE' : ['o3G-FREE MOBILE', 'o4G-FREE MOBILE', 'o5G-FREE MOBILE'] ,
      'BOUYGUES TELECOM' : ['o3G-BOUYGUES TELECOM', 'o4G-BOUYGUES TELECOM', 'o5G-BOUYGUES TELECOM'] , 'SFR' : ['o3G-SFR', 'o4G-SFR', 'o5G-SFR']  }# offres
Si = list(pd.read_csv("EXISTING_SITES_instance.csv", sep=";")["EXISTING_SITES"]) # sites de l’opérateur τ
NG = 'o5G-ORANGE'

model.T = pyo.Set(initialize=T)  # horizon temporel
model.A = pyo.Set(initialize=A)  # zones
model.I = pyo.Set(initialize=I)  # opérateurs
model.O = pyo.Set(initialize=O)  # offres !!! il faut mettre l'offre de chaque opérateur
model.O_i = O_i
model.S = pyo.Set(initialize=Si) # sites de l’opérateur i
# Couples utiles
C_space = list(product([0,1], repeat=len(I)))  # Toutes les combinaisons de couverture
df = pd.read_csv("AREAS_SITES_LINK_instance.csv", sep=";")   # contient colonnes "a" et "s"

Sa_dict = {}

for _, row in df.iterrows():
    zone = row["AREAS"]
    site = row["SITES"]

    if zone not in Sa_dict:
        Sa_dict[zone] = []

    if site not in Sa_dict[zone]:     #éviter d'ajouter plusieurs fois la même zone
        Sa_dict[zone].append(site)
        

As_dict = {}

for _, row in df.iterrows():
    zone = row["AREAS"]
    site = row["SITES"]

    if site not in As_dict:
        As_dict[site] = []
    if zone not in As_dict[site]:
        As_dict[site].append(zone)

    


model.Cvec = pyo.Set(initialize=C_space)   # Toutes les combinaisons de couverture
model.Sa = Sa_dict                         # mapping a → {sites}
model.As = As_dict                         # mapping s → {zones}

# 2. PARAMÈTRES
df_zmax = pd.read_csv("OPERATIONAL_LIMITS2.csv", sep=";")
Zmax_data = {row.TIME_SLOTS: row["MAX_NUMBER_OF_DEPLOYMENTS"] for _,row in df_zmax.iterrows()}
Zmax_data [0] = 0
model.Zmax = pyo.Param(model.T, initialize = Zmax_data)  # nombre max de sites déployables par période

df_qa = pd.read_csv("STRATEGIC_GUIDELINES.csv", sep=";")
QA_data = {row.TIME_SLOTS: row["QA"] for _,row in df_qa.iterrows()}
QA_data[0] = 0
model.QA = pyo.Param(model.T, initialize = QA_data) # couverture minimale de la population par période

df = pd.read_csv("AREAS_instance.csv", sep=";")

ua0_data = {}
for _, row in df.iterrows():
    for col in df.columns [1:]:
        ua0_data[row["AREAS"],col.split("-")[1], col] = row[col]    #partie entière retirée
for a in A:
    for i in I:
        for o in O:
            if (a,i,o) not in ua0_data:
                ua0_data[a,i,o] = 0
    
        
model.ua0 = pyo.Param(model.A, model.I, model.O, initialize=ua0_data)  # utilisateurs initiaux

df_dng = pd.read_csv("DEMAND.csv", sep=";")

DNG_data = {row.TIME_SLOTS: row["5G"] for _,row in df_dng.iterrows()}
DNG_data[0] = 0
model.DNG = pyo.Param(model.T, initialize = DNG_data)  # DNG dépend du temps

df_capang = pd.read_csv("CAPACITY.csv", sep=";")
CAPANG_data = {row.TIME_SLOTS: row["5G"] for _,row in df_capang.iterrows()}
CAPANG_data[0] = 0
model.CAPANG = pyo.Param(model.T, initialize = CAPANG_data) # DNG et CAPANG dépendent du temps

df_ua = pd.read_csv("AREAS_instance.csv", sep=";")
u_a_data = {}
for _, row in df.iterrows():
    somme =0
    for col in df.columns [1:]:
        somme += row[col]
    u_a_data [row["AREAS"]]= somme

model.u_a = pyo.Param(model.A, initialize = u_a_data)  # utilisateurs totaux dans la zone a

df_Rcomp = pd.read_csv("COMPETITORS_STRATEGY_instance.csv", sep=";")
Rcomp_data = {}
for _, row in df_Rcomp.iterrows():
    for col in df_Rcomp.columns[2:5]:
        Rcomp_data[row["TIME_SLOTS"],row["AREAS"], col] = 1 * row[col]
model.Rcomp = pyo.Param(model.T, model.A, model.I, initialize=Rcomp_data)  # couverture des autres opérateurs

#f

df_fdata = pd.read_csv("UPGRADE_FUNCTION.csv", sep=";")

f_data = {}
for a in A:
    for _, row in df_fdata.iterrows():
        C  = (row["ORANGE"], row["FREE MOBILE"], row["BOUYGUES TELECOM"], row["SFR"])
        O1 = row["OFFERS"] + "-" + row["FROM_OPERATOR"]
        O2 = "o5G-" + row["TO_OPERATOR"]
        f_data[(a, C, O1, O2)] = row["PERCENTAGES"]

O_full= ['o3G-FREE MOBILE', 'o3G-BOUYGUES TELECOM', 'o3G-SFR', 'o3G-ORANGE',
       'o4G-FREE MOBILE', 'o4G-BOUYGUES TELECOM', 'o4G-SFR', 'o4G-ORANGE',
          'o5G-FREE MOBILE', 'o5G-BOUYGUES TELECOM', 'o5G-SFR', 'o5G-ORANGE']

for a in A:
    for C in C_space:  # tuples de 0/1
        for O1 in O_full:    # toutes les offres existantes
            for O2 in O_full:                                               
                if (a, C, O1, O2) not in f_data:
                   f_data[a, C, O1, O2] = 0

for a in A:
    for C in C_space:  
        for O1 in O_full:
            v = 1
            for O2 in O_full:
                if O2 != O1 :
                    v -= f_data[a, C, O1, O2]
            f_data[a, C, O1, O1] = v                             #tous ceux qui ne bougent pas restent chez eux.



model.f = pyo.Param(model.A, model.Cvec, model.O, model.O, initialize=f_data) # taux de migration

# 3. VARIABLES

model.z = pyo.Var(model.T, model.S, within=pyo.Binary)
model.r = pyo.Var(model.T, model.A, within=pyo.Binary)
model.delta = pyo.Var(model.T, model.A, model.Cvec, within=pyo.Binary)

model.u = pyo.Var(model.T, model.A, model.I, model.O, within=pyo.NonNegativeReals)
model.u_site = pyo.Var(model.T, model.A, model.S, within=pyo.NonNegativeReals)  #
# 4. CONTRAINTES

# (2) r_ta ≤ sum_s z_ts
def coverage_upper(m, t, a):
    return m.r[t, a] <= sum(m.z[t, s] for s in Sa_dict[a])
model.c_2 = pyo.Constraint(model.T, model.A, rule=coverage_upper)

# (3) z_ts ≤ r_ta  ∀ s couvrant a
def coverage_lower(m, t, s, a):
    if a in As_dict[s]:
        return m.z[t, s] <= m.r[t, a]
    return pyo.Constraint.Skip
model.c_3 = pyo.Constraint(model.T, model.S, model.A, rule=coverage_lower)


def delta_implication(m, t, a, *C):
    # C est un tuple binaire représentant (cτ, c1, c2, ...)
    res = 1
    # opérateur τ (index 0)
    cτ = C[0]
    res *= (m.r[t, a] * cτ + (1 - cτ) * (1 - m.r[t, a]))
    # autres opérateurs
    autres_operateurs = list(m.I)[1:]  # on exclut τ
    for k, i in enumerate(autres_operateurs):
        ck = C[k+1]
        R = m.Rcomp[t, a, i]
        res *= (R * ck + (1 - ck) * (1 - R))

    return m.delta[t, a, C] == res 

model.c_4 = pyo.Constraint(model.T, model.A, model.Cvec, rule=delta_implication)

# (5) Migration non-linéaire
# --- Ajout des variables auxiliaires et paramètres M ---
# Note: suppose que model.ua0 est disponible comme borne supérieure pour u (tu l'as déjà)
# On crée un param M[a,C,o]
M = {}
for a in model.A:
    for C in model.Cvec:
        for o in model.O:
            M_val = 0.0
            for i_prev in model.I:
                for o_prev in model.O_i[i_prev]:
                    M_val += pyo.value(model.f[a, C, o_prev, o]) * pyo.value(model.u_a[a])
            M[(a, C, o)] = M_val

M_data = M

model.M = pyo.Param(model.A, model.Cvec, model.O, initialize=M_data, mutable=True)

# variable auxiliaire y_{t-1,a,C,o}
model.y = pyo.Var(model.T, model.A, model.Cvec, model.O, within=pyo.NonNegativeReals)

# (5) 1) y <= M * delta
def y_le_Mdelta(m, t, a, o, *C):
    if t == min(m.T): return pyo.Constraint.Skip
    return m.y[t-1, a, C, o] <= m.M[a, C, o] * m.delta[t-1, a, C]
model.c_y1 = pyo.Constraint(model.T, model.A, model.O, model.Cvec, rule=y_le_Mdelta)

# (5) 2) y >= 0  (déjà forcé par var NonNegativeReals) -> pas nécessaire

# (5) 3) y <= S
def y_le_S(m, t, a, o, *C):
    if t == min(m.T): return pyo.Constraint.Skip
    S_expr = sum(model.f[a, C, o_prev, o] * m.u[t-1, a, i_prev, o_prev]
                 for i_prev in m.I for o_prev in m.O_i[i_prev])
    return m.y[t-1, a, C, o] <= S_expr
model.c_y2 = pyo.Constraint(model.T, model.A, model.O, model.Cvec, rule=y_le_S)

# (5) 4) y >= S - M*(1-delta)
def y_ge_S_minus_M_1minusdelta(m, t, a, o, *C):
    if t == min(m.T): return pyo.Constraint.Skip
    S_expr = sum(model.f[a, C, o_prev, o] * m.u[t-1, a, i_prev, o_prev]
                 for i_prev in m.I for o_prev in m.O_i[i_prev])
    return m.y[t-1, a, C, o] >= S_expr - m.M[a, C, o] * (1 - m.delta[t-1, a, C])
model.c_y3 = pyo.Constraint(model.T, model.A, model.O, model.Cvec, rule=y_ge_S_minus_M_1minusdelta)

# Remplacer la contrainte migration non-linéaire par la somme des y
def migration_lin(m, t, a, i, o):
    if o in model.O_i[i]:    
        if t == min(m.T):
            return m.u[t, a, i, o] == pyo.value(m.ua0[a, i, o])
        return m.u[t, a, i, o] == sum(m.y[t-1, a, C, o] for C in m.Cvec)
    else:
        return m.u[t, a, i, o] == 0
model.c_5_lin = pyo.Constraint(model.T, model.A, model.I, model.O, rule=migration_lin)


# (6) u_NO = somme sur les sites # à revoir
def assign_users(m, t, a):
    return m.u[t, a, τ, NG] == sum(m.u_site[t, a, s] for s in Sa_dict[a])
model.c_6 = pyo.Constraint(model.T, model.A, rule=assign_users)

# # (7) capacité
def capacity(m, t, s):
    return sum(model.DNG[t] * m.u_site[t, a, s] for a in As_dict[s]) <= model.CAPANG[t] * m.z[t, s]
model.c_7 = pyo.Constraint(model.T, model.S, rule=capacity)

# (8) budget sur le nombre de sites déployés par période
def limit_z(m, t):
    if t == min(m.T):
        return sum(m.z[t, s] for s in model.S) <= model.Zmax[t]
    return sum(m.z[t, s] - m.z[t-1, s] for s in model.S) <= model.Zmax[t]

model.c_8 = pyo.Constraint(model.T, rule=limit_z)

# (9) Couverture population
def cov_pop(m, t):
    return sum(m.u_a[a] * m.r[t, a] for a in model.A) >= model.QA[t] * sum(m.u_a[a] for a in model.A)
model.c_9 = pyo.Constraint(model.T, rule=cov_pop)

# (10) Croissance des z_st
def growth_z(m, t, s):
    if t == min(m.T):
        return pyo.Constraint.Skip
    return m.z[t, s]>= m.z[t-1, s]
model.c_growth_z = pyo.Constraint(model.T, model.S, rule=growth_z)

  
# 5. OBJECTIF

def objective(m):
    T_end = max(m.T)
    return sum(m.u[T_end, a, τ, 'o5G-ORANGE'] for a in m.A)
model.obj = pyo.Objective(rule=objective, sense=pyo.maximize)



model.write('model.lp', io_options={'symbolic_solver_labels': True})

# Create solver
solver = pyo.SolverFactory('glpk')

# Solve
results = solver.solve(model, tee=True)

# Display solver status




# 1) Valeur de l'objectif
print("Valeur de l'objectif :", value(model.obj))

# 2) Sites NG déployés (z_t,s = 1)
print("\nSites NG déployés (z[t,s] = 1) :")
for t in model.T:
    for s in model.S:
        if value(model.z[t, s]) > 0.5:
            print(f"t={t}, site={s}, z={value(model.z[t,s])}")

# 3) Couverture des zones (r_t,a)
print("\nCouverture par zone (r[t,a]) :")
for t in model.T:
    for a in model.A:
        if value(model.r[t, a]) > 0.5:
            print(f"t={t}, zone={a}, r={value(model.r[t,a])}")

# 4) Nombre de clients 5G ORANGE à la fin de l'horizon
T_end = max(model.T)
print(f"\nClients 5G ORANGE à t={T_end} :")
for a in model.A:
    print(f"zone={a}, u={value(model.u[T_end, a, 'ORANGE', 'o5G-ORANGE'])}")
print("\nSolver status:", results.solver.status)
print("Termination condition:", results.solver.termination_condition)
df_fdata = pd.read_csv("UPGRADE_FUNCTION.csv", sep=";")
print("FROM_OPERATOR uniques :", df_fdata["FROM_OPERATOR"].unique())
print("TO_OPERATOR uniques   :", df_fdata["TO_OPERATOR"].unique())
print("OFFERS uniques        :", df_fdata["OFFERS"].unique())
print("O (dans le modèle)    :", model.O.data())
some_a = next(iter(model.A))
for C in model.Cvec:
    val = value(model.f[some_a, C, 'o4G-ORANGE', 'o5G-ORANGE'])
    if val > 0:
        print("Transition non nulle vers o5G-ORANGE :", some_a, C, val)
        break
