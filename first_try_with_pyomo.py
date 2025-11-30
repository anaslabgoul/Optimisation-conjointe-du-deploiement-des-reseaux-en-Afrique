import pyomo.environ as pyo
from itertools import product
import numpy as np

model = pyo.ConcreteModel()


# 1. SETS

T=[0, 1, 2, 3]  # horizon temporel
A=['A1', 'A2', 'A3']  # zones
I=['I', 'I1']  # opérateurs
τ='I'  # notre opérateur
O=['O1', 'O2', 'NO']  # offres
Si = ['S1', 'S2', 'S3']  # sites de l’opérateur τ

model.T = pyo.Set(initialize=T)  # horizon temporel
model.A = pyo.Set(initialize=A)  # zones
model.I = pyo.Set(initialize=I)  # opérateurs
model.O = pyo.Set(initialize=O)  # offres !!! il faut mettre l'offre de chaque opérateur
model.S = pyo.Set(initialize=Si) # sites de l’opérateur i

# Couples utiles
C_space = list(product([0,1], repeat=len(I)))  # Toutes les combinaisons de couverture
Sa_dict = {  # mapping a → {sites}
    'A1': ['S1', 'S2'],
    'A2': ['S2', 'S3'],
    'A3': ['S1', 'S3'],}
As_dict = {  # mapping s → {zones}
    'S1': ['A1', 'A3'],
    'S2': ['A1', 'A2'],
    'S3': ['A2', 'A3']}


model.Cvec = pyo.Set(initialize=C_space)   # Toutes les combinaisons de couverture
model.Sa = Sa_dict                         # mapping a → {sites}
model.As = As_dict                         # mapping s → {zones}

# 2. PARAMÈTRES

Zmax_data = {0: 2, 1: 2, 2: 2, 3: 2}
model.Zmax = pyo.Param(model.T, initialize = Zmax_data)  # nombre max de sites déployables par période

QA_data = {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2}
model.QA = pyo.Param(model.T, initialize = QA_data) # couverture minimale de la population par période

ua0_data = {('A1', 'I', 'O1'): 50,
            ('A1', 'I', 'O2'): 100,
            ('A1', 'I', 'NO'): 0,
            ('A1', 'I1', 'O1'): 50,
            ('A1', 'I1', 'O2'): 100,
            ('A1', 'I1', 'NO'): 0,
            ('A2', 'I', 'O1'): 50,
            ('A2', 'I', 'O2'): 100,
            ('A2', 'I', 'NO'): 0,
            ('A2', 'I1', 'O1'): 50,
            ('A2', 'I1', 'O2'): 100,
            ('A2', 'I1', 'NO'): 0,
            ('A3', 'I', 'O1'): 50,
            ('A3', 'I', 'O2'): 100,
            ('A3', 'I', 'NO'): 0,
            ('A3', 'I1', 'O1'): 50,
            ('A3', 'I1', 'O2'): 100,
            ('A3', 'I1', 'NO'): 0,}
model.ua0 = pyo.Param(model.A, model.I, model.O, initialize=ua0_data)  # utilisateurs initiaux

DNG_data = {0: 1, 1: 1, 2: 1, 3: 1}
model.DNG = pyo.Param(model.T, initialize = DNG_data)  # DNG dépend du temps

CAPANG_data = {0: 200, 1: 200, 2: 200, 3: 200}
model.CAPANG = pyo.Param(model.T, initialize = CAPANG_data) # DNG et CAPANG dépendent du temps

u_a_data = {'A1': 1000, 'A2': 1500, 'A3': 2000}
model.u_a = pyo.Param(model.A, initialize = u_a_data)  # utilisateurs totaux dans la zone a

Rcomp_data = {}
for t in model.T:
    for a in model.A:
        for i in model.I:
            if i != τ:
                if t==0:
                    Rcomp_data[(t, a, i)] = 0  # pas de couverture initiale des autres opérateurs
                else:
                    if Rcomp_data[(t-1, a, i)] == 1:
                        Rcomp_data[(t, a, i)] = 1 # une fois couvert, toujours couvert
                    else:
                        Rcomp_data[(t, a, i)] = np.random.randint(0,2)
model.Rcomp = pyo.Param(model.T, model.A, model.I, initialize=Rcomp_data)  # couverture des autres opérateurs

f_data = {}
for a in model.A:
    for C in model.Cvec:
        for o1 in model.O:
            for o2 in model.O:
                f_data[(a, C, o1, o2)] =  np.random.rand()
model.f = pyo.Param(model.A, model.Cvec, model.O, model.O, initialize=f_data)  # taux de migration

# 3. VARIABLES

model.z = pyo.Var(model.T, model.S, within=pyo.Binary)
model.r = pyo.Var(model.T, model.A, within=pyo.Binary)
model.delta = pyo.Var(model.T, model.A, model.Cvec, within=pyo.Binary)

model.u = pyo.Var(model.T, model.A, model.I, model.O, within=pyo.NonNegativeIntegers)
model.u_site = pyo.Var(model.T, model.A, model.S, within=pyo.NonNegativeIntegers)  # on ne définit pas l'opérateur et l'offre car on ne parle que de notre opérateur et la NG

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
        ck = C[k]
        R = m.Rcomp[t, a, i]
        res *= (R * ck + (1 - ck) * (1 - R))

    return m.delta[t, a, C] == res 

model.c_4 = pyo.Constraint(model.T, model.A, model.Cvec, rule=delta_implication)

# (5) Migration / churn
def migration(m, t, a, i, o):
    if t == min(m.T):  # initialisation
        return m.u[t, a, i, o] == model.ua0[a, i, o]

    return m.u[t, a, i, o] == sum(
        m.delta[t-1, a, C] * sum(
            model.f[a, C, o_prev, o] * m.u[t-1, a, i_prev, o_prev]
            for i_prev in m.I for o_prev in m.O
        )
        for C in model.Cvec
    )
model.c_5 = pyo.Constraint(model.T, model.A, model.I, model.O, rule=migration)

# (6) u_NO = somme sur les sites
def assign_users(m, t, a):
    return m.u[t, a, τ, "NO"] == sum(m.u_site[t, a, s] for s in Sa_dict[a])
model.c_6 = pyo.Constraint(model.T, model.A, rule=assign_users)

# (7) capacité
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

# 5. OBJECTIF

def objective(m):
    T_end = max(m.T)
    return sum(m.u[T_end, a, τ, "NO"] for a in m.A)
model.obj = pyo.Objective(rule=objective, sense=pyo.maximize)

model.write('model.lp', io_options={'symbolic_solver_labels': True})