import pyomo.environ as pyo
from itertools import product

model = pyo.ConcreteModel()


# 1. SETS
Zmax_schedule = {0 : 10 , 1 : 20 , 2 : 15 , 3 : 10 , 4 : 15}
QA_t = {0 : 10 , 1 : 20 , 2 : 15 , 3 : 10 , 4 : 15}

T=[0, 1, 2, 3, 4]  # horizon temporel
A=['A1', 'A2', 'A3']  # zones
I=['I', 'I1', 'I2']  # opérateurs
τ='I'  # notre opérateur
O=['O', 'O1', 'O2', 'NO', 'NO1', 'NO2']  # offres
Si = ['S1', 'S2', 'S3']  # sites de l’opérateur τ

model.T = pyo.Set(initialize=T)  # horizon temporel
model.A = pyo.Set(initialize=A)  # zones
model.I = pyo.Set(initialize=I)  # opérateurs
model.O = pyo.Set(initialize=O)  # offres !!! il faut mettre l'offre de chaque opérateur
model.S = pyo.Set(initialize=Si) # sites de l’opérateur i

ua0_dict = {(a,i,o): 1 for a in A for i in I for o in O}
DNG_dict = {t: 0.1 for t in T}
CAPANG_dict = {t: 100 for t in T}
u_a_dict = {'A1': 1000, 'A2': 1500, 'A3': 1200}
f_dict = {(a, c, o1, o2): 0.01 for a in A for c in C_space for o1 in O for o2 in O}
Rcomp_dict = {(t,a,i): 1 for t in T for a in A for i in I}
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

model.Zmax = pyo.Param(model.T , initialize=Zmax_schedule)  # nombre max de sites déployables par période
model.QA = pyo.Param(model.T , initialize = QA_t) # couverture minimale de la population par période
model.ua0 = pyo.Param(model.A, model.I, model.O , initialize =ua0_dict)
model.DNG = pyo.Param(model.T , initialize = DNG_dict)  # DNG dépend du temps
model.CAPANG = pyo.Param(model.T , initialize = CAPANG_dict) # DNG et CAPANG dépendent du temps
model.u_a = pyo.Param(model.A, within=pyo.NonNegativeIntegers , initialize = u_a_dict)  # utilisateurs totaux dans la zone a

model.Rcomp = pyo.Param(model.T, model.A, model.I, within=pyo.Binary , initialize = Rcomp_dict)
model.f = pyo.Param(model.A, model.Cvec, model.O, model.O , initialize = f_dict)

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


def delta_implication(m, t, a, C):

    # Départ : produit = 1
    expr = 1

    # opérateur τ (index 0)
    c_tau = C[0]
    expr *= (m.r[t,a] * c_tau + (1 - c_tau) * (1 - m.r[t,a]))

    # opérateurs concurrents (constantes uniquement)
    for k, i in enumerate(list(m.I)[1:], start=1):
        c_i = C[k]
        R = m.Rcomp[t,a,i]     # paramètre !
        expr *= (c_i * R + (1 - c_i) * (1 - R))

    return m.delta[t,a,C] == expr


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
    return m.u[t, a, τ, "NOτ"] == sum(m.u_site[t, a, s] for s in Sa_dict[a])
model.c_6 = pyo.Constraint(model.T, model.A, rule=assign_users)

# (7) capacité
def capacity(m, t, s):
    return sum(model.DNG * m.u_site[t, a, s] for a in As_dict[s]) <= model.CAPANG * m.z[t, s]
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
    return sum(m.u[T_end, a, τ, "NOτ"] for a in m.A)
model.obj = pyo.Objective(rule=objective, sense=pyo.maximize)
