import pyomo.environ as pyo
from itertools import product
import numpy as np

model = pyo.ConcreteModel()


# 1. SETS

from importation_des_données import T, A, I, O, O_i, Si 

model.T = pyo.Set(initialize=T)  # horizon temporel
model.A = pyo.Set(initialize=A)  # zones
model.I = pyo.Set(initialize=I)  # opérateurs
τ = 'ORANGE'  # notre opérateur
model.O = pyo.Set(initialize=O)  # offres !!! il faut mettre l'offre de chaque opérateur
model.O_i = O_i  # offres par opérateur
model.S = pyo.Set(initialize=Si) # sites de l’opérateur i

# Couples utiles
from importation_des_données import C_space, Sa_dict, As_dict


model.Cvec = pyo.Set(initialize=C_space)   # Toutes les combinaisons de couverture
model.Sa = Sa_dict                         # mapping a → {sites}
model.As = As_dict                         # mapping s → {zones}

# 2. PARAMÈTRES
from importation_des_données import Zmax_data, QA_data, ua0_data, DNG_data, CAPANG_data, u_a_data, Rcomp_data, f_data

model.Zmax = pyo.Param(model.T, initialize = Zmax_data)  # nombre max de sites déployables par période

model.QA = pyo.Param(model.T, initialize = QA_data) # couverture minimale de la population par période

model.ua0 = pyo.Param(model.A, model.I, model.O, initialize=ua0_data)  # utilisateurs initiaux

model.DNG = pyo.Param(model.T, initialize = DNG_data)  # DNG dépend du temps

model.CAPANG = pyo.Param(model.T, initialize = CAPANG_data) # DNG et CAPANG dépendent du temps

model.u_a = pyo.Param(model.A, initialize = u_a_data)  # utilisateurs totaux dans la zone a

model.Rcomp = pyo.Param(model.T, model.A, model.I, initialize=Rcomp_data)  # couverture des autres opérateurs

model.f = pyo.Param(model.Cvec, model.O, model.O, initialize=f_data)  # taux de migration (ne dépend pas de la zone selon les données)

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




# (5) Migration non-linéaire
# --- Ajout des variables auxiliaires et paramètres M ---
# Note: suppose que model.ua0 est disponible comme borne supérieure pour u (tu l'as déjà)
# On crée un param M[a,C,o]
def compute_M():
    M = {}
    for a in model.A:
        for C in model.Cvec:
            for o in model.O:
                M_val = 0.0
                for i_prev in model.I:
                    for o_prev in model.O_i[i_prev]:
                        M_val += pyo.value(model.f[a, C, o_prev, o]) * pyo.value(model.u_a[a])
                M[(a, C, o)] = M_val
    return M

M_data = compute_M()

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
    if t == min(m.T):
        return m.u[t, a, i, o] == pyo.value(m.ua0[a, i, o])
    return m.u[t, a, i, o] == sum(m.y[t-1, a, C, o] for C in m.Cvec)
model.c_5_lin = pyo.Constraint(model.T, model.A, model.I, model.O, rule=migration_lin)




# (6) u_NO = somme sur les sites
def assign_users(m, t, a, o):
    return m.u[t, a, τ, o] == sum(m.u_site[t, a, s] for s in Sa_dict[a])
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

# Create solver
solver = pyo.SolverFactory('glpk')

# Solve
results = solver.solve(model, tee=True)

# Display solver status
print("\nSolver status:", results.solver.status)
print("Termination condition:", results.solver.termination_condition)