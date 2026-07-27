import pandas as pd
import numpy as np
import math
from itertools import product


# 1. SETS

T=[0, 1, 2, 3, 4, 5]  # horizon temporel
A = list(pd.read_csv("Petite instance d'essai/areas_instance.csv", sep=";")["AREAS"])  # zones
I=['ORANGE','FREE MOBILE', 'BOUYGUES TELECOM"','SFR']  # opérateurs
τ='ORANGE'  # notre opérateur
O=['o3G-FREE MOBILE', 'o3G-BOUYGUES TELECOM', 'o3G-SFR','o3G-ORANGE','o4G-FREE MOBILE','o4G-BOUYGUES TELECOM','o4G-SFR','o4G-ORANGE','o5G-FREE MOBILE','o5G-BOUYGUES TELECOM','o5G-SFR','o5G-ORANGE']  # offres
O_i = {'ORANGE': ['o3G-ORANGE','o4G-ORANGE','o5G-ORANGE'],
        'FREE MOBILE': ['o3G-FREE MOBILE','o4G-FREE MOBILE','o5G-FREE MOBILE'],
        'BOUYGUES TELECOM"': ['o3G-BOUYGUES TELECOM','o4G-BOUYGUES TELECOM','o5G-BOUYGUES TELECOM'],
        'SFR': ['o3G-SFR','o4G-SFR','o5G-SFR']}  # offres par opérateur
Si = list(pd.read_csv("Petite instance d'essai/Existing_sites_instance.csv", sep=";")["EXISTING_SITES"]) # sites de l’opérateur τ

# Couples utiles
C_space = list(product([0,1], repeat=len(I)))  # Toutes les combinaisons de couverture
df = pd.read_csv("Petite instance d'essai/Areas_sites_link_instance.csv", sep=";")   # contient colonnes "a" et "s"

Sa_dict = {}

for _, row in df.iterrows():
    zone = row["AREAS"]
    site = row["SITES"]

    if zone not in Sa_dict:
        Sa_dict[zone] = []

    Sa_dict[zone].append(site)

df = pd.read_csv("Petite instance d'essai/Areas_sites_link_instance.csv", sep=";")   # contient colonnes "a" et "s"

As_dict = {}

for _, row in df.iterrows():
    zone = row["AREAS"]
    site = row["SITES"]

    if site not in As_dict:
        As_dict[site] = []

    As_dict[site].append(zone)


# 2. PARAMÈTRES
df_zmax = pd.read_csv("Petite instance d'essai/OPERATIONAL_LIMITS.csv", sep=";")
Zmax_data = {row.TIME_SLOTS: row["MAX_NUMBER_OF_DEPLOYMENTS"] for _,row in df_zmax.iterrows()}

df_qa = pd.read_csv("Petite instance d'essai/STRATEGIC_GUIDELINES.csv", sep=";")
QA_data = {row.TIME_SLOTS: row["QA"] for _,row in df_qa.iterrows()}

df = pd.read_csv("Petite instance d'essai/areas_instance.csv", sep=";")

ua0_data = {}
for _, row in df.iterrows():
    for col in df.columns [1:]:
        ua0_data[int(row["AREAS"]),col.split("-")[1], col] = math.floor(row[col])
        
df_dng = pd.read_csv("Petite instance d'essai/DEMAND.csv", sep=";")

DNG_data = {row.TIME_SLOTS: row["5G"] for _,row in df_dng.iterrows()}

df_capang = pd.read_csv("Petite instance d'essai/CAPACITY.csv", sep=";")
CAPANG_data = {row.TIME_SLOTS: row["5G"] for _,row in df_capang.iterrows()}

df_ua = pd.read_csv("Petite instance d'essai/areas_instance.csv", sep=";")
u_a_data = {}
for _, row in df.iterrows():
    somme =0
    for col in df.columns [1:]:
        somme += math.floor(row[col])
    u_a_data [row["AREAS"]]= somme


df_Rcomp = pd.read_csv("Petite instance d'essai/COMPETITORS_STRATEGY_instance.csv", sep=";")
Rcomp_data = {}
for _, row in df_Rcomp.iterrows():
    for col in df_Rcomp.columns[2:5]:
        Rcomp_data[row["TIME_SLOTS"],row["AREAS"], col] = 1 * row[col]

df_fdata = pd.read_csv("Petite instance d'essai/UPGRADE_FUNCTION.csv", sep=";")

f_data = {}
for _, row in df_fdata.iterrows():
    C = (row["ORANGE"],row["FREE MOBILE"],row["BOUYGUES TELECOM"],row["SFR"])
    O = row["OFFERS"]+"-"+row["FROM_OPERATOR"]
    O2 = "o5G"+"-"+row["TO_OPERATOR"]
    f_data[(C, O, O2)] = row["PERCENTAGES"]
    
print(A)