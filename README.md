# Optimisation conjointe du déploiement des réseaux mobiles en Afrique

> Déploiement de la nouvelle génération (5G/NG) & stratégie énergétique d'un
> opérateur télécom en environnement concurrentiel.
>
> **PSC — Projet Scientifique Collectif — École Polytechnique × Orange**

Ce dépôt regroupe l'ensemble des modèles mathématiques, heuristiques et
approches d'intelligence artificielle développés pour déterminer la
**meilleure réponse** d'un opérateur (ici **Orange**) face aux stratégies de
déploiement de ses concurrents, sous contraintes de **budget**, de
**capacité**, de **couverture réglementaire** et — dans l'extension — d'**énergie**
et d'**empreinte carbone**.

Le rapport complet est disponible dans [`rapport/Optimisation_conjointe.pdf`](rapport/Optimisation_conjointe.pdf).

---

## 1. Contexte et problème

Le trafic mobile explose, porté par la multiplication des terminaux et les usages
gourmands en bande passante. En Afrique, moderniser le réseau (déployer la 5G) est
un levier stratégique de captation de parts de marché, mais il se heurte à des
budgets annuels limités, à des exigences réglementaires de couverture, et à des
contraintes d'infrastructure (énergie fiable) fortes.

On considère un marché concurrentiel :

| Élément | Notation | Description |
|---|---|---|
| Opérateurs | `I = {ORANGE, FREE MOBILE, BOUYGUES TELECOM, SFR}` | Orange = opérateur cible `τ` |
| Technologies legacy | `G = {2G, 3G, 4G}` | La 2G est conservée (service minimal) |
| Nouvelle génération | `NG` (5G) | Technologie à déployer |
| Zones géographiques | `A` | Chaque zone `a` a une population potentielle `uₐ` |
| Sites | `Sτ` | Sites de l'opérateur cible |
| Horizon | `T = {0, 1, …, |T|}` | Périodes de planification |

**Décision principale :** les variables binaires `zₛᵗ = 1` si la NG est installée
sur le site `s` à la période `t`. Les plans des concurrents `Rₐ,ᵢᵗ` sont supposés
**connus** (meilleure réponse). L'**objectif** est de maximiser la part de marché
NG d'Orange à la fin de l'horizon.

---

## 2. Démarche méthodologique

Le projet explore une **hiérarchie de méthodes de résolution**, du modèle exact
aux approches d'apprentissage, chacune répondant au compromis
*qualité de solution ↔ temps de calcul*.

```
MINLP initial ──► MILP initial ──► MILP reformulé          (modèles exacts)
                                        │
                                        ├──► Fix-and-Relax
                                        ├──► Algorithme Génétique (solveur / direct)
                                        ├──► Algorithme Mémétique
                                        ├──► MILP-GNN (warm-start par réseau de neurones sur graphe)
                                        └──► Apprentissage par Renforcement (PPO)

Extension : Optimisation conjointe déploiement–retrait legacy (énergie & CO₂)
```

### Résultats synthétiques

| Méthode | Qualité (% de l'optimal) | Gain de temps | Remarque |
|---|---|---|---|
| **MILP reformulé** | 100 % (exact) | −63 % vs MILP initial | −76 % de contraintes, −17 % de variables |
| **Fix-and-Relax** | — | inefficace | Brise les déductions du *presolve*, > 30 min |
| **AG direct** (glouton structuré) | ≈ 81,6 % | −87,7 % (550 zones) | Complexité polynomiale `O(GSA)` |
| **Algorithme mémétique** | ≈ 97,8 % | −83 % vs solveur | Recherche locale *first-improvement* |
| **MILP-GNN** (ratio 0,5) | ≈ 97,5 % | −71,6 % | ratio 0,8 → −91,7 % ; ratio 0,95 → infaisable |
| **RL — PPO** | 65,3 % de part de marché* | inférence en ~ms | *≈ 2× le glouton (34,6 %), mais viole QA dans 40 % des pas |

> Instance de référence : **Mayenne** (723 sites, 616 zones), fournie par Orange.

---

## 3. Structure du dépôt

Le dépôt est organisé pour suivre la progression du rapport. Chaque dossier
« exécutable » embarque une copie du jeu de données dont ses notebooks ont besoin,
de sorte qu'ils fonctionnent en lançant Jupyter **depuis ce dossier**.

```
.
├── rapport/                          # Rapport final (PDF)
│
├── data/                             # Jeu de données canonique : instance MAYENNE (723 sites, 616 zones)
├── Petite instance d'essai/          # Petite instance jouet (validation manuelle des modèles)
│
├── 0_exploration_donnees/            # Exploration / visualisation des fichiers CSV (lit ../data)
│
├── 1_modeles_milp/                   # § Modèles mathématiques
│   ├── MILP initial.ipynb            #   Linéarisation Big-M du MINLP
│   └── MILP reformulé.ipynb          #   Reformulation (–76 % de contraintes)
│
├── 2_heuristiques/                   # § Heuristiques & métaheuristiques
│   ├── Fix&Relax - AG.ipynb          #   Fix-and-Relax + algorithme génétique basé solveur
│   ├── Algorithmes_heuristiques_Work.ipynb  # AG direct + mémétique (évaluation sans solveur)
│   └── Heuristic_algorithms.ipynb    #   Première version des heuristiques
│
├── 3_milp_gnn/                       # § MILP-GNN
│   └── MILP GNN.ipynb                #   GraphSAGE pour prédire les variables à fixer (warm-start)
│
├── 4_apprentissage_renforcement/     # § Apprentissage par renforcement (PPO)
│   ├── ng_deployment_env.py          #   Environnement Gymnasium (MDP de déploiement)
│   ├── train_ppo.py                  #   Entraînement de l'agent PPO
│   └── RL_evaluate_ng_deployment.py  #   Évaluation vs baselines (Greedy, Random)
│
├── 5_instances_aleatoires/           # Génération & tests sur instances aléatoires (1ᵉ / 2ⁿᵈ modèle)
│
├── prototypes/                       # Travaux exploratoires & versions antérieures
│   ├── modeles_initiaux/             #   Premier / Deuxième modèle (+ données étendues)
│   ├── tests_pyomo/                  #   Prototypes Pyomo (dont données ANFR)
│   └── tests_petite_instance/        #   Tests de rectification sur petite instance
│
└── utils/                            # Utilitaires de chargement des données
    ├── importation_des_données.py    #   (à lancer depuis la racine du dépôt)
    └── test-petite-instance-donnees.py
```

---

## 4. Les modèles et méthodes

### 4.1 Modèles MILP (`1_modeles_milp/`)
Le problème est d'abord posé comme un **MINLP** (programme non linéaire en nombres
entiers mixtes). Deux non-linéarités sont traitées :
- la **contrainte de migration** des clients (produit binaire × continu) est
  linéarisée par la technique **Big-M** avec des bornes affinées ;
- la **contrainte de décodage de l'indicatrice** `δₐ,Cᵗ` (explosion combinatoire
  en `2^|I|`) est **éliminée** dans le *MILP reformulé* en exploitant le fait que
  la couverture concurrente `Rₐᵗ` est un paramètre fixé.

Ajouts de modélisation : dépendance temporelle de la demande `Dᴺᴳ` et de la
capacité `CAPAᴺᴳ`, **monotonie** du déploiement (`zₛᵗ ≥ zₛᵗ⁻¹`), relaxation de
l'intégrité des variables de population.

→ **−76 % de contraintes** (1 653 574 → 399 580) et **−63 % de temps** sur des
instances à 500 zones, à valeur objectif quasi identique (solveur **Gurobi**).

### 4.2 Heuristiques & métaheuristiques (`2_heuristiques/`)
- **Fix-and-Relax** — fenêtre glissante entière / futur relaxé / passé fixé.
  Peu efficace ici : relâcher le futur casse la contrainte de monotonie et donc
  le pouvoir de coupe du *presolve*.
- **Algorithme génétique** — deux versions :
  1. *basée solveur* (chaque individu = affectation `z`, fitness = sous-problème
     MILP) : fidèle mais intractable ;
  2. *directe* : encodage par **dates de premier déploiement**, **évaluation
     analytique sans solveur** (`z ⇒ r ⇒ u`), précalcul des trajectoires,
     mémoïsation, et **initialisation gloutonne structurée** (swap / time-shift /
     jitter).
- **Algorithme mémétique** — AG direct + **recherche locale first-improvement** et
  **mutations structurées** (`swap_periods`, `advance_best`, `delay_worst`).
  Meilleur compromis qualité/temps (≈ 97,8 % de l'optimal).

### 4.3 MILP-GNN (`3_milp_gnn/`)
Représentation en **graphe de sites** (un nœud par site, arête si deux sites
couvrent une zone commune ; features = degré + potentiel client). Un réseau
**GraphSAGE** à 3 couches prédit l'instant de déploiement de chaque site. Un
**ratio** de variables `zₛᵗ` les plus sûres est fixé pour donner un
**« démarrage à chaud »** au solveur. Compromis précision ↔ temps piloté par le
ratio (0,5 → excellent ; 0,95 → risque d'infaisabilité).

### 4.4 Apprentissage par renforcement (`4_apprentissage_renforcement/`)
Le déploiement est formulé comme un **MDP à horizon fini** et résolu par **PPO**
(*Proximal Policy Optimization*). L'agent observe l'état complet (déploiement,
couverture, distribution des abonnés, couverture concurrente, budget, écart
réglementaire) et produit un score par site ; les `Zᵗ` meilleurs sites sont
déployés. Une fois entraîné, il décide en quelques millisecondes. Limite : la
récompense pénalise mais ne garantit pas les contraintes (QA violée dans 40 % des
pas), et l'architecture MLP impose une taille d'instance fixe.

### 4.5 Extension énergétique (rapport, § 7)
Modèle d'**optimisation conjointe déploiement–retrait** `(PE)` : un second levier
de décision permet de **retirer** les couches legacy (3G/4G) pour transférer leur
enveloppe énergétique vers la NG, sous **budget énergétique** `Eₜᵐᵃˣ` et
**plafond carbone** `CO₂,ₜᵐᵃˣ`, avec redistribution des clients évincés. Formulé
dans le rapport (implémentation en perspective).

---

## 5. Les données (instance Mayenne)

Les données proviennent de l'instance **Mayenne** (723 sites, 616 zones) fournie
par les tuteurs Orange. Correspondance fichiers ↔ paramètres du modèle :

| Fichier source | Paramètre(s) | Description |
|---|---|---|
| `AREAS.csv` | `u⁰ₐ,ᵢ,ₒ`, `uₐ` | Population initiale par zone, opérateur et offre |
| `EXISTING_SITES.csv` | `Sτ` | Sites de l'opérateur cible (+ état 3G/4G/5G) |
| `AREAS_SITES_LINK.csv` | `Sₐ,τ`, `Aₛ` | Lien zones ↔ sites (couverture géographique) |
| `COMPETITORS_STRATEGY.csv` | `Rₐ,ᵢᵗ` | Couverture NG des concurrents par zone & période |
| `DEMAND.csv` | `Dᴺᴳᵗ` | Demande de trafic 5G par période |
| `CAPACITY.csv` | `CAPAᴺᴳᵗ` | Capacité d'un site NG par période |
| `OPERATIONAL_LIMITS.csv` | `Z̄ᵗ` | Budget de déploiement par période |
| `STRATEGIC_GUIDELINES.csv` | `QAᵗ` | Cible réglementaire de couverture minimale |
| `UPGRADE_FUNCTION.csv` | `fₐ,C,o′,o` | Fonction de migration entre offres selon le contexte de couverture |

Une **petite instance jouet** (`Petite instance d'essai/`) sert à valider
manuellement les modèles.

---

## 6. Prise en main

### Dépendances principales
Les notebooks et scripts reposent sur l'écosystème Python scientifique :

- `pandas`, `numpy` — manipulation des données
- `pyomo` — modélisation MILP, avec un solveur (**Gurobi**, ou open-source
  **HiGHS** / **GLPK**)
- `torch`, `torch-geometric` — GNN (GraphSAGE)
- `gymnasium`, `stable-baselines3` — apprentissage par renforcement (PPO)
- `matplotlib` — visualisations

### Exécuter un modèle / une heuristique
Les notebooks lisent leurs CSV **par nom de fichier simple** (ex. `AREAS.csv`).
Chaque dossier exécutable contient déjà sa propre copie du jeu de données :

```bash
cd 1_modeles_milp        # ou 2_heuristiques, 3_milp_gnn, 5_instances_aleatoires
jupyter notebook         # ouvrir puis exécuter le notebook souhaité
```

### Lancer l'apprentissage par renforcement
Les scripts RL prennent le jeu de données en argument (`--data_dir`) :

```bash
cd 4_apprentissage_renforcement
python train_ppo.py                              # entraînement (ajuster data_dir dans le script)
python RL_evaluate_ng_deployment.py --data_dir ../data --episodes 10
```

### Utilitaire de chargement
`utils/importation_des_données.py` charge la petite instance ; à lancer **depuis
la racine du dépôt** (chemins relatifs à `Petite instance d'essai/`).

---

## 7. Équipe

**Auteurs** — Labgoul Anas · Oujaa Haitam Yassine · Takfa Anass ·
Belfatmi Ayoub · Ait Mansour Abderrahmane

**Tuteurs (Orange / École Polytechnique)** — Matthieu Chardy · Amal Benhamiche ·
Youssouf Hadhbi · Aurélien Bechler

---

## 8. Références principales

1. A. Cambier, M. Chardy, R. Figueiredo, A. Ouorou, M. Poss — *Optimizing the
   investments in mobile networks and subscriber migrations for a telecommunication
   operator*, Networks, 77(4):495–519, 2021.
2. M. Chardy, M. Ben Yahia, Y. Bao — *3G/4G load-balancing optimization for mobile
   network planning*, 2016.
3. A. Benhamiche, M. Chardy, B. Mebrek — *Modelling the mobile investment strategies
   under competition using mathematical programming*.
4. W. Hamilton, R. Ying, J. Leskovec — *Inductive Representation Learning on Large
   Graphs (GraphSAGE)*, 2017.
5. M. Gasse, D. Chételat, N. Ferroni, L. Charlin, A. Lodi — *Exact Combinatorial
   Optimization with Graph Convolutional Neural Networks*, 2019.
6. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov — *Proximal Policy
   Optimization Algorithms*, 2017.
7. P. Zappalà — *Méthodes de résolution des jeux en forme extensive avec application
   au marché des réseaux mobiles*, Thèse, Avignon Université, 2024.

> La bibliographie complète figure dans le rapport.
