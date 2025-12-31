# ⚡ EcoVolt : Prévision de la Demande Énergétique (Smart Grid)

## 📖 Contexte du Projet

**EcoVolt** est une start-up opérant un réseau intelligent de distribution d'électricité (**Smart Grid**). Pour optimiser la gestion des bornes de recharge de véhicules électriques et éviter les surcharges du réseau, l'entreprise doit anticiper les pics de consommation.

**La Mission :** Concevoir une solution d'Intelligence Artificielle capable de prédire la consommation électrique globale à l'heure **`t+1`** en se basant sur l'historique des **24 dernières heures** et le mix énergétique (Nucléaire, Eolien, Solaire, Hydro, etc.).

---

## 🏗️ Architecture du Projet

Ce projet démontre une approche professionnelle structurée en deux phases distinctes :

1. **Exploration (Lab) :** Analyse exploratoire, ingénierie des fonctionnalités et création d'une "Baseline" (Random Forest).
2. **Industrialisation (Prod) :** Pipeline Deep Learning modulaire et réutilisable.

```text
EcoVolt_Project/
│
├── data/
│   └── df_LSTM.csv          # Données nettoyées (Granularité horaire)
│
├── notebooks/
│   └── ml_lab.ipynb         # EDA, Corrélation & Baseline ML (Random Forest)
│
├── src/                     # Code source modulaire (Deep Learning)
│   ├── __init__.py
│   ├── data_loader.py       # Ingestion, Scaling (-1, 1), Windowing (24h)
│   └── lstm_model.py        # Architecture du Réseau de Neurones LSTM
│
├── main.py                  # Orchestrateur (Entraînement & Évaluation)
├── requirements.txt         # Dépendances du projet
└── README.md                # Documentation

```

---

## 🚀 Installation & Exécution

### Prérequis

* Python 3.11 (Recommandé pour compatibilité TensorFlow)
* Environnement virtuel (`venv`)

### 1. Cloner et Installer

```bash
# Cloner le dépôt
git clone https://github.com/votre-username/EcoVolt-LSTM.git
cd EcoVolt-LSTM

# Créer l'environnement virtuel (Windows)
python -m venv venv
.\venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

```

### 2. Lancer l'Analyse Exploratoire (Baseline)

Ouvrez le notebook pour visualiser les données et exécuter le modèle Random Forest :

```bash
jupyter notebook notebooks/ml_lab.ipynb

```

### 3. Lancer l'Entraînement Deep Learning (LSTM)

Exécutez le script principal pour lancer le pipeline complet :

```bash
python main.py

```

---

## 📊 Résultats & Performances

Nous avons comparé une approche classique (Machine Learning) avec notre modèle Deep Learning sur le jeu de test (20% des données, soit ~1 an d'historique).

### 1. Tableau Comparatif (RMSE)

Le modèle LSTM réduit l'erreur de prédiction de plus de **50%** par rapport à la méthode classique.

| Modèle | RMSE (Erreur Moyenne) | Performance |
| --- | --- | --- |
| **Random Forest (Baseline)** | **436 MW** | Référence |
| **LSTM (Deep Learning)** | **~206 MW** | **+53% de précision** |

### 2. Convergence de l'Entraînement

Le modèle converge rapidement (15 époques) sans signes de surapprentissage, grâce à l'utilisation du **Dropout**.

*Extrait des logs d'exécution réels :*

```text
Epoch 1/15
1219/1219 ━━━━━━━━━━━━━━━━━━━━ 20s 15ms/step - loss: 0.0111 - val_loss: 0.0031
...
Epoch 15/15
1219/1219 ━━━━━━━━━━━━━━━━━━━━ 8s 7ms/step - loss: 0.0017 - val_loss: 0.0013

```

> **Note :** Le `val_loss` (Test) est inférieur au `loss` (Train), ce qui confirme une excellente généralisation du modèle.

### 3. Visualisation

Le modèle LSTM capture parfaitement les dynamiques temporelles, notamment les pics de consommation matinaux et nocturnes que les modèles linéaires peinent à anticiper.

*Vous pouvez aussi la trouver dans `output/Graph - LSTM Results [Actual vs Predicted] (-1,1).png`.*
![alt text](<Graph - LSTM Results [Actual vs Predicted] (-1,1).png>)

---

## 🧠 Choix Techniques Justifiés

Ce projet valide les compétences techniques de niveau 2/3 (Adapter/Transposer).

### 1. Stratégie de Données (Time Series)

* **Windowing (Fenêtrage) :** Transformation des données en séquences glissantes de **24 heures**. Cela permet au modèle de "voir" une journée complète avant de prédire l'heure suivante.
* **Split Chronologique :** Pas de mélange aléatoire (`shuffle=False`) pour respecter la causalité temporelle et éviter le *Data Leakage*.

### 2. Normalisation Spécifique

* **Choix :** `MinMaxScaler(feature_range=(-1, 1))`.
* **Justification :** Les cellules LSTM utilisent par défaut la fonction d'activation **tanh** (tangente hyperbolique), dont la sortie est comprise entre -1 et 1. Aligner les données d'entrée sur cette plage facilite la convergence.

### 3. Architecture du Modèle

* **LSTM (64 unités) :** Capable de retenir les dépendances à long terme (ex: impact de la production solaire de la veille).
* **Dropout (0.2) :** Désactive aléatoirement 20% des neurones pendant l'entraînement pour forcer le réseau à apprendre des motifs robustes.

