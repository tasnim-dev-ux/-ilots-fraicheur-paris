# 🌿 Analyse des Îlots de Fraîcheur à Paris
### Test Data Scientist — Open Data Ville de Paris

> **Question centrale :** Paris garantit-elle une équité d'accès aux îlots de fraîcheur selon les arrondissements ?

---

## 📌 Présentation du projet

Ce projet analyse la distribution des équipements dits "îlots de fraîcheur" 
(fontaines Wallace, parcs, piscines, brumisateurs...) dans les 20 arrondissements 
de Paris, en croisant les données Open Data de la Ville de Paris avec les données 
de population INSEE 2021.

L'objectif est de **prouver ou infirmer** l'hypothèse suivante :

- **H0** : La distribution est proportionnelle à la population (équité)
- **H1** : Il existe une inégalité structurelle de distribution

---

## 🗂️ Structure du projet

```
📁 ilots-fraicheur-paris/
│
├── 📓 notebook/
│   └── ilots_fraicheur_paris.ipynb     ← Analyse complète Python
│
├── 🖥️ streamlit_app/
│   └── app.py                          ← Interface interactive Streamlit
│
├── 📊 presentation/
│   └── presentation_ilots_paris.pptx   ← Slides de présentation
│
├── 🗺️ cartes/
│   ├── carte_ilots_types.html          ← Carte interactive par type
│   ├── carte_clusters.html             ← Carte clustering géographique
│   └── carte_heatmap_densite.html      ← Heatmap densité
│
├── 📈 exports/
│   ├── ratio_ilots_par_arrondissement.png
│   ├── heatmap_type_arrondissement.png
│   └── methode_du_coude.png
│
├── requirements.txt
└── README.md
```

---

## 🔬 Méthodologie

1. **Collecte** via l'API officielle Open Data Paris (pagination automatique)
2. **Nettoyage** : suppression des doublons, valeurs manquantes, coordonnées invalides
3. **Croisement** avec les données de population INSEE 2021
4. **Calcul** du ratio équipements / 10 000 habitants par arrondissement
5. **Tests statistiques** : corrélation de Spearman, test du Chi², indice de Gini
6. **Clustering géographique** KMeans (nombre de clusters optimisé par méthode du coude)
7. **Visualisations** : carte choroplèthe, heatmap, crosstab type × arrondissement
8. **Conclusion chiffrée** + recommandations pour la Ville de Paris

---

## 🖥️ Application Streamlit

Une interface interactive a été développée avec **Streamlit** pour explorer 
les résultats de manière dynamique sans avoir à exécuter le notebook.

### Lancer l'application

```bash
# 1. Cloner le repo
git clone https://github.com/TON_USERNAME/ilots-fraicheur-paris.git
cd ilots-fraicheur-paris

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'interface Streamlit
streamlit run streamlit_app/app.py
```

L'application s'ouvre automatiquement dans le navigateur sur `http://localhost:8501`

### Fonctionnalités de l'interface
- 🗺️ Carte interactive des équipements (filtre par type et arrondissement)
- 📊 Graphique du ratio équipements / habitant par arrondissement
- 🔬 Résultats des tests statistiques en langage clair
- 🌡️ Heatmap de la composition des équipements
- 📍 Visualisation des clusters géographiques

---

## 📓 Notebook Jupyter

Pour exécuter l'analyse complète :

```bash
# Installer Jupyter si nécessaire
pip install jupyter

# Lancer le notebook
jupyter notebook notebook/ilots_fraicheur_paris.ipynb
```

Exécuter toutes les cellules dans l'ordre : **Kernel → Restart & Run All**

> ⚠️ La cellule 7 (fusion avec les données INSEE) peut nécessiter une adaptation  
> du format des noms d'arrondissements selon le dataset récupéré par l'API.

---

## 📦 Installation des dépendances

```bash
pip install -r requirements.txt
```

### requirements.txt
```
requests>=2.28.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.9.0
scikit-learn>=1.1.0
folium>=0.14.0
streamlit>=1.25.0
```

---

## 📊 Source des données

| Source | Description | Lien |
|--------|-------------|------|
| Open Data Paris | Îlots de fraîcheur — équipements & activités | [opendata.paris.fr](https://opendata.paris.fr/explore/dataset/ilots-de-fraicheur-equipements-activites) |
| INSEE | Population légale 2021 par arrondissement | [insee.fr](https://www.insee.fr/fr/statistiques/2011101) |

---

## 🔑 Résultats clés

> *Les valeurs ci-dessous sont à mettre à jour après exécution du notebook*

- **Test Chi²** : p-value < 0.05 → H0 rejetée, inégalité statistiquement significative
- **Corrélation de Spearman** : ρ = X.XX → corrélation [faible/modérée] entre population et nb d'îlots
- **Indice de Gini** : 0.XX → inégalité [modérée/significative]
- **Écart max/min** : l'arrondissement le mieux doté dispose de X fois plus d'îlots/habitant que le moins doté
- **X arrondissements sur 20** se situent sous la moyenne parisienne

---

## 💡 Recommandations principales

1. **Renforcer en priorité** les arrondissements du Flop 3 (ratio le plus faible)
2. **Équiper les zones blanches** identifiées par le clustering géographique
3. **Diversifier les types** d'équipements dans les arrondissements mono-dépendants
4. **Créer un indicateur annuel** ratio/habitant pour suivre la progression

---

## ⚠️ Limites de l'étude

- Le dataset reflète uniquement les équipements **référencés** par la Ville de Paris
- Un équipement n'est pas équivalent à un autre (capacité d'accueil non prise en compte)
- La proximité entre arrondissements n'est pas modélisée (un équipement peut servir plusieurs arrondissements)
- Données de population issues du recensement 2021

---

## 👤 Auteur

Réalisé dans le cadre d'un **test Data Scientist**  
Données : Open Data Ville de Paris — Mai 2026
