# Scoring LCL


Ce projet se fait dans le cadre d'un challenge proposé par Mohamed Sneiba Hamoud. Le but est de challenger le modèle de scoring existant dans l'octroi de crédit immobilier.
Les milestones de sont projets sont les suivantes: 
- Vision globale de la base de données: étude de la période + constitution des échantillons
- Preprocessing sur l'ensemble des variables (valeurs manquantes, traitement de types, etc..) + réflexion nouvelles variables
- Sélection de variables (Lasso, tests univariés) 
- Discrétisation des variables quantitatives + regroupement de modalités des variables qualitatives -
- Sélection post-discrétisation: Random Forest + stabilité temporelle
- Modélisation classique: régression logistique + performances
- Modélisation alternative: modèle challengeant
- Grille de score



## Installer le projet
 Voici un petit guide pour vous expliquer comment accéder au contenu du projet:
 
Créer un répertoire à l'endroit souhaité

Sur terminal:
```
cd "[chemin d'accès du répertoire, par exemple: /Users/luciegabagnou/Documents/MOSEF/SCORING/projet]

```
Deux options pour récupérer le projet: 

- Via Github: Après s'être placé dans ce dossier via la commande cd, on va initialiser le dépôt git en local associé au projet sur Github 

```
git clone https://github.com/luciegaba/scoring_project.git
git pull 
```

- Via Dossier transmis: on va simplement dézipper le dossier LCL

Configurer l'environnement du projet:


Créer un environnement virtuel 

```
conda create -n"[nom de l'environnemnt, exemple: LCL]
conda activate [nom de l'environnement]
conda install python 
conda install pip
pip install -r requirements.txt
```

Ensuite, vous pouvez aisément accéder au(x) notebook(s)
Il suffit juste de dérouler le notebook 
