# Analyse des Transactions Bancaires pour Identifier des Fraudes

Ce projet a pour objectif d’analyser des transactions bancaires afin d'identifier des schémas de fraude. L’analyse repose sur un ensemble de données contenant des transactions par carte de crédit effectuées par des titulaires européens en septembre 2013. Un tableau de bord interactif a également été développé avec Streamlit pour visualiser les résultats.

## Fonctionnalités
- Analyse exploratoire des données (EDA) pour comprendre la structure des transactions.
- Identification des transactions frauduleuses.
- Visualisation des données via un tableau de bord Streamlit.

## Données
L’ensemble de données, `creditcard.csv`, a été téléchargé depuis [Kaggle](https://www.kaggle.com/) et contient :
- **284 807 transactions**, dont **492 fraudes**.
- Variables anonymisées pour des raisons de confidentialité.

## Structure du Projet
- **`analyse.ipynb`** : Notebook contenant l'analyse exploratoire des données (EDA).
- **`app.py`** : Script Python pour le déploiement de l'application Streamlit.
- **`requirements.txt`** : Liste des dépendances Python nécessaires pour exécuter le projet.
- **`creditcard.csv/creditcard.csv`** : Fichier de données ( géré avec Git LFS).
- **`.gitignore`** : Fichier pour ignorer les fichiers sensibles ou volumineux lors du push.
- **`.gitattributes`** : Configuration de Git LFS pour gérer les fichiers volumineux.

## Prérequis
- Python 3.7 ou version ultérieure.
- Bibliothèques listées dans `requirements.txt`.

## Installation
1. Clonez le dépôt GitHub :
   ```bash
   git clone https://github.com/ernestdave12/Analyse_transaction_bancaire.git
   ```
2. Accédez au répertoire du projet :
   ```bash
   cd <nom-du-dossier>
   ```
3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation
### Exécution du Notebook
1. Ouvrez le fichier `analyse.ipynb` dans Jupyter Notebook ou JupyterLab.
2. Exécutez les cellules pour réaliser l'analyse exploratoire.

### Lancer l’application Streamlit
1. Exécutez le script `app.py` :
   ```bash
   streamlit run app.py
   ```
2. Accédez à l’application dans votre navigateur à l’adresse :
   [http://localhost:8501](http://localhost:8501)

### Application déployée
Vous pouvez explorer l’application déployée à cette adresse : [Analyse des Transactions Bancaires](https://analysetransactionbancaire.streamlit.app/).

## Crédits
- Dataset : [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
- Analyse et visualisation : Réalisées avec Python et Streamlit.


