# BioResistanceAI

Prédiction de la résistance aux antibiotiques à partir de données génomiques et transcriptomiques de bactéries.

## Objectif

Ce projet vise à prédire si une souche bactérienne est résistante ou non à différents antibiotiques, à partir de :

- Gènes de résistance (`X_gpa`)
- Mutations (`X_snps`)
- Expression génétique (`X_genexp`)

La cible est un phénotype binaire (`résistant` / `susceptible`) pour 5 antibiotiques.

## Pipeline

1. Nettoyage et fusion des données
2. Feature engineering
3. Entraînement de plusieurs modèles (RF, SVM, XGBoost, MLP)
4. Évaluation avec validation croisée (macro recall)
5. Visualisation et comparaison des performances

## Structure

