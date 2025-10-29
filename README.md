# MayGraphKNN

Ce projet a été réalisé dans le cadre de mon stage de 4ᵉ année au Japon, chez F-Carrer K.K..

L’objectif principal était de développer un algorithme capable de proposer la meilleure entreprise ou équipe pour un candidat, en utilisant une approche de KNN (K-Nearest Neighbors) adaptée à la gestion de données et de graphes.

# Partie technique

Le projet utilise des graphes pour représenter les relations entre candidats, équipes et entreprises, et applique l’algorithme KNN pour identifier les correspondances les plus pertinentes. Les étapes principales incluent :

* Prétraitement des données : normalisation et vectorisation des caractéristiques des candidats et des équipes.

* Construction du graphe : chaque nœud représente un candidat ou une équipe, les liens indiquent les similarités ou compatibilités.

* Application du KNN : pour chaque candidat, le modèle identifie les équipes/entreprises les plus proches dans l’espace des caractéristiques.

* Évaluation : mesure de la pertinence des recommandations via des métriques adaptées (précision, rappel, score global).

Le projet a été développé en Python, avec un focus sur la modularité et la réutilisabilité du code, permettant d’adapter facilement l’algorithme à de nouvelles données ou à d’autres cas d’usage.

Voici un lien vers un serveur de test, connecté à une base de données contenant des données fictives. Ce serveur sert d’aperçu et de test avant le déploiement définitif : https://maygraphknn.onrender.com
