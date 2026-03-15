# Posters LaTeX pour Makeitalive

Ce dossier contient les fichiers LaTeX pour générer les posters du projet Makeitalive.

## Fichiers

- `poster1.tex` : Poster général présentant le projet, la collecte de données et les approches
- `poster2.tex` : Poster technique détaillant l'architecture et l'implémentation
- `poster1.pdf` : Version compilée du poster 1 (99 KB)
- `poster2.pdf` : Version compilée du poster 2 (131 KB)
- `README.md` : Ce fichier

## Compilation

Les posters ont été compilés avec succès en utilisant pdflatex. Les packages LaTeX requis sont installés.

### Commande de compilation

```bash
pdflatex poster1.tex
pdflatex poster2.tex
```

### Avec latexmk (recommandé)

```bash
latexmk -pdf poster1.tex
latexmk -pdf poster2.tex
```

## Contenu des Posters

### Poster 1 : Présentation Générale
- Contexte et problématique du projet Makeitalive
- Collection de données : extraction de vidéos YouTube
- Approches techniques : Fine-tuning LoRa vs Motion Flow from scratch
- Résultats préliminaires et applications potentielles

### Poster 2 : Détails Techniques
- Architecture complète du MotionFlowUNet
- Code d'implémentation PyTorch
- Processus d'entraînement et configuration
- Algorithme de génération d'animation
- Comparaison quantitative des approches
- Métriques d'évaluation et défis techniques

## Technologies Utilisées

- **LaTeX avec tikzposter** : Pour la mise en page des posters
- **PyTorch** : Framework d'entraînement du modèle
- **OpenCV & SciPy** : Traitement d'images et warping
- **YouTube-dlp** : Téléchargement de vidéos sources

## Structure du Projet

Le projet Makeitalive comprend :
- Collecte de données depuis YouTube
- Entraînement de modèles de motion flow
- Génération d'animations à partir d'images statiques
- Évaluation qualitative et quantitative