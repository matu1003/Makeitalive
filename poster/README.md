# Posters LaTeX pour Makeitalive

Ce dossier contient les sources LaTeX pour la présentation du projet Makeitalive.

---

## 🛠️ Comment modifier les Posters

Voici un guide pour personnaliser le contenu une fois le fichier `.tex` ouvert (sur Overleaf ou en local).

### 1. Structure et Colonnes
Les posters sont configurés en **Portrait** (A0) avec **2 colonnes**.
- La largeur est gérée par `\column{0.5}`. 
- Pour ajuster la répartition (ex: une colonne plus large), changez par exemple en `\column{0.4}` et `\column{0.6}`.

### 2. Style Visuel
- **Thème** : `\usetheme{Rays}` définit le look global. Autres options : `Wave`, `Simple`, `Default`.
- **Couleurs** : `\usecolorstyle{Default}`. Vous pouvez tester `BlueGrayOrange`, `GreenMarine` ou `RedWhiteBlue`.

### 3. Ajouter/Modifier du Contenu
Chaque section est un bloc :
```latex
\block{Titre de la section}{
    Votre texte ici...
}
```

### 4. Schémas et Dessins (TikZ)
Le poster 1 contient un schéma de l'U-Net.
- Cherchez `\begin{tikzpicture}`.
- `node distance=1.5cm` gère l'espacement vertical entre les couches.
- `scale=0.8` permet de redimensionner tout le schéma sans changer les proportions.

### 5. Compilation
Utilisez toujours **pdflatex**. 
- **Attention aux Underscores** : Les caractères `_` doivent être précédés d'un anti-slash `\_` s'ils sont dans du texte normal, ou être mis entre symboles dollar `$x_i$` s'ils sont mathématiques.

---

## 🚀 Générer les PDFs

```bash
pdflatex poster1_v2.tex
pdflatex poster2_v2.tex
```