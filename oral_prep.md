# 🎤 Préparation de l'Oral : Projet Makeitalive

Ce document résume les points clés de votre projet et propose des réponses aux questions probables du jury.

---

## 📝 Résumé du Projet (Pitch de 2 minutes)
**Objectif** : Transformer une image de paysage statique en une vidéo animée réaliste.
**Stratégie** : Nous avons exploré deux architectures polaires :
1.  **Motion Flow (Le "Lightweight")** : Un U-Net léger qui prédit le mouvement des pixels existants (Optical Flow). On déplace les pixels de l'image originale sans en créer de nouveaux.
2.  **Stable Video Diffusion (Le "Generative")** : Un modèle de diffusion de 1,5 milliard de paramètres fine-tuné pour générer des séquences vidéo complètes (création de pixels).

---

## 🔍 Détails par Poster

### Poster 1 : Architecture et Inférence
*   **Concept du Flow** : On ne prédit par l'image suivante directement, mais un champ de vecteurs $(dx, dy)$. L'image finale est obtenue par **Backward Mapping** (on va chercher la couleur à la position source).
*   **U-Net** : Pourquoi ? Les **skip-connections** sont vitales. Elles permettent de conserver les textures haute résolution de l'image d'entrée pour que le warping soit net.
*   **Inférence** :
    *   *Oscillatory (Ping-Pong)* : Une seule prédiction, on fait varier l'amplitude. Zéro perte de qualité.
    *   *Auto-Régressive* : On ré-injecte l'image générée. Permet un mouvement infini mais crée du flou (resampling successif).
*   **SVD** : Utilisation du conditionnement par image via un encodeur VAE et CLIP. On ne touche qu'aux couches temporelles.

### Poster 2 : Données et Entraînement
*   **Dataset Pipeline** : Extraction automatique depuis YouTube 4K. Importance du **Frame Gap** (4 images) : trop court = pas de mouvement, trop long = mouvement trop complexe pour le modèle.
*   **Filtrage** : Le point technique fort. L'utilisation de la distance de **Bhattacharyya** (histogrammes) et de la **MAE** sur miniatures pour détecter les coupures de montage vidéo.
*   **Loss Warping** : C'est de l'apprentissage **auto-supervisé**. On n'a pas de "vérité" pour le mouvement, donc on demande au modèle : "Prédit un mouvement tel que si je déplace les pixels de l'image A, je tombe sur l'image B".

---

## ❓ Questions Probables du Jury

### 1. "Pourquoi avoir utilisé une reconstruction loss (warping) plutôt que de prédire l'image B directement ?"
*   **Réponse** : Prédire l'image B directement (pixel-to-pixel) produit souvent des images floues car le modèle fait une "moyenne" des possibilités. En prédisant le **flow**, on force le modèle à réutiliser les pixels de l'image originale, ce qui garantit une netteté parfaite et une fidélité totale aux textures.

### 2. "Quelles sont les limites de votre approche Motion Flow ?"
*   **Réponse** : Le Motion Flow ne peut pas gérer les **occlusions** (un objet qui passe derrière un autre) ou les changements topologiques (de l'eau qui jaillit). Comme on ne fait que déplacer des pixels existants, si un pixel manque (parce qu'il était caché), le modèle crée un étirement (stretch). C'est là que SVD est supérieur car il peut "inventer" les pixels manquants.

### 3. "À quoi servent les skip-connections dans votre U-Net pour ce projet ?"
*   **Réponse** : Dans l'encodeur, on perd de la résolution spatiale. Les skip-connections ramènent les détails fins (le grain de la roche, les feuilles) directement vers le décodeur. Sans elles, le champ de vecteurs serait trop "lisse" et le mouvement ne suivrait pas précisément les bords des objets.

### 4. "Pourquoi le filtrage des données est-il si important ici ?"
*   **Réponse** : Si le dataset contient des "cuts" (passage d'une forêt à une montagne entre deux frames), le modèle de Motion Flow va essayer de trouver un mouvement aberrant pour transformer des arbres en rochers. Cela corrompt totalement les poids du réseau. Le filtrage garantit que $I_A$ et $I_B$ sont la même scène.

### 5. "Quel est l'intérêt de l'inférence auto-régressive ?"
*   **Réponse** : Le warping classique (oscillatoire) est limité à un mouvement de va-et-vient. L'auto-régressif permet de simuler un flux continu (une rivière qui coule toujours dans le même sens). Le défi est de limiter l'accumulation d'erreurs d'interpolation qui finit par flouter l'image.

---

## 📈 Perspectives de conclusion
Si on vous demande "Et si vous aviez 6 mois de plus ?", répondez :
1.  **Segmentation Sémantique** : Intégrer un masque pour que le ciel bouge différemment de l'eau.
2.  **Hybridization** : Utiliser la rapidité du Motion Flow pour générer une structure de mouvement, puis utiliser un modèle de diffusion type SVD pour "peindre" les détails et corriger les étirements.

C'est parti ! Voici des réponses précises pour ton oral, basées exactement sur ton code et tes posters.

1. C'est quoi le warping ?
Le warping (ou déformation) est l'action de déplacer les pixels d'une image vers de nouvelles coordonnées.

Techniquement : On utilise un champ de vecteurs $(dx, dy)$ appelé Motion Flow. Pour chaque pixel $(x, y)$ de l'image de destination, on va regarder quel pixel de l'image source "atterrit" là.
Backward Mapping : Dans ton code (

train.py
), on fait du Backward Mapping (via grid_sample). On ne pousse pas les pixels vers l'avant (ce qui créerait des trous), on part de l'image finale et on va "chercher" la couleur dans l'image initiale.
Analogie : Imagine que ton image est imprimée sur une feuille de chewing-gum. Le warping, c'est étirer localement la feuille selon les flèches du Motion Flow.
2. Pourquoi un U-Net et pas un autre modèle ?
Le U-Net est l'architecture "reine" pour les tâches Image-to-Image (ici : Image $\to$ Flow).

Skip Connections (Connexions résiduelles) : C'est le point clé. L'encodeur compresse l'image (perdant des détails mais comprenant le contexte global), et le décodeur reconstruit le flux. Les skip-connections permettent de transmettre les détails fins (textures, bords des objets) directement de l'entrée vers la sortie.
Précision spatiale : Pour le Motion Flow, le mouvement doit suivre exactement les contours (ex: l'eau bouge, mais pas le rocher d'à côté). Le U-Net excelle à conserver cette précision chirurgicale.
Alternatives : Un simple CNN sans skip-connections produirait un flux très flou et imprécis. Un Transformer serait beaucoup plus lourd et difficile à entraîner sur un petit dataset.
3. Comment fonctionne le U-Net dans les détails ? Utilise-t-il des CNN ?
Oui, il est entièrement basé sur des CNN (Convolutional Neural Networks). Dans ton fichier 

model.py
 :

Encodeur (Descente) : Successions de Conv2d (pour extraire les caractéristiques) et de MaxPool2d (pour réduire la taille de l'image et augmenter le champ de vision du modèle).
Bottleneck (Le bas du U) : La couche la plus profonde où le modèle a une vision "abstraite" et globale de l'image.
Décodeur (Montée) : Utilise des Upsample (pour remonter à la taille 512x512). À chaque étape, il concatène (torch.cat) les caractéristiques venant de l'encodeur (les fameuses skip connections).
BatchNorm & ReLU : Utilisés après chaque convolution pour stabiliser l'apprentissage et introduire de la non-linéarité.
4. Quelle "loss" pour le tri des images (make_dataset_video.py) ?
Ce n'est pas une "loss" d'entraînement, mais des métriques de distance pour filtrer les "cuts" (changements de scène) :

Hist Distance (Bhattacharyya) : Compare les histogrammes de couleurs des deux images. Si les couleurs changent brutalement (ex: bleu ciel $\to$ vert forêt), on jette la paire.
Thumbnail MAE (Mean Absolute Error) : On réduit les images en 16x16 niveaux de gris et on compare la différence. C'est très efficace pour détecter si la structure globale de l'image a changé (ce qui arrive lors d'un montage vidéo).
5. Quelle loss pour l'entraînement (train.py) ? Est-ce que le poster est juste ?
La loss utilisée est la MSE (Mean Squared Error), comme indiqué sur ton poster !

Dans le code (

train.py
) : criterion = nn.MSELoss().
Le concept : C'est une loss de reconstruction. Le modèle prédit un Flow. On "warpe" l'image A avec ce Flow pour obtenir une image "prédite" $B'$. La loss mesure l'écart entre cette image $B'$ et la vraie image suivante $B$ du dataset.
Verdict : Ton poster est 100% fidèle au code. Tu peux dire au jury : "C'est un apprentissage auto-supervisé : la supervision vient de la frame suivante de la vidéo, pas d'étiquettes manuelles."
Conseil pour l'oral : Si on te demande pourquoi le Motion Flow est "limité" (comme tu l'as écrit en Section 6 du Poster 2), précise que c'est parce qu'il ne peut pas inventer de nouveaux pixels (il ne fait que déplacer les existants), contrairement à SVD qui est un modèle génératif.