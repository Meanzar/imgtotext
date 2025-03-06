TP 4/03/2025 DeepLearning

Romain DANIZEL
Melissa HANOUTI
# Alt Text Generator – Image Captioning

Ce projet permet de générer un texte alternatif (**alt text**) à partir d'une image en utilisant un modèle d'apprentissage profond basé sur **ViT-GPT2**.

## Prérequis

- **Python 3.8**
- Installation des dépendances :
  
  ```bash
  pip install transformers datasets Pillow torch
  ```

## Installation

Clonez le projet :

```bash
git clone <URL_DU_PROJET>
cd <NOM_DU_PROJET>
```

Installez les dépendances :

```bash
pip install transformer
pip install datasets
pip install PIL
```

## Entraînement du modèle

L'entraînement repose sur :
1. Un modèle préentraîné basé sur **ViT-GPT2**
2. Un dataset **COCO Captions** pour le transfer learning

Le modèle utilise un encodeur préentraîné qui est  *frozen* pour ne pas interagir avec l'apprentissage du nouveau dataset.

Pour lancer l'entraînement :

```bash
python train.py
```

Après l'entraînement, un fichier **imgtotext_transformer.pth** sera généré. Ce fichier contient les poids du modèle et sera utilisé pour la prédiction.

## Génération d'une légende à partir d'une image

Le fichier **generate.py** contient la fonction `pred_steps` qui :

- Ouvre l'image
- Convertit l'image en **RGB**
- Redimensionne l'image en **224x224**
- Génère une légende / alt descriptive

Le modèle retourne une prédiction lorsque l'utilisateur soumet une image.

## Interface utilisateur

### Backend – `app.py`

- Affiche l'interface web (`index.html`)
- Appel la function *pred_steps* écrite dans `generate.py` 
- Crée un dossier `uploads/` pour stocker les images envoyées
