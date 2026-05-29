# Manuel d'utilisation

## Lancer l'outil

```powershell
python napari_fiber_studio.py
```

Ou avec une image précise:

```powershell
python napari_fiber_studio.py "C:\chemin\vers\image.jpg"
```

## Flux recommandé

1. Ouvrir une image avec `Open Image`.
2. Cliquer `Run Auto SAM`.
3. Vérifier les propositions dans `SAM Proposals`.
4. Cliquer `Create Instance Labels` pour construire une seule couche `Instance Labels`.
5. Corriger les instances:
   - `Delete Selected Proposal` pour supprimer une proposition SAM.
   - `Add Selected Proposal` pour injecter seulement une proposition choisie.
   - `Delete Selected Instance` pour supprimer une instance finale.
   - `Edit Instances` pour peindre une instance.
   - `Erase Instances` pour effacer des pixels.
   - `Fill Selected Instance` pour remplir une zone connexe.
   - `New Empty Instance` pour créer une nouvelle instance vide à peindre.
   - `Sync Manual Edits` pour recopier les modifications du layer napari dans l'état interne.
6. Pour une correction locale:
   - `ROI Rectangle` ou `ROI Polygon`
   - `Delete Proposals In ROI` ou `Erase Instance Pixels In ROI`
7. Pour une correction géométrique fine:
   - sélectionner une instance
   - `Edit Keypoints`
   - déplacer les keypoints
   - `Refine Spline`
8. Cliquer `Save COCO + Next Image` pour enregistrer dans `coco_fiber.json` puis passer à l'image suivante.

## Sauvegarde des annotations

- Toutes les annotations validées sont centralisées dans un seul fichier:

```text
coco_fiber.json
```

- Chaque image met à jour ses annotations dans ce fichier unique.
- Les anciennes annotations de la même image sont remplacées lors de l'enregistrement.

## Navigation des images

- `Prev Image` et `Next Image` permettent de naviguer dans le dossier courant, `images`, ou `test_images`.
- Le compteur en haut indique la position courante.

## Conseils d'édition

- Utiliser `Show Combined Masks` pour ne voir que la couche finale des instances.
- Utiliser `Show Proposals + Instances` pour comparer les propositions SAM avec les masks finaux.
- Si une proposition SAM suit mal la géométrie, préférer:
  - `Add Selected Proposal` puis édition manuelle
  - ou `Run Prompted SAM` avec points positifs/négatifs et box
  - puis `Refine Spline`

## Entraînement FiberR-CNN

Le pipeline d'entraînement lit maintenant le fichier central `coco_fiber.json`.

```powershell
python pipeline.py
```

Points importants:

- le modèle de base est `Keypoint R-CNN`
- le dataset mapper filtre les instances vides correctement
- les métriques fibre sont synchronisées avec les annotations restantes après transformation

## Débogage

L'application écrit des messages dans le terminal avec le préfixe:

```text
[FiberStudio]
```

Cela permet de suivre:

- la sélection d'instance ou de proposition
- les sauvegardes COCO
- les changements de mode
- la navigation entre images
