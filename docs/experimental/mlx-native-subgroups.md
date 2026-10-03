# MLX natif : sous-groupes (`Group.split`) — expérimental

## Contexte

MLX upstream 0.32.2 : `Group.split` n'existe pas par défaut pour les backends ring et JACCL.
Le patch local adapte les PR upstream fermées et non fusionnées
https://github.com/ml-explore/mlx/pull/4218 et
https://github.com/ml-explore/mlx/pull/4282, ainsi que l'issue #3205.
Aucune PR n'est publiée, aucun push n'est demandé.

## Kit

- Script : `scripts/build_mlx_subgroups.py`
- Base MLX épinglée : `0e3ff3643b1c3719f78814b98e0d222afbad867c`
- Patch : `omlx/patches/native_mlx_group_split.patch`
- Licence : fichier compagnon de licence fourni avec le patch.

## Périmètre

- Supporté : anneau TCP et maillage (mesh) JACCL.
- Non supporté : sous-groupes arbitraires en anneau JACCL (la topologie physique ne le permet pas).
- Aucun repli silencieux vers TCP.
- Le callback du maillage enfant utilise son propre groupe RDMA après le bootstrap.
  Cela supprime la dépendance au parent.

## Construction

```
python scripts/build_mlx_subgroups.py --work-dir /tmp/omlx-mlx-subgroups --python /path/to/python3.11 --jobs 2
```

Prérequis : en-têtes Python et `xcrun metal` pour le GPU. Aucune installation globale.
`--cpu-only` : explicite, réservé au logiciel et aux tests d'anneau. Pas d'inférence GPU.

## Installation

Installer la roue produite UNIQUEMENT dans un venv de worker isolé :

```
python -m pip install /chemin/wheel
```

Avant : vérifier manifeste et hash, et la compatibilité OS, Python et architecture.
Toute la flotte doit avoir le même runtime compatible.

Retour arrière : supprimer le venv isolé. L'application partagée n'est pas touchée.

## Mesuré

- Roue CPU isolée compilée, bibliothèque JACCL incluse.
- Quatre processus en anneau : `test_groups` réussi (split, key, imbriqué, singleton).
- Build GPU natif compilé : MLX 0.32.4.dev, base `0e3ff36`, Metal disponible, test GPU minimal (somme = 6) réussi.
- Roue GPU stockée hors dépôt (`outputs`) : `cp311-cp311-macosx_27_0_arm64`, SHA256 `8363195459afca3eeacf1ffaf21f78f6a06eb988c37e07ebe42a43ce1cac5fd4`.
- Preuve locale (`outputs/expert-hybrid-ring-gpu.log`) : 3 tests réussis. Qwen4 EP quantifié en world 3 et world 6, puis TP2xPP2 en world 4 sur GPU, en localhost. Parité sur tous les tokens et logits à 1e-4.
- L'ancien constat « compilateur Metal absent, test bloqué » est obsolète : il est faux désormais.
- N Macs physiques, RDMA et performances : REPORTÉS (DEFERRED).

## Compatibilité de la roue

- ABI et plateforme actuelles : macOS 27 arm64, Python 3.11. Ce n'est pas une roue universelle.
- Pour d'autres workers compatibles, reconstruire avec `scripts/build_mlx_subgroups.py`.

## API

- Option `expert_parallel_size` pour l'EP.
- `allow_experimental_subgroups` : opt-in, réservé au mode hybride et au runtime patché.
- TP+EP en 3D : rejeté explicitement pour l'instant.
- La feuille de route n'est pas terminée ; rien n'indique le contraire.

## Limites

- Aucune preuve physique JACCL. Le maillage JACCL exige des liens directs.
- Aucune affirmation de performance supérieure.
- Preuve locale seulement : pas de validation multi-Mac ni RDMA.
