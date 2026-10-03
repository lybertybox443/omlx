# MLX natif : sous-groupes (`Group.split`) — expérimental

## Contexte

MLX upstream 0.32.2 : `Group.split` n'existe pas par défaut pour les backends ring et JACCL.
Le patch local adapte les PR upstream fermées et non fusionnées
https://github.com/ml-explore/mlx/pull/4218 et /4282, ainsi que l'issue #3205.
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
- Qwen4 TPxPP complet en anneau : bloqué, runtime GPU patché indisponible sur ce Mac (compilateur Metal absent).
- N Macs physiques, RDMA et performances : reportés par l'utilisateur.

## Limites

- Aucune preuve physique JACCL.
- Aucune affirmation de performance supérieure.
