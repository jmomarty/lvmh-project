# LVMH Test

Bonjour à tous !

Voici un code pour la partie Thompson Sampling pour la sélection et évaluation des modèles
de recommandation.

### Lancement du serveur :
`python api.py`

puis

`python main.py`

### Fichiers :
- main.py
    * Interface utilisateur
- api.py
    * API qui sert le modèle
- recommendation_system.py
    * Classe du méta-modèle de recommendation
- thompson_sampling.py
    * Classe de l'échantillonneur de Thompson
- data/*
    * Fichiers comportant des listes de chansons, triées a priori par un modèle de reco différent
- test_config.json
    * Fichier de configuration pour inialiser le systême de reco
- tests.py
    * Tests unitaires



