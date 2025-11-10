# Bingham Fluid Lubrication Solver

Solveur numérique pour la théorie de la lubrification hydrodynamique avec des fluides de Bingham (fluides à seuil de cisaillement), basé sur les travaux de **Tichy (1991)**.

## Description

Ce projet implémente la résolution des équations de Reynolds modifiées pour les fluides de Bingham dans des géométries cylindriques :

- **Squeeze Film Damper** : amortisseur à film comprimé avec précession orbitale
- **Journal Bearing** : palier lisse avec rotation

Les fluides de Bingham sont caractérisés par :
- Un **seuil de contrainte** τ₀ en dessous duquel le matériau est rigide
- Une **viscosité** μ au-dessus du seuil

## Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation rapide
```bash
# Cloner le repository
git clone https://github.com/votre-username/bingham-lubrication.git
cd bingham-lubrication

# Créer un environnement virtuel (recommandé)
python -m venv venv

# Activer l'environnement virtuel
# Sur Windows :
venv\Scripts\activate
# Sur Linux/Mac :
source venv/bin/activate

```

### Installation en mode développement
```bash
# Installation éditable avec dépendances de dev
pip install -e .
```

## 📖 Utilisation

### Exemple basique : Squeeze Film Damper
```python
from bingham_lubrication import SqueezeFilmDamper, BearingConfig

# Paramètres géométriques et physiques
omega = 500      # rad/s - vitesse de précession
mu = 0.2         # Pa.s - viscosité dynamique
R_outer = 0.02   # m - rayon externe (20 mm)
R_inner = 0.018  # m - rayon interne (18 mm)
tau_0_star = 5   # contrainte d'écoulement 
epsilon = 0.75   # rapport d'excentricité
nb_points = 201  

squeeze_damper = SqueezeDamper(
    U1=U1,
    U2=U2,
    mu=mu,
    R0=R_outer,
    Ri=R_inner,
)

squeeze_damper.solve(
    tau_zero_star=tau_0_star,
    epsilon=epsilon,
    nb_points=nb_points,
    verbose=True,
)

print(squeeze_damper.post_processing_datas)
```


## 📚 Références

**Article principal :**
- Tichy, J. A. (1991). "Hydrodynamic lubrication theory for the Bingham plastic flow model". *Journal of Rheology*, 35(4), 477-496.
