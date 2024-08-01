"""
@author: Nouzouhati ATHOUMANI AHAMADA
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Création d'un dataframe qui est fictif 
donnees = {
    'Surface (m2)': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
    'Nombre de chambres': [1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
    'Prix (€)': [150000, 180000, 210000, 240000, 270000, 300000, 330000, 360000, 390000, 420000]
}


df = pd.DataFrame(donnees)

# Séparation des caractéristiques et de la cible
X = df[['Surface (m2)', 'Nombre de chambres']]
y = df['Prix (€)']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Création du modèle de régression linéaire
modele = LinearRegression()

# Entraînement du modèle
modele.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = modele.predict(X_test)

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
print(f'Erreur Quadratique Moyenne (MSE) : {mse}')
print(f'Coefficients : {modele.coef_}')
print(f'Ordonnée à l\'origine : {modele.intercept_}')

# Visualisation améliorée avec Seaborn
plt.figure(figsize=(12, 6))


# Graphique des prix réels vs. prédit
sns.scatterplot(x=X_test['Surface (m2)'], y=y_test, color='blue', label='Vraies valeurs', s=100, edgecolor='w', alpha=0.7)
sns.scatterplot(x=X_test['Surface (m2)'], y=y_pred, color='red', label='Prédictions', s=100, edgecolor='w', alpha=0.7)

#Ajout des lignes de régression pour la visualisation
sns.regplot(x=X_test['Surface (m2)'], y=y_pred, scatter=False, color='red', line_kws={"linestyle": "--"})

plt.xlabel('Surface (m2)', fontsize=14)
plt.ylabel('Prix (€)', fontsize=14)
plt.title('Prédictions vs. Vraies valeurs', fontsize=16)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
