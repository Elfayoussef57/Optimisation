import numpy as np
import matplotlib.pyplot as plt

# Use Agg backend for non-interactive environments
plt.switch_backend('Agg')

# Fonction et son gradient
def f(x):
    return x[0]**2 + 7*x[1]**2 / 2

def grad_f(x):
    return np.array([2 * x[0], 7 * x[1]])

# Algorithme de descente de gradient à pas fixe
def grad_pasfixe(rho, eps, x0, max_iter):
    x = np.array(x0, dtype=float)
    trajectoire = [[0, x[0], x[1], np.linalg.norm(grad_f(x))]]
    for i in range(1, max_iter + 1):
        grad = grad_f(x)
        x = x - rho * grad
        norm_grad = np.linalg.norm(grad)
        trajectoire.append([i, x[0], x[1], norm_grad])
        if norm_grad < eps:
            print(f"Le problème de minimisation est résolu à l'itération {i}!")
            break
    return np.array(trajectoire)

# Initialisation des paramètres
x0 = np.array([-5, 5])
eps = 1e-5
rhos = [0.5, 0.1, 0.01]
max_iter = 50

# Résultats pour chaque pas
# Initialisation de la figure
plt.figure(figsize=(10, 8))

# Définir le maillage pour les courbes de niveau
x = np.linspace(-6, 6, 100)
y = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])

# Tracer les courbes de niveau (pas de label ici)
plt.contour(X, Y, Z, levels=50, cmap='viridis')

# Tracer les trajectoires pour chaque rho
for rho in rhos:
    trajectoire = grad_pasfixe(rho, eps, x0, max_iter)
    fichier = f"resultat_pas_{rho}.csv"
    np.savetxt(
        fichier,
        trajectoire,
        fmt="%.6f",
        header="k, xk[0], xk[1], ||grad||",
        delimiter=",",
        comments=""
    )

    # Charger les données du fichier et tracer la trajectoire
    donnees = np.loadtxt(fichier, delimiter=",", skiprows=1)
    plt.plot(donnees[:, 1], donnees[:, 2], label=f"Trajectoire Rho={rho}")
    plt.scatter(donnees[:, 1], donnees[:, 2], s=10)  # Points sans label pour éviter de surcharger la légende

# Ajouter les détails au graphique
plt.title("Trajectoires de la descente de gradient")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()  # S'assure que seuls les éléments avec un label sont inclus

# Save the plot to a file
plt.savefig('trajectoires_descente_gradient.png')


