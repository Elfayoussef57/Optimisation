import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x[0]**2 + 7 * x[1]**2 / 2

def grad_f(x):
    return np.array([2 * x[0], 7 * x[1]])

def grad_pas_variable(x0, eps, max_iter, rho0=1.0, beta=0.5, c=1e-4):
    x = np.array(x0, dtype=float)
    trajectoire = [[0, x[0], x[1], np.linalg.norm(grad_f(x))]]
    
    for k in range(1, max_iter + 1):
        grad = grad_f(x)
        norm_grad = np.linalg.norm(grad)
        
        # Recherche du pas optimal (Armijo)
        rho = rho0
        while f(x - rho * grad) > f(x) - c * rho * norm_grad**2:
            rho *= beta
        
        # Mise à jour
        x = x - rho * grad
        trajectoire.append([k, x[0], x[1], norm_grad])
        
        if norm_grad < eps:
            print(f"Convergence atteinte à l'itération {k}")
            break
    
    return np.array(trajectoire)

# Initialisation
x0 = [-5, 5]
eps = 1e-5
max_iter = 50


#aficher les resultat

trajectoire = grad_pas_variable(x0, eps, max_iter)
# Lancer l'algorithme
trajectoire = grad_pas_variable(x0, eps, max_iter)

# Sauvegarder les résultats
fichier_optimal = "resultat_pas_optimal.csv"
np.savetxt(
    fichier_optimal,
    trajectoire,
    fmt="%.6f",
    header="k, xk[0], xk[1], ||grad||",
    delimiter=",",
    comments=""
)

# Charger les données
donnees = np.loadtxt(fichier_optimal, delimiter=",", skiprows=1)

# Définir le maillage pour les courbes de niveau
x = np.linspace(-6, 6, 100)
y = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])

# Création du graphique
plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=50, cmap='viridis')  # Courbes de niveau
plt.plot(donnees[:, 1], donnees[:, 2], 'r-', label="Trajectoire")  # Trajectoire
plt.scatter(donnees[:, 1], donnees[:, 2], color='red', s=10, label="Points")  # Points de la trajectoire

# Ajouter des informations au graphique
plt.title("Trajectoires de la descente de gradient sur les courbes de niveau")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.colorbar(label='Valeurs de f(x, y)')  # Barre de couleur pour les courbes de niveau
plt.grid(True)
plt.savefig("figure2.png")

