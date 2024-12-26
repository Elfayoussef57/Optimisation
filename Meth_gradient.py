#bibliotheques
import numpy as np
import matplotlib.pyplot as plt

# Definition de la fonction objectif
def f(x):
    return x[0]**2/2+7*x[1]**2/2

# Definition du grad de f
def grad_f(x):
    return np.array([x[0],7*x[1]])

#print("f(1,-2)=",f([1,-2]), "et grad_f(1,-2)=",grad_f([1,-2]))

def gradient_pas_fixe(rho,eps,x0,iter_max):
    x=x0
    trajectoire=[]
    for i in range(1,iter_max):
        grad=grad_f(x)
        norm_grad=np.linalg.norm(grad)
        trajectoire.append([i,x[0],x[1],norm_grad])
        x=x-rho*grad
        if norm_grad<eps:
            print("Convergence atteinte.")
            break
    return np.array(trajectoire)

# Visualisation des trajectoires pour différents pas
# Points initiaux, pas, et tolérance
x0=np.array([-5,-5]) # point initial
eps=1e-5 # tolerance
iter_max=100 # nbr maximal d'iterations 
rhos=[0.2,0.1,0.001] # pas fixe

# Minimisation par la methode de gradient a pas fixe pour diff pas
# Affichage des nombres a 3 chiffres significatives
#np.set_printoptions(precision=3) 
for rho in rhos :
    trajectoire_sol=gradient_pas_fixe(rho,eps,x0,iter_max) # (x_k)_k
    #print(trajectoire_sol)
    # Sauvegarde des resultats en fichier_rho.txt
    fichier=f'res_meth_grad_pas_fixe_{rho}.csv'
    np.savetxt(fichier,trajectoire_sol,delimiter=',',fmt=['%d','%f','%f','%f'],header="i, x1, x2, ||grad||")

# Définir le maillage pour f(x, y)
x = np.linspace(-6, 6, 100)
y = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])
colors = ['red', 'blue', 'green']
symboles=['o','^', 's']
# Tracé des courbes de niveau de f(x, y)
plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.title(f"Trajectoire de descente de gradient pour pas fixe $\\rho = {rhos[0]:.2f}$")
plt.xlabel("x")
plt.ylabel("y")

# Tracer les trajectoires pour chaque pas
for rho, color,symbole in zip(rhos, colors,symboles):
    # Charger les données depuis le fichier des resultats
    data = np.loadtxt(f'res_meth_grad_pas_fixe_{rho}.csv', delimiter=',', skiprows=1)  # skiprows=1 pour ignorer l'entête
    plt.plot(data[:, 1], data[:, 2], label=f"Pas = {rho}", color=color,linewidth=1)
    plt.scatter(data[:, 1], data[:, 2], color=color,s=20, marker=symbole)  # Points de la trajectoire

# Légende et affichage
plt.legend()
plt.grid()
plt.savefig("Meth_gradient_3.png")