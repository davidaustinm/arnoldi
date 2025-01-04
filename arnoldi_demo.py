# %% Taken from https://nbviewer.org/github/mitmath/18335/blob/spring21/notes/Arnoldi.ipynb
import numpy as np
from scipy.linalg import qr, norm, eig, det
import scipy.linalg
import matplotlib.pyplot as plt

# %%
# Arnoldi iteration and convergence routines
def arnoldi(A:np.ndarray,x: np.ndarray, niters=50):
    m = x.size  # x is a 1D vector
    Q = np.zeros((m, niters))
    H = np.zeros((niters, niters-1))
    Q[:,0] = x/norm(x)
    for n in np.arange(H.shape[1]):   # Perform niter-1 matvecs with Aqn
        v = A@Q[:,n]                # Matrix multiply Aqn
        for j in np.arange(0,n+1):          # Change from julia to python indexing
            q = Q[:,j]
            H[j,n] = np.dot(v,q)
            v -= H[j,n]*q                   # Orthogonalize Aqn against each previous vector in Q
        H[n+1,n] = np.linalg.norm(v)
        Q[:,n+1] = v / H[n+1,n]
    return H, Q

def arnoldi_convergence(H, lamExact, nev=6):
    lamExact = np.sort(lamExact)[::-1][:nev]
    errors = np.zeros((H.shape[1]-nev+1, nev))      # Want to start on the nev_th ritz value
    for i, n in enumerate(np.arange(nev, H.shape[1]+1)):
        ritz_vals, _ = eig(H[:n,:n])
        ritz_vals = np.sort(ritz_vals)[::-1][:nev]
        # ritz_vals = np.sqrt(ritz_vals.real)
        errors[i,:] = np.abs(ritz_vals-lamExact)
    return errors

if __name__ == "__main__":
    # %%
    # Testing arnoldi convergence

    # Create a random nonhermitian matrix with logarithmically spaced eigenvalues, and a random vector to start Arnoldi
    X = np.random.randn(1000,1000)
    lam = np.logspace(-2,3,1000)
    A = np.linalg.solve(X,np.diag(lam)) @ X
    b = np.random.randn(1000)

    lam1, EV = eig(A)
    x1 = np.arange(lam.size)
    # Double check that the created matrix has the right eigenvalues
    # lam1, EV = eig(A)
    # x1 = np.arange(lam.size)
    # plt.scatter(x1, lam[::-1])
    # plt.scatter(x1, lam1)
    # plt.show()

    # %%
    # Perform arnoldi and check convergence
    H, Q = arnoldi(A,b,200)
    # np.savetxt('H.txt', H, fmt='%15.5f')

    nev = 6
    errors = arnoldi_convergence(H, lam, nev)
    for i in np.arange(nev):
        plt.semilogy(np.arange(errors.shape[0])+nev, errors[:,i], label=f'Lam = {lam[-(i+1)]:.2f}')
    plt.grid()
    plt.legend()
    plt.title('Arnoldi convergence')
    plt.xlabel('# iterations')
    plt.ylabel('Eigenvalue absolute error')
    plt.show()

    # %%
    # Test with nonhermitian A, eigenvalues clustered near the origin
    m=400
    A = np.random.randn(m,m)/np.sqrt(m)
    b=np.random.randn(m)
    eigs,_ = eig(A)
    niter=401
    H, Q = arnoldi(A,b,niter)

    # %%
    # Create animation figures
    fig,ax = plt.subplots(figsize=(9,9))
    for nplot in np.arange(400)+1:
        print(nplot)
        ritz_vals, _ = eig(H[:nplot,:nplot])

        ax.cla()
        ax.scatter(eigs.real, eigs.imag, s=2, c='black')
        ax.set_aspect('equal')
        ax.scatter(ritz_vals.real, ritz_vals.imag, s=2, c='red')
        circ = plt.Circle((0, 0), radius=1, edgecolor='k', facecolor='None')
        ax.add_patch(circ)
        ax.grid()
        ax.set_xlabel('Real')
        ax.set_ylabel('Imag')
        ax.set_title(f'Ritz value convergence for nonhermitian A, Arnoldi iteration: {nplot}')
        # plt.show()
        plt.savefig(f'./anim/{nplot:04}.png')

    # %%
    # Plot arnoldi polynomials
    lam20 = np.logspace(-2,0,20) + 1
    X20,_ = qr(np.random.randn(20,20)) # a random unitary matrix via QR on a random matrix => this puts all the eigenvalues on the real axis so we can plot the 1D characteristic polynomial
    A20 = X20 @ np.diag(lam20) @ X20.T
    b20 = np.random.randn(20)
    H20,_ = arnoldi(A20, b20, 21)

    lams = np.linspace(0,3,1000)    # For plotting
    for n in np.arange(6):
        n+= 1
        print(n)
        pn = [det(H20[:n,:n]-lami*np.eye(n)) for lami in lams]
        p_at_eig = [det(H20[:n,:n]-lami*np.eye(n)) for lami in lam20]
        maxabs = max(np.abs(p_at_eig))    # Arnoldi polynomial is monic in the last (highest) term pn that is not in the KSP Kn (nth iteration). The max(p(A)) works out to be the max(p(lam_i)), so take the maximum of the poly evaluated at all the eigenvalues
        plt.plot(lams, pn/maxabs, label=f'{n}')
        plt.xlim([.9,2.1])
        plt.ylim([-5,3])
    plt.legend()
    plt.grid()
    plt.scatter(lam20, np.zeros_like(lam20), marker='*', s=20,c='black')
    plt.title('Arnoldi polynomials normalized by max(p(lam_i))')
    plt.show()

    # %%
