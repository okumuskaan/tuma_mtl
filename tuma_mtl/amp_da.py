"""
AMP-DA Implementation

This file contains the AMP-DA algorithm.

References:
- L. Qiao, Z. Gao, M. B. Mashhadi, and D. Gündüz, "Massive Digital Over-the-Air Computation for Communication-Efficient Federated Edge Learning," 
  IEEE Journal on Selected Areas in Communications, 2024. https://github.com/liqiao19/MD-AirComp
"""

import numpy as np # type: ignore
import scipy.stats as st # type: ignore

def AMP_DA(y, C, Ka, maxIte, exx=1e-10, damp=0.3):
    """
    Approximate Message Passing - Digital Aggregation (AMP-DA).

    Parameters:
    y : np.ndarray
        Received signal matrix.
    C : np.ndarray
        Codebook matrix.
    Ka : int
        Number of active users.
    maxIte : int
        Maximum number of iterations.

    Returns:
    x_hat : np.ndarray
        Estimated multiplicity vector for the transmitted messages.
    """


    N_RAs = C.shape[0] # N_RAs = H.shape[0] # Y.shape[0]
    N_UEs = C.shape[1] #N_UEs = H.shape[1] # M in our case
    N_dim = y.shape[1] #y.shape[1] # This should be 1!
    N_M = y.shape[1] # Number of antennas, F in our case

    alphabet = np.arange(0.0, Ka + 1, 1)
    M = len(alphabet) - 1

    lam = N_RAs / N_UEs
    c = np.arange(0.01, 10, 10 / 1024)
    rho = (1 - 2 * N_UEs * ((1 + c ** 2) * st.norm.cdf(-c) - c * st.norm.pdf(c)) / N_RAs) / (
            1 + c ** 2 - 2 * ((1 + c ** 2) * st.norm.cdf(-c) - c * st.norm.pdf(c)))
    alpha = lam * np.max(rho) * np.ones((N_UEs, N_dim)) 
    x_hat0 = np.ones((N_UEs, N_dim))[:, :, None].repeat(N_M,axis=2)
    x_hat = (x_hat0 + 1j * x_hat0) 
    var_hat = np.ones((N_UEs, N_dim, N_M)) 

    V = np.ones((N_RAs, N_dim, N_M))  
    V_new = np.ones((N_RAs, N_dim, N_M))
    Z_new = y.copy()
    sigma2 = 100
    t = 1
    Z = y.copy()
    MSE = np.zeros(maxIte)  
    MSE[0] = 100
    hvar = (np.linalg.norm(y) ** 2 - N_RAs * sigma2) / (N_dim * lam * np.max(rho) * np.linalg.norm(C) ** 2)
    hmean = 0
    alpha_new = np.ones((N_UEs, N_dim, N_M))  
    x_hat_new = (np.ones((N_UEs, N_dim, N_M)) + 1j * np.ones((N_UEs, N_dim, N_M))) 
    var_hat_new = np.ones((N_UEs, N_dim, N_M))  

    hvarnew = np.zeros(N_M)  
    hmeannew = (np.zeros(N_M) + 1j * np.zeros(N_M))  
    sigma2new = np.zeros(N_M) 

    alphabet = alphabet 
    alpha = alpha 
    while t < maxIte:
        x_hat_pre = x_hat.copy()
        for i in range(N_M):
            V_new[:, :, i] = np.abs(C) ** 2 @ var_hat[:, :, i]
            Z_new[:, :, i] = C @ x_hat[:, :, i] - ((y[:, :, i] - Z[:, :, i]) / (sigma2 + V[:, :, i])) * V_new[:, :, i]  # + 1e-8

            Z_new[:, :, i] = damp * Z[:, :, i] + (1 - damp) * Z_new[:, :, i]
            V_new[:, :, i] = damp * V[:, :, i] + (1 - damp) * V_new[:, :, i]

            var1 = (np.abs(C) ** 2).T @ (1 / (sigma2 + V_new[:, :, i]))
            var2 = C.conj().T @ ((y[:, :, i] - Z_new[:, :, i]) / (sigma2 + V_new[:, :, i]))

            Ri = var2 / (var1) + x_hat[:, :, i]
            Vi = 1 / (var1)

            sigma2new[i] = ((np.abs(y[:, :, i] - Z_new[:, :, i]) ** 2) / (
                        np.abs(1 + V_new[:, :, i] / sigma2) ** 2) + sigma2 * V_new[:, :, i] / (
                                        V_new[:, :, i] + sigma2)).mean()

            if i == 0:
                r_s = np.tile(Ri[None, :, :], (M + 1, 1, 1)) - np.tile(alphabet[:, None, None], (1, N_UEs, N_dim))
                max_exp_input = 700  # Safe range for np.exp
                exp_input = np.clip(-(np.abs(r_s) ** 2 / Vi), -max_exp_input, max_exp_input)
                pf8 = np.exp(exp_input) / Vi / np.pi
                pf7 = np.zeros((M + 1, N_UEs, N_dim))  
                pf7[0, :, :] = pf8[0, :, :] * (np.ones((N_UEs, N_dim)) - alpha)  
                pf7[1:, :, :] = pf8[1:, :, :] * (alpha / M)
                del pf8
                PF7 = np.sum(pf7, axis=0)
                pf6 = pf7 / PF7
                del pf7, PF7
                AAA = np.tile(alphabet[None, :, None], (N_dim, 1, 1))
                BBB = np.transpose(pf6, (2, 1, 0)) 
                x_hat_new[:, :, i] = (np.einsum("ijk,ikn->ijn", BBB, AAA).squeeze(-1)).T
                del AAA
                alphabet2 = alphabet ** 2
                AAA2 = np.tile(alphabet2[None, :, None], (N_dim, 1, 1))
                var_hat_new[:, :, i] = (np.einsum("ijk,ikn->ijn", BBB, AAA2).squeeze(-1)).T - np.abs(
                    x_hat_new[:, :, i]) ** 2
                del AAA2
                alpha_new[:, :, i] = np.clip(np.sum(pf6[1:, :, :], axis=0), exx, 1 - exx)
                del pf6
            else:
                A = (hvar * Vi) / (Vi + hvar)
                B = (hvar * Ri + Vi * hmean) / (Vi + hvar)
                
                lll = (
                    np.log(Vi / (Vi + hvar)) / 2
                    + np.abs(Ri)**2 / (2 * Vi)
                    - np.abs(Ri - hmean)**2 / (2 * (Vi + hvar))
                )
                pai = np.clip(
                    alpha / (alpha + (1 - alpha) * np.exp(-lll)),
                    exx,
                    1 - exx
                )
                x_hat_new[:, :, i] = pai * B
                var_hat_new[:, :, i] = (pai * (np.abs(B) ** 2 + A)) - np.abs(x_hat_new[:, :, i]) ** 2
                # mean update
                hmeannew[i] = (np.sum(pai * B, axis=0) / np.sum(pai, axis=0)).mean()
                # variance update
                hvarnew[i] = (np.sum(pai * (np.abs(hmean - B) ** 2 + Vi), axis=0) / np.sum(pai, axis=0)).mean()
                # activity indicator update
                alpha_new[:, :, i] = np.clip(pai, exx, 1 - exx)
        if N_M > 1:
            hvar = hvarnew[1:].mean()
            hmean = hmeannew[1:].mean()
        sigma2 = sigma2new.mean()
        alpha = (np.sum(alpha_new, axis=2) / N_M)
        III = x_hat_pre - x_hat_new
        NMSE_iter = np.sum(np.abs(III) ** 2) / np.sum(np.abs(x_hat_new) ** 2)
        MSE[t] = (
            np.sum(
                np.abs(
                    y
                    - np.transpose(
                        np.einsum(
                            "ijk,ikn->ijn",
                            np.transpose(x_hat, (2, 1, 0)),
                            np.tile(C.T[None, :, :], (N_M, 1, 1))
                        ),
                        (2, 1, 0)
                    )
                ) ** 2
            ) / N_RAs / N_dim / N_M
        )
        x_hat = x_hat_new.copy()
        if t > 15 and MSE[t] >= MSE[t - 1]:
            x_hat = x_hat_pre.copy()
            break

        var_hat = var_hat_new.copy()
        V = V_new.copy()
        Z = Z_new.copy()
        t = t + 1

    est_k = x_hat[:, 0, 0].real.astype(int)
    est_k = est_k.reshape(9, -1).sum(axis=0)

    return est_k
