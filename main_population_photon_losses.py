import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from scipy.integrate import odeint

plt.rcParams["text.usetex"] = True
plt.rc('text.latex', preamble=r'\usepackage{siunitx}')

def model(y, t, l):
    N1, N2, F = y

    pulse_width = 1e-5
    t_0_pulse = 0
    # R2 = 1*np.exp(-(t-t_0_pulse)**2/(2.*pulse_width**2))
    R2 = 100
    sigma_a = 0.34e-3
    sigma_e = sigma_a
    tau_1 = 1e-5
    tau_2 = 1

    dydt = [sigma_e * F * N2 - sigma_a * F * N1 - N1 / tau_1, R2 - sigma_e * F * N2 + sigma_a * F * N1 - N2 / tau_2, (sigma_e * N2 - sigma_a * N1) * F - l * F]
    return dydt

y_0 = [0, 1000, 1]

t = np.linspace(-1, 5, 200000)

cmap = matplotlib.cm.get_cmap('plasma')

c = 0
for l in np.arange(0, 1, 0.1):
    sol = odeint(model, y_0, t, args = (l,))
    plt.plot(t, sol[:, 2], label = rf"$l = {round(l,2)}$", color = cmap(l))





# plt.plot(t, sol[:, 0], label = "N1")
# plt.plot(t, sol[:, 1], label = "N2")
plt.ylabel("Photon flux")
plt.xlabel(r"Time $[\si{\micro\s}]$")

pulse_width = 1e-5
t_0_pulse = 0
# plt.plot(t, 1*np.exp(-(t-t_0_pulse)**2/(2.*pulse_width**2)), label = "Pulse")
# plt.plot(t, sol[:, 2], label = "F")
plt.legend()
plt.savefig("Photon_flux_dependence_losses.pdf", dpi = 300)
plt.show()