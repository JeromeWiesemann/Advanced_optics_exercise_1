import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import odeint

plt.rcParams["text.usetex"] = True
plt.rc('text.latex', preamble=r'\usepackage{siunitx}')

def model(y, t):
    N1, N2, F = y

    pulse_width = 1e-5
    t_0_pulse = 0
    # R2 = 100 * np.abs(np.cos(t * 2))
    R2 = 1e8*np.exp(-(t-t_0_pulse)**2/(2.*pulse_width**2)) / (pulse_width * np.sqrt(2*np.pi))
    # R2 = 100
    l = 1
    sigma_a = 0.34e-3
    sigma_e = sigma_a
    tau_1 = 1e-5
    tau_2 = 1

    dydt = [sigma_e * F * N2 - sigma_a * F * N1 - N1 / tau_1, R2 - sigma_e * F * N2 + sigma_a * F * N1 - N2 / tau_2, (sigma_e * N2 - sigma_a * N1) * F - l * F]
    return dydt

y_0 = [0, 1000, 100]

t = np.linspace(-1e-4, 11e-4, 200000)

sol = odeint(model, y_0, t)

plt.plot(t, sol[:, 0] / max(sol[:, 0]), label = "N1", color = "C0")
plt.plot(t, sol[:, 1] / max(sol[:, 1]), label = "N2", color = "olivedrab")
plt.plot(t, sol[:, 2] / max(sol[:, 2]), label = "F", color = "red")
# plt.plot(t, sol[:, 0], label = "N1")
# plt.plot(t, sol[:, 1], label = "N2")
plt.ylabel("a.u.")
plt.xlabel(r"Time $[\si{\micro\s}]$")


y_0 = [0, 1000, 0.001]

t = np.linspace(-1e-4, 11e-4, 200000)

sol = odeint(model, y_0, t)

plt.plot(t, sol[:, 0] / max(sol[:, 0]), label = "N1", linestyle = "dotted",  color = "C0")
plt.plot(t, sol[:, 1] / max(sol[:, 1]), label = "N2", linestyle = "dotted", color = "olivedrab")
plt.plot(t, sol[:, 2] / max(sol[:, 2]), label = "F", linestyle = "dotted", color = "red")
# plt.plot(t, sol[:, 0], label = "N1")
# plt.plot(t, sol[:, 1], label = "N2")
plt.ylabel("a.u.")
plt.xlabel(r"Time $[\si{\micro\s}]$")

pulse_width = 1e-5
t_0_pulse = 0
plt.plot(t, 1*np.exp(-(t-t_0_pulse)**2/(2.*pulse_width**2)), label = "Pulse", color = "black")
# plt.plot(t, sol[:, 2], label = "F")
plt.legend()
plt.savefig("All_gauss.pdf", dpi = 300)
plt.show()