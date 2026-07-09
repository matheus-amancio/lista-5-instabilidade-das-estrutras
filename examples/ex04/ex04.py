import math

import numpy as np
import plotly.graph_objects as go


def n_otimo_inteiro(c: float) -> int:
    """
    Retorna o inteiro positivo n que minimiza
        f(n) = n² + c/n²,
    assumindo c > 0.
    """
    if c <= 0:
        raise ValueError("c deve ser positivo.")

    n_real = c**0.25

    n_inf = max(1, math.floor(n_real))
    n_sup = math.ceil(n_real)

    f_inf = n_inf**2 + c / (n_inf**2)
    f_sup = n_sup**2 + c / (n_sup**2)

    return n_inf if f_inf <= f_sup else n_sup


def show_mode(L, n):
    layout = {
        "template": "plotly_white",
    }

    v = lambda x: np.sin(n * math.pi * x / L)
    x = np.linspace(0.0, L, 1000)

    fig = go.Figure(layout=layout)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=v(x),
        )
    )
    fig.update_layout(
        title=f"n = {n}",
    )
    fig.show()


L = 5.0
EI = 450
k = 1.0e4

k_b = L**4 * k / (math.pi**4 * EI)

n = n_otimo_inteiro(k_b)

P_E = math.pi**2 * EI / (L**2)


P_cr = P_E * (n**2 + k_b / (n**2))
print(f"n = {n}")
print(f"P_E = {P_E:.2f}")
print(f"P_cr = {P_cr:.2f}")
show_mode(L, n)

n += 1
P2 = P_E * (n**2 + k_b / (n**2))
print(f"P2 = {P2:.2f}")
show_mode(L, n)

n += 1
P3 = P_E * (n**2 + k_b / (n**2))
print(f"P3 = {P3:.2f}")
show_mode(L, n)
