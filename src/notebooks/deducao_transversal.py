import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import sympy as sp

    # Variáveis simbólicas
    x, L, EI = sp.symbols("x L EI", positive=True)
    kt, kr, N0 = sp.symbols("kt kr N0")

    # Coordenada adimensional
    xi = x / L

    # Funções de forma
    N1 = 1 - 3*xi**2 + 2*xi**3
    N2 = L*(xi - 2*xi**2 + xi**3)
    N3 = 3*xi**2 - 2*xi**3
    N4 = L*(-xi**2 + xi**3)

    N = sp.Matrix([[N1, N2, N3, N4]])

    # Primeira derivada
    dNdX = N.diff(x)

    # Segunda derivada
    d2NdX2 = dNdX.diff(x)

    # Rigidez elástica
    Ke = sp.integrate(
        EI * (d2NdX2.T * d2NdX2),
        (x, 0, L)
    )

    # Base elástica transversal
    Kt = sp.integrate(
        kt * (N.T * N),
        (x, 0, L)
    )

    # Base elástica rotacional
    Kr = sp.integrate(
        kr * (dNdX.T * dNdX),
        (x, 0, L)
    )

    # Rigidez geométrica
    Kg = sp.integrate(
        N0 * (dNdX.T * dNdX),
        (x, 0, L)
    )

    # Simplificação
    Ke = sp.simplify(Ke)
    Kt = sp.simplify(Kt)
    Kr = sp.simplify(Kr)
    Kg = sp.simplify(Kg)
    return EI, Ke, Kg, Kr, Kt, L, N0, kr, kt, sp


@app.cell
def _(Ke):
    Ke
    return


@app.cell
def _(EI, Ke, L, sp):
    sp.factor(Ke/(EI/L**3))
    return


@app.cell
def _(Kt):
    Kt
    return


@app.cell
def _(Kt, L, kt, sp):
    sp.factor(Kt/(kt*L/420))
    return


@app.cell
def _(Kr):
    Kr
    return


@app.cell
def _(Kr, L, kr, sp):
    sp.factor(Kr/(kr/(30*L)))
    return


@app.cell
def _(Kg):
    Kg
    return


@app.cell
def _(Kg, L, N0, sp):
    sp.factor(Kg/(N0/(30*L)))
    return


if __name__ == "__main__":
    app.run()
