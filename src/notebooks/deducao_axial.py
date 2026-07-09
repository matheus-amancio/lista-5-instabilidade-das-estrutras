import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import sympy as sp

    # Variáveis simbólicas
    x, L, EA = sp.symbols("x L EA", positive=True)
    kl = sp.symbols("kl")


    # Funções de forma
    N1 = 1 - x / L
    N2 = x / L

    N = sp.Matrix([[N1, N2]])

    # Primeira derivada
    dNdX = N.diff(x)

    # Rigidez elástica
    Ke = sp.integrate(
        EA * (dNdX.T * dNdX),
        (x, 0, L)
    )

    # Base elástica longitudinal
    Kb = sp.integrate(
        kl * (N.T * N),
        (x, 0, L)
    )

    Ke = sp.simplify(Ke)
    Kb = sp.simplify(Kb)
    return EA, Kb, Ke, L, N, kl, sp, x


@app.cell
def _(Ke):
    Ke
    return


@app.cell
def _(EA, Ke, L, sp):
    sp.factor(Ke/(EA/L))
    return


@app.cell
def _(Kb):
    Kb
    return


@app.cell
def _(Kb, L, kl, sp):
    sp.factor(Kb/(kl*L/6))
    return


@app.cell
def _(L, N, sp, x):
    # Vetor de forças nodais equivalentes para carregamento uniformemente distribuído
    p = sp.symbols("p")
    f_eq = sp.integrate(
        N.T * p,
        (x, 0, L)
    )
    return f_eq, p


@app.cell
def _(f_eq):
    f_eq
    return


@app.cell
def _(L, f_eq, p, sp):
    sp.factor(f_eq/(p*L/2))
    return


if __name__ == "__main__":
    app.run()
