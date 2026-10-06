r"""Laboratorio 3 · graficar los polinomios basales y usarlos con el PIB.

Se ejecuta desde 07_interpolacion:

    .venv\Scripts\python ..\docs\curso\lab03\graficar.py          (Windows)
    .venv/bin/python ../docs/curso/lab03/graficar.py              (Mac)

Guarda dos figuras en runs\lab03\ (basales.png y pib_<ejemplo>.png) y escribe una
tabla en la terminal. Tú solo cambias el bloque «AJUSTA AQUÍ»; el resto se lee, no
se toca. Ábrelo con el Bloc de notas, cambia, guarda (Ctrl+S) y vuelve a correr
(flecha arriba, Enter).
"""

# ============================ AJUSTA AQUÍ ============================

NODOS = [-1, 0, 1]      # nodos de los basales, en la variable t
PUNTOS = [-0.5, 0.5]    # valores de t donde quieres leer cada basal
RANGO = None            # None = dibujar solo entre el primer y el último nodo;
                        # o un par (t mínimo, t máximo), por ejemplo (-1.5, 2.5)

EJEMPLO = "vietnam"     # "vietnam", "colombia" o "china"

# Cada ejemplo: país, años de los nodos, cambio de variable t = (año − centro)/escala,
# años que se quieren estimar y ventana de años de la gráfica.
EJEMPLOS = {
    "vietnam": dict(pais="Viet Nam", nodos=[2012, 2014, 2016], centro=2014, escala=2,
                    estimar=[2013, 2015], ventana=(2010, 2018)),
    "colombia": dict(pais="Colombia", nodos=None, centro=None, escala=None,
                     estimar=None, ventana=None),
    "china": dict(pais="China", nodos=None, centro=None, escala=None,
                  estimar=None, ventana=None),
}

# ====================================================================

import csv
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "pib_tres_paises.csv"
SALIDA = Path("runs") / "lab03"
URL = "https://raw.githubusercontent.com/datasets/gdp/main/data/gdp.csv"


def base_lagrange(nodos, k, t):
    """L_k(t): vale 1 en nodos[k] y 0 en los demás nodos."""
    valor = 1.0
    for j, tj in enumerate(nodos):
        if j != k:
            valor *= (t - tj) / (nodos[k] - tj)
    return valor


def interpolar(nodos, valores, t):
    """P(t) = suma de valores[k] · L_k(t)."""
    return sum(valores[k] * base_lagrange(nodos, k, t) for k in range(len(nodos)))


def leer_pib():
    pib = {}
    with open(DATOS, encoding="utf-8") as f:
        filas = [linea for linea in f if not linea.startswith("#")]
    for fila in csv.DictReader(filas):
        pib.setdefault(fila["pais"], {})[int(fila["anio"])] = float(fila["pib"])
    return pib


def reconstruir():
    """Vuelve a construir pib_tres_paises.csv desde la fuente original (necesita internet)."""
    import urllib.request

    texto = urllib.request.urlopen(URL, timeout=30).read().decode("utf-8")
    filas = list(csv.DictReader(texto.splitlines()))
    paises = ("Viet Nam", "Colombia", "China")
    with open(DATOS, "w", encoding="utf-8", newline="") as f:
        f.write("# PIB anual en miles de millones de dólares corrientes (USD).\n")
        f.write("# Fuente: Banco Mundial (NY.GDP.MKTP.CD), vía https://github.com/datasets/gdp (CC BY 4.0).\n")
        f.write("# Reconstruido con graficar.py --reconstruir y redondeado a un decimal.\n")
        f.write("pais,anio,pib\n")
        for p in paises:
            for r in filas:
                if r["Country Name"] == p and 2005 <= int(r["Year"]) <= 2023:
                    f.write(f"{p},{r['Year']},{float(r['Value']) / 1e9:.1f}\n")
    print(f"reconstruido {DATOS}")


def graficar_basales(plt):
    lo, hi = RANGO if RANGO is not None else (min(NODOS), max(NODOS))
    ts = [lo + (hi - lo) * i / 400 for i in range(401)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k in range(len(NODOS)):
        ax.plot(ts, [base_lagrange(NODOS, k, t) for t in ts], label=f"$L_{k}$")
    ax.plot(NODOS, [0] * len(NODOS), "ko")
    ax.plot(NODOS, [1] * len(NODOS), "ko", mfc="white")
    ax.axhline(0, color="gray", lw=0.8)
    ax.axhline(1, color="gray", lw=0.8, ls="--")
    for p in PUNTOS:
        ax.axvline(p, color="crimson", lw=0.8, ls=":")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("t")
    ax.set_title(f"Basales de Lagrange con nodos {NODOS}")
    ax.grid(alpha=0.3)
    ax.legend()
    ruta = SALIDA / "basales.png"
    fig.savefig(ruta, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"nodos {NODOS}")
    for p in PUNTOS:
        L = [base_lagrange(NODOS, k, p) for k in range(len(NODOS))]
        print(f"  t = {p:>5}: " + "  ".join(f"L{k} = {v:+.4f}" for k, v in enumerate(L))
              + f"   suma = {sum(L):.4f}   suma |L| = {sum(abs(v) for v in L):.4f}")
    print(f"  figura: {ruta}\n")
    return ruta


def graficar_pib(plt):
    cfg = EJEMPLOS[EJEMPLO]
    faltan = [k for k, v in cfg.items() if v is None]
    if faltan:
        print(f"El ejemplo '{EJEMPLO}' tiene campos sin llenar: {', '.join(faltan)}. Llénalos en AJUSTA AQUÍ.")
        return None
    serie = leer_pib()[cfg["pais"]]
    t_nodos = [(a - cfg["centro"]) / cfg["escala"] for a in cfg["nodos"]]
    f_nodos = [serie[a] for a in cfg["nodos"]]
    a0, a1 = cfg["ventana"]
    anios = [a0 + (a1 - a0) * i / 400 for i in range(401)]
    P = [interpolar(t_nodos, f_nodos, (a - cfg["centro"]) / cfg["escala"]) for a in anios]
    reales = sorted(a for a in serie if a0 <= a <= a1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(reales, [serie[a] for a in reales], "o-", color="lightgray", label="PIB real")
    ax.plot(anios, P, color="tab:blue", label="interpolante de grado 2")
    ax.plot(cfg["nodos"], f_nodos, "o", color="tab:blue", ms=9, label="nodos")
    print(f"{cfg['pais']}: nodos en t = {t_nodos}")
    print(f"  {'año':>6} {'t':>6} {'P(t)':>10} {'real':>10} {'error':>9} {'error %':>8}")
    for a in cfg["estimar"]:
        t = (a - cfg["centro"]) / cfg["escala"]
        est = interpolar(t_nodos, f_nodos, t)
        ax.plot(a, est, "X", color="crimson", ms=10)
        print(f"  {a:>6} {t:>6.2f} {est:>10.1f} {serie[a]:>10.1f} {est - serie[a]:>9.1f} {100 * (est - serie[a]) / serie[a]:>7.1f}%")
    ax.plot([], [], "X", color="crimson", label="estimaciones")
    ax.set_xlabel("año")
    ax.set_ylabel("PIB (miles de millones de USD corrientes)")
    ax.set_title(cfg["pais"])
    ax.grid(alpha=0.3)
    ax.legend()
    ruta = SALIDA / f"pib_{EJEMPLO}.png"
    fig.savefig(ruta, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura: {ruta}")
    return ruta


def main():
    if "--reconstruir" in sys.argv:
        reconstruir()
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Falta matplotlib. Instálalo con:  .venv\\Scripts\\python -m pip install matplotlib")
        return
    SALIDA.mkdir(parents=True, exist_ok=True)
    graficar_basales(plt)
    graficar_pib(plt)
    print("\nAbre las figuras desde el explorador de archivos (carpeta runs\\lab03).")


if __name__ == "__main__":
    main()
