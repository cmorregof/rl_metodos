"""Interfaz en vivo para la terminal (estilo "mira cómo aprende")."""

from __future__ import annotations

import math

from rich.console import Group, RenderableType
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .env_interp import ACTION_SHORT, EST_LABELS, GAMMAS, N_ACTIONS, TREND_LABELS, theoretical_action
from .proof_kb import MIN_PROOF_LENGTH, STEPS
from .trainer import ALL_STATES, Trainer

SPARK = "▁▂▃▄▅▆▇█"


def fmt_time(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def sparkline(values: list[float], width: int, lo: float = 0.0, hi: float = 1.0) -> str:
    vals = [v for v in values[-width:] if not math.isnan(v)]
    if not vals:
        return ""
    out = []
    for v in vals:
        k = 0 if hi == lo else int((v - lo) / (hi - lo) * (len(SPARK) - 1))
        out.append(SPARK[max(0, min(len(SPARK) - 1, k))])
    return "".join(out)


def bar(frac: float, width: int) -> str:
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return "█" * n + "░" * (width - n)


# ---------------------------------------------------------------------------
# Fase 1
# ---------------------------------------------------------------------------
def plot_interp(tr: Trainer, width: int = 46, height: int = 8) -> Text:
    """f (cian) frente a pₙ (amarillo) en [−1, 1]; nodos ▲ en la línea base."""
    env = tr.env1
    cols = [int(k * (len(env.grid) - 1) / (width - 1)) for k in range(width)]
    fy = [env.fgrid[c] for c in cols]
    py = [env.pgrid[c] if env.pgrid else float("nan") for c in cols]
    good = [y for y in fy if math.isfinite(y)]
    lo, hi = min(good + [0.0]), max(good + [0.0])
    pad = 0.15 * (hi - lo or 1)
    lo, hi = lo - pad, hi + pad
    grid = [[" "] * width for _ in range(height)]
    styles = [[""] * width for _ in range(height)]

    def row_of(y):
        if not math.isfinite(y) or y < lo or y > hi:
            return None
        return height - 1 - int((y - lo) / (hi - lo) * (height - 1))

    z = row_of(0.0)
    if z is not None:
        for c in range(width):
            grid[z][c], styles[z][c] = "─", "dim"
    for c, y in enumerate(py):
        r = row_of(y)
        if r is not None:
            grid[r][c], styles[r][c] = "·", "yellow"
        elif math.isfinite(y):  # se sale del recuadro: Runge
            r = 0 if y > hi else height - 1
            grid[r][c], styles[r][c] = "↑" if y > hi else "↓", "bold red"
    for c, y in enumerate(fy):
        r = row_of(y)
        if r is not None:
            grid[r][c], styles[r][c] = "•", "cyan"
    base = z if z is not None else height - 1
    for xn in env.x:
        c = int((xn + 1) / 2 * (width - 1))
        grid[base][c], styles[base][c] = "▲", "bold magenta"
    t = Text()
    for r in range(height):
        for c in range(width):
            t.append(grid[r][c], style=styles[r][c])
        if r < height - 1:
            t.append("\n")
    return t


def error_bars(tr: Trainer, width: int = 46, rows: int = 5) -> Text:
    env = tr.env1
    t = Text(no_wrap=True, overflow="ellipsis")
    errs = env.err_hist[-rows:]
    ests = env.est_hist[-rows:]
    start = env.n - len(errs) + 1
    for i, (e, s) in enumerate(zip(errs, ests)):
        frac = max(0.0, min(1.0, (math.log10(max(e, 1e-300)) + 9) / 12)) if e > 0 else 0.0  # 1e-9 .. 1e3
        n = int(frac * (width - 30))
        t.append(f"n={start + i:<2}", style="dim")
        t.append("█" * n + "░" * ((width - 30) - n), style="yellow" if i == len(errs) - 1 else "dark_orange")
        t.append(f" err {e:8.1e}", style="dim")
        t.append(f" est {s:8.1e}" if math.isfinite(s) else " est   —    ", style="cyan" if math.isfinite(s) and s / env.scale < env.tol else "dim")
        if i < len(errs) - 1:
            t.append("\n")
    return t


def phase1_panel(tr: Trainer, height: int) -> Panel:
    env = tr.env1
    active = tr.phase == 1
    inner = height - 2
    bars_h = 5 if inner >= 15 else 4
    plot_h = max(5, inner - 2 - bars_h)
    status = {
        "stop_ok": "[bold green]■ entregó: error < tol ✔[/]",
        "stop_bad": "[bold red]■ entregó antes de tiempo ✘[/]",
        "add": "[cyan]… añadiendo nodos[/]",
        "timeout": "[yellow]⏱ agotó los 40 nodos[/]",
        "overflow": "[red]desbordó[/]",
        "": "",
    }[tr.last_outcome1]
    fam = "equiespaciados" if env.gamma == 0 else ("Chebyshev" if env.gamma == 1 else f"mezcla γ={env.gamma}")
    head = Text.from_markup(
        f"[bold]f = {env.problem.name}[/]  nodos: [magenta]{fam}[/]  n = {env.n:>2}  "
        f"error real = {env.err:.1e}  estimador = {env.est:.1e}  acción: {ACTION_SHORT[tr.last_action1]}  {status}",
        overflow="ellipsis",
    )
    head.no_wrap = True
    body = Group(head, plot_interp(tr, height=plot_h), error_bars(tr, rows=bars_h))
    title = "Fase 1 · aprendiendo a interpolar (dónde y cuántos nodos)" + ("" if active else " · completada ✔")
    return Panel(body, title=title, border_style="yellow" if active else "green", padding=(0, 1))


def law_panel(tr: Trainer) -> Panel:
    qs = tr.gamma_q()
    counts = tr.agent_gamma.counts[("gamma",)]
    best = tr.gamma_policy()
    lo, hi = min(qs + [-1.0]), max(qs + [0.0])
    g_tab = Table.grid(padding=(0, 1))
    for _ in range(5):
        g_tab.add_column(no_wrap=True)
    g_tab.columns[0].style = "dim"
    for g, q, c in zip(GAMMAS, qs, counts):
        name = "equiesp." if g == 0 else ("Chebyshev" if g == 1 else f"γ={g:.2f}")
        star = "★" if abs(g - best) < 1e-9 and c > 0 else " "
        g_tab.add_row(f"{name:<9}", bar((q - lo) / (hi - lo) if hi > lo else 0.0, 14), f"{q:+6.1f}", f"{c:>5}", Text(star, style="bold yellow"))

    law = tr.control_law()
    l_tab = Table.grid(padding=(0, 1))
    for _ in range(6):
        l_tab.add_column(no_wrap=True)
    l_tab.columns[0].style = l_tab.columns[1].style = "dim"
    l_tab.add_row("estimador", "tend.", "n≤6", "", "n>6", "")
    for eb in range(4):
        for trend in (0, 1):
            cells = []
            for late in (0, 1):
                s = (eb, trend, late)
                a, theo = law[s], theoretical_action(eb, trend, late)
                if a is None:
                    cells += [Text("?", style="dim"), Text("/".join(ACTION_SHORT[x] for x in theo), style="dim")]
                else:
                    ok = a in theo
                    cells += [Text(f"{ACTION_SHORT[a]} {'✔' if ok else '✘'}", style="green" if ok else "red"),
                              Text("/".join(ACTION_SHORT[x] for x in theo), style="dim")]
            l_tab.add_row(EST_LABELS[eb], TREND_LABELS[trend], *cells)
    ok, n = tr.law_matches_theory()
    foot = Text.from_markup(f"parada vs teoría: [bold]{ok}/{n}[/]  (+ añadir ■ entregar)  éxito: ", overflow="ellipsis")
    foot.no_wrap = True
    foot.append(sparkline(tr.success_curve, 16), style="green")
    foot.append(f" {100 * tr.success_curve[-1]:3.0f}%" if tr.success_curve else "", style="bold")
    body = Group(Text("familia de nodos (Q = retorno medio, usos)", style="bold"), g_tab, Text(""), l_tab, foot)
    return Panel(body, title="Políticas aprendidas (✔/✘ frente a la teoría)", border_style="yellow" if tr.phase == 1 else "green", padding=(0, 1))


# ---------------------------------------------------------------------------
# Fase 2
# ---------------------------------------------------------------------------
def proof_lines(attempts: list[tuple[int, bool]], max_lines: int, show_text: bool = True, tail: bool = True) -> Text:
    t = Text(no_wrap=True, overflow="ellipsis")
    if tail:
        shown = attempts[-max_lines:]
        if len(attempts) > max_lines:
            t.append(f"  … {len(attempts) - max_lines} intentos anteriores\n", style="dim")
    else:
        shown = attempts[:max_lines]
    n = 0
    for a, ok in shown:
        st = STEPS[a]
        if ok:
            n += 1
            style = "bold green" if st.is_qed else "green"
            t.append(f"{n:>2}. ", style="dim")
            t.append(f"{st.key:<8}", style=style)
            if show_text:
                t.append(f" {st.text}", style="white" if st.is_qed else "")
        else:
            t.append(" ✗  ", style="red")
            t.append(f"{st.key:<8}", style="red")
            if show_text:
                t.append(f" {st.text}", style="dim red strike")
        t.append("\n")
    if not tail and len(attempts) > max_lines:
        t.append(f"  … {len(attempts) - max_lines} intentos más", style="dim")
    return t


def dependency_progress(mask_keys: set[str]) -> Text:
    t = Text(no_wrap=True, overflow="ellipsis")
    for st in STEPS:
        if st.distractor:
            continue
        done = st.key in mask_keys
        t.append(st.key, style="black on green" if done else "dim on grey23")
        t.append(" ")
    return t


def phase2_panel(tr: Trainer, height: int) -> Panel:
    env = tr.env2
    active = tr.phase == 2
    if tr.phase == 1:
        body: RenderableType = Text("en espera de la fase 1…", style="dim")
        return Panel(body, title="Fase 2 · aprendiendo a demostrar", border_style="grey50")
    n_valid = len(env.proof_keys)
    head = Text.from_markup(
        f"intento actual · [green]{n_valid}[/] pasos válidos · [red]{env.n_invalid}[/] saltos lógicos"
        + ("  [bold green]∎ QED[/]" if env.finished else "")
    )
    prog = dependency_progress(set(env.proof_keys))
    lines = proof_lines(env.attempts, max_lines=max(4, height - 5))
    title = "Fase 2 · aprendiendo a demostrar el teorema del error" + ("" if active else " · completada ✔")
    return Panel(Group(head, prog, lines), title=title, border_style="magenta" if active else "green", padding=(0, 1))


def belief_panel(tr: Trainer, height: int) -> Panel:
    if tr.phase == 1:
        return Panel(Text("", style="dim"), title="Lo que el agente cree ahora", border_style="grey50")
    g = tr.greedy_proof
    keys = [STEPS[a].key for a, ok in g if ok]
    clean = all(ok for _, ok in g)
    finished = "QED" in keys
    verdict = (
        "[bold green]demostración mínima y correcta ✔[/]" if clean and finished and len(keys) == MIN_PROOF_LENGTH
        else "[green]completa[/]" + ("" if clean else " [red]con saltos lógicos[/]") if finished
        else "[yellow]incompleta[/]"
    )
    head = Text.from_markup(f"política voraz (sin explorar): {verdict}")
    lines = proof_lines(g, max_lines=max(4, height - 5), show_text=True, tail=False)
    return Panel(Group(head, lines), title="Lo que el agente cree ahora", border_style="magenta", padding=(0, 1))


# ---------------------------------------------------------------------------
# Pie
# ---------------------------------------------------------------------------
def stats_panel(tr: Trainer) -> Panel:
    tb = Table.grid(padding=(0, 1))
    for _ in range(4):
        tb.add_column(no_wrap=True)
    tb.columns[0].style = tb.columns[2].style = "dim"
    tb.columns[1].justify = tb.columns[3].justify = "right"
    p1_rate = f"{100 * tr.success_curve[-1]:.0f}%" if tr.success_curve else "—"
    p2_rate = f"{100 * sum(tr.finished2) / len(tr.finished2):.0f}%" if tr.finished2 else "—"
    it = tr.node_curve[-1] if tr.node_curve else float("nan")
    eps = tr.agent_stop.eps if tr.phase == 1 else tr.agent2.eps
    tb.add_row("⏱ tiempo", f"[bold]{fmt_time(tr.elapsed)}[/]", "generación", f"[bold]{tr.ep1 if tr.phase == 1 else tr.ep2}[/]")
    tb.add_row("fase", f"[bold]{tr.phase}/2[/]", "ε exploración", f"{eps:.3f}")
    tb.add_row("F1 ok", f"[green]{tr.stop_ok_total}[/]", "F1 errores", f"[red]{tr.stop_bad_total + tr.timeout_total}[/]")
    tb.add_row("F1 éxito", p1_rate, "F1 nodos", f"{it:.1f}" if not math.isnan(it) else "—")
    tb.add_row("F2 ok", f"[green]{tr.qed_total}[/]", "F2 mínimas", f"[bold green]{tr.minimal_total}[/]")
    tb.add_row("F2 éxito", p2_rate, "F2 mejor", f"{tr.best_reward:+.1f}" if tr.best_reward > -1e9 else "—")
    n_upd = tr.agent_gamma.updates + tr.agent_stop.updates + tr.agent2.updates
    if tr.reward_curve:
        lo, hi = min(tr.reward_curve[-60:]), max(tr.reward_curve[-60:])
        tb.add_row("Q updates", f"{n_upd:,}", "F2 retorno", sparkline(tr.reward_curve, 22, lo, hi))
    else:
        tb.add_row("Q updates", f"{n_upd:,}", "F1 tiempo", fmt_time(tr.phase1_time) if tr.phase1_time else "—")
    return Panel(tb, title="Marcador", border_style="blue", padding=(0, 1))


def log_panel(tr: Trainer, n: int) -> Panel:
    t = Text(no_wrap=True, overflow="ellipsis")
    for line in tr.log[-n:]:
        style = "bold green" if any(k in line for k in ("mínima", "teoría", "criterio", "Chebyshev", "completa!")) else ""
        t.append(line + "\n", style=style)
    return Panel(t, title="Bitácora de hitos", border_style="blue", padding=(0, 1))


def header(tr: Trainer) -> Panel:
    txt = Text.from_markup(
        "[bold white]InterpRL[/] · un agente aprende dónde y cuántos nodos poner para interpolar, y demuestra el teorema del error   "
        f"[bold yellow]⏱ {fmt_time(tr.elapsed)}[/]   "
        + ("[bold green]ENTRENAMIENTO TERMINADO[/]" if tr.done else f"[bold]Fase {tr.phase}[/] · generación [bold]{tr.ep1 if tr.phase == 1 else tr.ep2}[/]")
    )
    return Panel(txt, style="on grey11")


def build_layout(tr: Trainer, height: int) -> Layout:
    root = Layout()
    footer_h = 11 if height >= 42 else 9
    body_h = max(20, height - 3 - footer_h)
    law_h = 21
    root.split_column(
        Layout(header(tr), name="header", size=3),
        Layout(name="body", size=body_h),
        Layout(name="footer", size=footer_h),
    )
    root["body"].split_row(Layout(name="left"), Layout(name="right"))
    root["left"].split_column(
        Layout(phase1_panel(tr, body_h - law_h), name="plot"),
        Layout(law_panel(tr), name="law", size=law_h),
    )
    root["right"].split_column(
        Layout(phase2_panel(tr, body_h // 2), name="proof"),
        Layout(belief_panel(tr, body_h - body_h // 2), name="belief"),
    )
    root["footer"].split_row(Layout(stats_panel(tr), ratio=5), Layout(log_panel(tr, footer_h - 2), ratio=7))
    return root
