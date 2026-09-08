"""Interfaz en vivo para la terminal (estilo "mira cómo aprende")."""

from __future__ import annotations

import math

from rich.console import Group, RenderableType
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .env_taylor import ACTION_SHORT, N_ACTIONS, RATIO_LABELS, TERM_LABELS, theoretical_action
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


# ---------------------------------------------------------------------------
# Fase 1
# ---------------------------------------------------------------------------
def plot_series(tr: Trainer, width: int = 46, height: int = 8) -> Text:
    """f (cian) frente a la suma parcial Pₙ (amarillo) en el dominio; x* marcado."""
    env = tr.env1
    ser, xstar = env.problem.series, env.problem.x
    lo_x, hi_x = ser.domain
    xs = [lo_x + (hi_x - lo_x) * i / (width - 1) for i in range(width)]

    def safe(fn, x):
        try:
            v = fn(x)
            return v if math.isfinite(v) else float("nan")
        except (OverflowError, ValueError, ZeroDivisionError):
            return float("nan")

    def poly(x: float) -> float:
        s, n = 0.0, -1
        for _ in range(env.n_terms):
            n += 1
            while ser.coef(n) == 0.0:
                n += 1
            s += ser.coef(n) * x ** n
        return s

    fy = [safe(ser.f, x) for x in xs]
    py = [safe(poly, x) for x in xs]
    good = [y for y in fy if math.isfinite(y) and abs(y) < 50]
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
    for c, y in enumerate(fy):
        r = row_of(y)
        if r is not None:
            grid[r][c], styles[r][c] = "•", "cyan"
    cx = int((xstar - lo_x) / (hi_x - lo_x) * (width - 1))
    for r in range(height):
        if grid[r][cx] == " ":
            grid[r][cx], styles[r][cx] = "┊", "magenta"
    r = row_of(env.true_value)
    if r is not None:
        grid[r][cx], styles[r][cx] = "◆", "bold magenta"
    t = Text()
    for r in range(height):
        for c in range(width):
            t.append(grid[r][c], style=styles[r][c])
        if r < height - 1:
            t.append("\n")
    return t


def term_bars(tr: Trainer, width: int = 46, rows: int = 5) -> Text:
    env = tr.env1
    t = Text(no_wrap=True, overflow="ellipsis")
    terms = env.terms[-rows:]
    start = len(env.terms) - len(terms)
    for i, term in enumerate(terms):
        a = abs(term)
        frac = max(0.0, min(1.0, (math.log10(max(a, 1e-300)) + 9) / 12)) if a > 0 else 0.0  # 1e-9 .. 1e3
        n = int(frac * (width - 26))
        t.append(f"t{start + i:<2}", style="dim")
        t.append("█" * n + "░" * ((width - 26) - n), style="yellow" if i == len(terms) - 1 else "dark_orange")
        t.append(f" {term:+9.2e}", style="dim")
        if start + i > 0:
            q = a / abs(env.terms[start + i - 1]) if abs(env.terms[start + i - 1]) > 0 else float("nan")
            t.append(f" q={q:4.2f}", style="red" if q >= 1 else "dim")
        if i < len(terms) - 1:
            t.append("\n")
    return t


def phase1_panel(tr: Trainer, height: int) -> Panel:
    env = tr.env1
    active = tr.phase == 1
    inner = height - 2
    bars_h = 5 if inner >= 15 else 4
    plot_h = max(5, inner - 2 - bars_h)
    status = {
        "stop_ok": "[bold green]■ paró: error < tol ✔[/]",
        "stop_bad": "[bold red]■ paró antes de tiempo ✘[/]",
        "abort_ok": "[bold green]✗ abandonó: la serie no converge en x* ✔[/]",
        "abort_bad": "[bold red]✗ abandonó una serie convergente ✘[/]",
        "add": "[cyan]… sumando[/]",
        "timeout": "[yellow]⏱ agotó términos[/]",
        "overflow": "[red]desbordó[/]",
        "": "",
    }[tr.last_outcome1]
    div = " [red](fuera del radio)[/]" if env.problem.diverges else ""
    head = Text.from_markup(
        f"[bold]f = {env.problem.series.name}[/]  x* = {env.problem.x:+.2f}{div}  f(x*) = {env.true_value:.6g}\n"
        f"n = {env.n_terms:>2} términos  Sₙ = {env.partial:.6g}  error = {env.error:.1e}  acción: {ACTION_SHORT[tr.last_action1]}  {status}",
        overflow="ellipsis",
    )
    head.no_wrap = True
    body = Group(head, plot_series(tr, height=plot_h), term_bars(tr, rows=bars_h))
    title = "Fase 1 · aprendiendo a sumar la serie de Taylor" + ("" if active else " · completada ✔")
    return Panel(body, title=title, border_style="yellow" if active else "green", padding=(0, 1))


def law_panel(tr: Trainer) -> Panel:
    law = tr.control_law()
    tb_ = Table.grid(padding=(0, 1))
    for _ in range(7):
        tb_.add_column(no_wrap=True)
    tb_.columns[0].style = tb_.columns[3].style = "dim"
    tb_.add_row("último término", "cociente", "n≤4", "", "n>4", "", "")
    for tb in range(4):
        for rb in range(3):
            cells = []
            for late in (0, 1):
                s = (tb, rb, late)
                a, theo = law[s], theoretical_action(tb, rb, late)
                if a is None:
                    cells += [Text("?", style="dim"), Text("/".join(ACTION_SHORT[x] for x in theo), style="dim")]
                else:
                    ok = a in theo
                    cells += [Text(f"{ACTION_SHORT[a]} {'✔' if ok else '✘'}", style="green" if ok else "red"),
                              Text("/".join(ACTION_SHORT[x] for x in theo), style="dim")]
            tb_.add_row(TERM_LABELS[tb], RATIO_LABELS[rb], cells[0], cells[1], cells[2], cells[3], "")
    ok, n = tr.law_matches_theory()
    foot = Text.from_markup(f"teoría: [bold]{ok}/{n}[/] estados  (+ sumar ■ parar ✗ abandonar)  éxito: ", overflow="ellipsis")
    foot.no_wrap = True
    foot.append(sparkline(tr.success_curve, 16), style="green")
    foot.append(f" {100 * tr.success_curve[-1]:3.0f}%" if tr.success_curve else "", style="bold")
    return Panel(Group(tb_, foot), title="Ley aprendida (acción ✔/✘ frente a la teoría)", border_style="yellow" if tr.phase == 1 else "green", padding=(0, 1))


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
            t.append(f"{st.key:<7}", style=style)
            if show_text:
                t.append(f" {st.text}", style="white" if st.is_qed else "")
        else:
            t.append(" ✗  ", style="red")
            t.append(f"{st.key:<7}", style="red")
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
    title = "Fase 2 · aprendiendo a demostrar el teorema de Taylor" + ("" if active else " · completada ✔")
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
    it = tr.iter_curve[-1] if tr.iter_curve else float("nan")
    eps = tr.agent1.eps if tr.phase == 1 else tr.agent2.eps
    tb.add_row("⏱ tiempo", f"[bold]{fmt_time(tr.elapsed)}[/]", "generación", f"[bold]{tr.ep1 if tr.phase == 1 else tr.ep2}[/]")
    tb.add_row("fase", f"[bold]{tr.phase}/2[/]", "ε exploración", f"{eps:.3f}")
    tb.add_row("F1 ok", f"[green]{tr.stop_ok_total + tr.abort_ok_total}[/]", "F1 errores", f"[red]{tr.stop_bad_total + tr.abort_bad_total}[/]")
    tb.add_row("F1 éxito", p1_rate, "F1 términos", f"{it:.1f}" if not math.isnan(it) else "—")
    tb.add_row("F2 ok", f"[green]{tr.qed_total}[/]", "F2 mínimas", f"[bold green]{tr.minimal_total}[/]")
    tb.add_row("F2 éxito", p2_rate, "F2 mejor", f"{tr.best_reward:+.1f}" if tr.best_reward > -1e9 else "—")
    n_upd = tr.agent1.updates + tr.agent2.updates
    if tr.reward_curve:
        lo, hi = min(tr.reward_curve[-60:]), max(tr.reward_curve[-60:])
        tb.add_row("Q updates", f"{n_upd:,}", "F2 retorno", sparkline(tr.reward_curve, 22, lo, hi))
    else:
        tb.add_row("Q updates", f"{n_upd:,}", "F1 tiempo", fmt_time(tr.phase1_time) if tr.phase1_time else "—")
    return Panel(tb, title="Marcador", border_style="blue", padding=(0, 1))


def log_panel(tr: Trainer, n: int) -> Panel:
    t = Text(no_wrap=True, overflow="ellipsis")
    for line in tr.log[-n:]:
        style = "bold green" if any(k in line for k in ("mínima", "teoría", "criterio", "completa!")) else ""
        t.append(line + "\n", style=style)
    return Panel(t, title="Bitácora de hitos", border_style="blue", padding=(0, 1))


def header(tr: Trainer) -> Panel:
    txt = Text.from_markup(
        "[bold white]TaylorRL[/] · un agente aprende a usar la serie de Taylor y a demostrar el teorema de Taylor   "
        f"[bold yellow]⏱ {fmt_time(tr.elapsed)}[/]   "
        + ("[bold green]ENTRENAMIENTO TERMINADO[/]" if tr.done else f"[bold]Fase {tr.phase}[/] · generación [bold]{tr.ep1 if tr.phase == 1 else tr.ep2}[/]")
    )
    return Panel(txt, style="on grey11")


def build_layout(tr: Trainer, height: int) -> Layout:
    root = Layout()
    footer_h = 11 if height >= 42 else 9
    body_h = max(20, height - 3 - footer_h)
    law_h = 18
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
