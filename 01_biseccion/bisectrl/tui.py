"""Interfaz en vivo para la terminal (estilo "mira cómo aprende")."""

from __future__ import annotations

import math
from collections import Counter

from rich.columns import Columns
from rich.console import Group, RenderableType
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .env_bisection import LAMBDAS
from .proof_kb import MIN_PROOF_LENGTH, REQUIRED_KEYS, STEPS
from .trainer import Trainer

SPARK = "▁▂▃▄▅▆▇█"


def fmt_time(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def sparkline(values: list[float], width: int, lo: float = 0.0, hi: float = 1.0) -> str:
    vals = values[-width:]
    if not vals:
        return ""
    out = []
    for v in vals:
        k = 0 if hi == lo else int((v - lo) / (hi - lo) * (len(SPARK) - 1))
        out.append(SPARK[max(0, min(len(SPARK) - 1, k))])
    return "".join(out)


def bar(frac: float, width: int, ch: str = "█") -> str:
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return ch * n + "░" * (width - n)


# ---------------------------------------------------------------------------
# Fase 1
# ---------------------------------------------------------------------------
def plot_function(tr: Trainer, width: int = 46, height: int = 9) -> Text:
    env = tr.env1
    p = env.problem
    a0, b0 = p.a, p.b
    xs = [a0 + (b0 - a0) * i / (width - 1) for i in range(width)]
    ys = [p.f(x) for x in xs]
    lo, hi = min(ys), max(ys)
    lo, hi = min(lo, 0), max(hi, 0)
    if hi - lo < 1e-12:
        hi = lo + 1
    grid = [[" "] * width for _ in range(height)]
    styles = [[""] * width for _ in range(height)]

    def row_of(y: float) -> int:
        return height - 1 - int((y - lo) / (hi - lo) * (height - 1))

    zero_row = row_of(0.0)
    for c in range(width):
        grid[zero_row][c] = "─"
        styles[zero_row][c] = "dim"
    # banda del intervalo actual sobre el eje
    ca = int((env.a - a0) / (b0 - a0) * (width - 1))
    cb = int((env.b - a0) / (b0 - a0) * (width - 1))
    for c in range(max(0, ca), min(width, cb + 1)):
        grid[zero_row][c] = "═"
        styles[zero_row][c] = "bold yellow"
    for c, y in enumerate(ys):
        r = row_of(y)
        grid[r][c] = "•"
        styles[r][c] = "cyan"
    cr = int((p.root - a0) / (b0 - a0) * (width - 1))
    grid[zero_row][cr] = "◆"
    styles[zero_row][cr] = "bold magenta"
    cx = int((env.x - a0) / (b0 - a0) * (width - 1))
    if 0 <= cx < width and env.n_iter > 0:
        rx = row_of(env.fx)
        grid[rx][cx] = "●"
        styles[rx][cx] = "bold white"
    t = Text()
    for r in range(height):
        for c in range(width):
            t.append(grid[r][c], style=styles[r][c])
        if r < height - 1:
            t.append("\n")
    return t


def interval_bars(tr: Trainer, width: int = 46, rows: int = 6) -> Text:
    env = tr.env1
    a0, b0 = env.problem.a, env.problem.b
    t = Text()
    hist = env.history[-rows:]
    start = len(env.history) - len(hist)
    for i, (a, b) in enumerate(hist):
        ca = int((a - a0) / (b0 - a0) * (width - 1))
        cb = int((b - a0) / (b0 - a0) * (width - 1))
        line = [" "] * width
        for c in range(ca, cb + 1):
            line[c] = "█"
        t.append(f"{start + i:>2} ", style="dim")
        t.append("".join(line), style="yellow" if i == len(hist) - 1 else "dark_orange")
        t.append(f" {b - a:.2e}", style="dim")
        if i < len(hist) - 1:
            t.append("\n")
    return t


def phase1_panel(tr: Trainer, height: int) -> Panel:
    env = tr.env1
    inner = height - 2
    bars_h = 4 if inner >= 13 else 3
    plot_h = max(5, inner - 2 - bars_h)
    active = tr.phase == 1
    status = {
        "converged": ("[bold green]✔ convergió[/]", ""),
        "lost": ("[bold red]✘ perdió la raíz (rompió Bolzano)[/]", ""),
        "timeout": ("[yellow]⏱ agotó iteraciones[/]", ""),
        "step": ("[cyan]… buscando[/]", ""),
        "": ("", ""),
    }[tr.last_outcome1][0]
    head = Text.from_markup(
        f"[bold]f(x) = {env.problem.name}[/]   [{env.problem.a:.2f}, {env.problem.b:.2f}]"
        f"   raíz ◆ {env.problem.root:.4f}\n"
        f"iteración {env.n_iter:>2}  λ = {tr.current_lam:.1f}   {status}",
        overflow="ellipsis",
    )
    head.no_wrap = True
    body = Group(head, plot_function(tr, height=plot_h), interval_bars(tr, rows=bars_h))
    title = "Fase 1 · aprendiendo a bisecar" + ("" if active else " · completada ✔")
    return Panel(body, title=title, border_style="yellow" if active else "green", padding=(0, 1))


def lambda_panel(tr: Trainer) -> Panel:
    qs = tr.lambda_q()
    lo, hi = min(qs), max(qs)
    best = tr.lambda_policy()
    recent = Counter(tr.lam_hist)
    total_recent = max(1, sum(recent.values()))
    t = Text(no_wrap=True, overflow="ellipsis")
    for lam, q in zip(LAMBDAS, qs):
        frac = 0.0 if hi == lo else (q - lo) / (hi - lo)
        mark = "★" if lam == best else " "
        style = "bold green" if lam == best else "cyan"
        t.append(f"{mark} λ={lam:.1f} ", style=style)
        t.append(bar(frac, 18), style=style)
        t.append(f" {(0.0 if abs(q) < 5e-4 else q):+.3f}", style="dim")
        t.append(f"  {100 * recent[lam] / total_recent:4.0f}%\n", style="dim")
    pol = tr.keep_policy()
    t.append("regla de conservación (signos f(a), f(x), f(b)):\n", style="dim")
    for (sa, sx, sb), k in sorted(pol.items()):
        if sx == 0:
            continue
        correct = "izq" if sa * sx < 0 else "der"
        ok = k == correct
        t.append(f"  ({'+' if sa > 0 else '−'},{'+' if sx > 0 else '−'},{'+' if sb > 0 else '−'}) → {k} ",
                 style="green" if ok else "red")
        t.append("✔\n" if ok else "✘\n", style="green" if ok else "red")
    return Panel(t, title="Política de corte · Q(λ) = bits de progreso − 1 por evaluación",
                 border_style="yellow" if tr.phase == 1 else "green", padding=(0, 1))


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
                t.append(f" {st.text[:70]}", style="white" if st.is_qed else "")
        else:
            t.append(" ✗  ", style="red")
            t.append(f"{st.key:<7}", style="red")
            if show_text:
                t.append(f" {st.text[:60]}", style="dim red strike")
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
    title = "Fase 2 · aprendiendo a demostrar la convergencia" + ("" if active else " · completada ✔")
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
# Pie: estadísticas e hitos
# ---------------------------------------------------------------------------
def stats_panel(tr: Trainer) -> Panel:
    tb = Table.grid(padding=(0, 1))
    tb.add_column(style="dim", no_wrap=True)
    tb.add_column(justify="right", no_wrap=True)
    tb.add_column(style="dim", no_wrap=True)
    tb.add_column(justify="right", no_wrap=True)
    p1_rate = f"{100 * tr.success_curve[-1]:.0f}%" if tr.success_curve else "—"
    p2_rate = f"{100 * sum(tr.finished2) / len(tr.finished2):.0f}%" if tr.finished2 else "—"
    eps = tr.agent_split.eps if tr.phase == 1 else tr.agent2.eps
    tb.add_row("⏱ tiempo", f"[bold]{fmt_time(tr.elapsed)}[/]", "generación", f"[bold]{tr.ep1 if tr.phase == 1 else tr.ep2}[/]")
    tb.add_row("fase", f"[bold]{tr.phase}/2[/]", "ε exploración", f"{eps:.3f}")
    tb.add_row("F1 convergió", f"[green]{tr.converged_total}[/]", "F1 raíz perdida", f"[red]{tr.lost_total}[/]")
    tb.add_row("F1 éxito/100", p1_rate, "F1 tiempo", fmt_time(tr.phase1_time) if tr.phase1_time else "—")
    tb.add_row("F2 completas", f"[green]{tr.qed_total}[/]", "F2 mínimas", f"[bold green]{tr.minimal_total}[/]")
    tb.add_row("F2 éxito/100", p2_rate, "F2 mejor", f"{tr.best_reward:+.1f}" if tr.best_reward > -1e9 else "—")
    n_upd = tr.agent_split.updates + tr.agent_keep.updates + tr.agent2.updates
    if tr.reward_curve:
        lo, hi = min(tr.reward_curve[-60:]), max(tr.reward_curve[-60:])
        tb.add_row("Q updates", f"{n_upd:,}", "F2 retorno", sparkline(tr.reward_curve, 22, lo, hi))
    elif tr.success_curve:
        tb.add_row("Q updates", f"{n_upd:,}", "F1 éxito", sparkline(tr.success_curve, 22))
    return Panel(tb, title="Marcador", border_style="blue", padding=(0, 1))


def log_panel(tr: Trainer, n: int) -> Panel:
    t = Text(no_wrap=True, overflow="ellipsis")
    for line in tr.log[-n:]:
        style = "bold green" if "∎" in line or "mínima" in line or "λ = 1/2" in line or "Bolzano" in line else ""
        t.append(line + "\n", style=style)
    return Panel(t, title="Bitácora de hitos", border_style="blue", padding=(0, 1))


def header(tr: Trainer) -> Panel:
    txt = Text.from_markup(
        "[bold white]BisectRL[/] · un agente aprende el método de bisección y a demostrar su convergencia   "
        f"[bold yellow]⏱ {fmt_time(tr.elapsed)}[/]   "
        + ("[bold green]ENTRENAMIENTO TERMINADO[/]" if tr.done else f"[bold]Fase {tr.phase}[/] · generación [bold]{tr.ep1 if tr.phase == 1 else tr.ep2}[/]")
    )
    return Panel(txt, style="on grey11")


def build_layout(tr: Trainer, height: int) -> Layout:
    root = Layout()
    footer_h = 11 if height >= 42 else 9
    body_h = max(20, height - 3 - footer_h)
    lambda_h = 17
    root.split_column(
        Layout(header(tr), name="header", size=3),
        Layout(name="body", size=body_h),
        Layout(name="footer", size=footer_h),
    )
    root["body"].split_row(Layout(name="left"), Layout(name="right"))
    root["left"].split_column(
        Layout(phase1_panel(tr, body_h - lambda_h), name="plot"),
        Layout(lambda_panel(tr), name="lambda", size=lambda_h),
    )
    right_h = body_h
    root["right"].split_column(
        Layout(phase2_panel(tr, right_h // 2), name="proof"),
        Layout(belief_panel(tr, right_h - right_h // 2), name="belief"),
    )
    root["footer"].split_row(Layout(stats_panel(tr), ratio=2), Layout(log_panel(tr, footer_h - 2), ratio=3))
    return root
