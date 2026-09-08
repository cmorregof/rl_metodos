"""Interfaz en vivo para la terminal (estilo "mira cómo aprende")."""

from __future__ import annotations

import math

from rich.console import Group, RenderableType
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .env_fixpoint import ACTION_SHORT, N_ACTIONS, RATIO_LABELS, theoretical_action
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
def plot_iterates(tr: Trainer, width: int = 46, height: int = 8) -> Text:
    """Gráfica de f con la raíz ◆ y los iterados xₙ marcados sobre el eje."""
    env = tr.env1
    p = env.problem
    finite = [x for x in env.history if math.isfinite(x)]
    span = max(abs(p.x0 - p.root) * 1.6, 0.5)
    lo_x, hi_x = p.root - span, p.root + span
    xs = [lo_x + (hi_x - lo_x) * i / (width - 1) for i in range(width)]
    ys = []
    for x in xs:
        try:
            ys.append(p.f(x))
        except (OverflowError, ValueError):
            ys.append(float("nan"))
    good = [y for y in ys if math.isfinite(y)]
    lo, hi = min(good + [0.0]), max(good + [0.0])
    if hi - lo < 1e-12:
        hi = lo + 1
    grid = [[" "] * width for _ in range(height)]
    styles = [[""] * width for _ in range(height)]

    def row_of(y: float) -> int:
        return max(0, min(height - 1, height - 1 - int((y - lo) / (hi - lo) * (height - 1))))

    def col_of(x: float) -> int:
        return int((x - lo_x) / (hi_x - lo_x) * (width - 1))

    zero = row_of(0.0)
    for c in range(width):
        grid[zero][c] = "─"
        styles[zero][c] = "dim"
    for c, y in enumerate(ys):
        if math.isfinite(y):
            grid[row_of(y)][c] = "•"
            styles[row_of(y)][c] = "cyan"
    for i, x in enumerate(finite[-8:]):
        c = col_of(x)
        if 0 <= c < width:
            last = i == len(finite[-8:]) - 1
            grid[zero][c] = "●" if last else "○"
            styles[zero][c] = "bold white" if last else "yellow"
    cr = col_of(p.root)
    if 0 <= cr < width:
        grid[zero][cr] = "◆"
        styles[zero][cr] = "bold magenta"
    t = Text()
    for r in range(height):
        for c in range(width):
            t.append(grid[r][c], style=styles[r][c])
        if r < height - 1:
            t.append("\n")
    return t


def residual_bars(tr: Trainer, width: int = 46, rows: int = 5) -> Text:
    """Últimas iteraciones: |f(xₙ)| en escala log y el α usado."""
    env = tr.env1
    t = Text(no_wrap=True, overflow="ellipsis")
    hist = env.history[-rows:]
    alphas = ([None] + env.alpha_history)[-rows:]
    start = len(env.history) - len(hist)
    for i, (x, al) in enumerate(zip(hist, alphas)):
        try:
            fx = abs(env.problem.f(x)) if math.isfinite(x) else float("inf")
        except (OverflowError, ValueError):
            fx = float("inf")
        if math.isfinite(fx) and fx > 0:
            frac = max(0.0, min(1.0, (math.log10(fx) + 7) / 9))  # 1e-7 .. 1e2
        else:
            frac = 1.0 if not math.isfinite(fx) else 0.0
        n = int(frac * (width - 22))
        t.append(f"{start + i:>2} ", style="dim")
        t.append("█" * n + "░" * ((width - 22) - n), style="yellow" if i == len(hist) - 1 else "dark_orange")
        t.append(f" |f|={fx:8.1e}", style="dim")
        t.append(f" α={al:+.3g}" if al is not None else "", style="dim")
        if i < len(hist) - 1:
            t.append("\n")
    return t


def phase1_panel(tr: Trainer, height: int) -> Panel:
    env = tr.env1
    active = tr.phase == 1
    inner = height - 2
    bars_h = 5 if inner >= 15 else 4
    plot_h = max(5, inner - 2 - bars_h)
    status = {
        "converged": "[bold green]✔ convergió[/]",
        "diverged": "[bold red]✘ divergió (g no era contracción)[/]",
        "timeout": "[yellow]⏱ agotó iteraciones[/]",
        "step": "[cyan]… iterando[/]",
        "": "",
    }[tr.last_outcome1]
    rho = env.rho
    rho_txt = f"{rho:.2f}" if math.isfinite(rho) else "—"
    head = Text.from_markup(
        f"[bold]f(x) = {env.problem.name}[/]   x₀ = {env.problem.x0:.3f}   raíz ◆ {env.problem.root:.4f}\n"
        f"it {env.n_iter:>2}  α = {env.alpha:+.3g}  ρ = {rho_txt} {'oscila' if env.oscillating else 'monót.'}"
        f"  acción: {ACTION_SHORT[tr.last_action1]}   {status}",
        overflow="ellipsis",
    )
    head.no_wrap = True
    body = Group(head, plot_iterates(tr, height=plot_h), residual_bars(tr, rows=bars_h))
    title = "Fase 1 · aprendiendo a controlar x ← x − α·f(x)" + ("" if active else " · completada ✔")
    return Panel(body, title=title, border_style="yellow" if active else "green", padding=(0, 1))


def law_panel(tr: Trainer) -> Panel:
    law = tr.control_law()
    tb = Table.grid(padding=(0, 1))
    tb.add_column(style="dim", no_wrap=True)
    tb.add_column(no_wrap=True)
    tb.add_column(no_wrap=True)
    tb.add_column(no_wrap=True)
    tb.add_row("estado (ρ, forma)", "acción", "teoría", "Q(=, ×2, ÷2, ±)")
    for s in ALL_STATES:
        b, o = s
        a = law[s]
        theo = theoretical_action(b, o)
        qs = tr.agent1.q.get(s, [0.0] * N_ACTIONS)
        if a is None:
            act, mark, style = "?", "", "dim"
        else:
            ok = a in theo
            act, mark, style = ACTION_SHORT[a], ("✔" if ok else "✘"), ("green" if ok else "red")
        tb.add_row(
            f"ρ {RATIO_LABELS[b]:<8} {'oscila' if o else 'monót.'}",
            Text(f"{act:>3} {mark}", style=style),
            "/".join(ACTION_SHORT[x] for x in theo),
            Text(" ".join(f"{q:+5.1f}" for q in qs), style="dim"),
        )
    ok, n = tr.law_matches_theory()
    foot = Text.from_markup(f"Banach: [bold]{ok}/{n}[/] estados   éxito/100: ", overflow="ellipsis")
    foot.no_wrap = True
    foot.append(sparkline(tr.success_curve, 22), style="green")
    foot.append(f" {100 * tr.success_curve[-1]:3.0f}%" if tr.success_curve else "", style="bold")
    return Panel(Group(tb, foot), title="Ley de control aprendida", border_style="yellow" if tr.phase == 1 else "green", padding=(0, 1))


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
    title = "Fase 2 · aprendiendo a demostrar el teorema del punto fijo" + ("" if active else " · completada ✔")
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
    tb.add_row("F1 ok", f"[green]{tr.converged_total}[/]", "F1 divergió", f"[red]{tr.diverged_total}[/]")
    tb.add_row("F1 éxito", p1_rate, "F1 iter. medias", f"{it:.1f}" if not math.isnan(it) else "—")
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
        style = "bold green" if any(k in line for k in ("mínima", "Banach", "invertirlo", "reducirlo", "completa!")) else ""
        t.append(line + "\n", style=style)
    return Panel(t, title="Bitácora de hitos", border_style="blue", padding=(0, 1))


def header(tr: Trainer) -> Panel:
    txt = Text.from_markup(
        "[bold white]FixpointRL[/] · un agente aprende la iteración de punto fijo y a demostrar el teorema de Banach   "
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
