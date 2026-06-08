"""
Multi-source data reconciliation.

Usage:
    venv/bin/python run_reconcile.py MSFT
    venv/bin/python run_reconcile.py MSFT --no-fmp      # skip FMP
    venv/bin/python run_reconcile.py MSFT --no-edgar    # skip EDGAR
    venv/bin/python run_reconcile.py MSFT --no-fmp --no-edgar
"""

from __future__ import annotations
import sys
import math

from rich.console import Console
from rich.table   import Table
from rich         import box

from valuation.sources   import SourceData, fetch_yahoo, fetch_fmp, fetch_edgar
from valuation.reconcile import reconcile, ALL_FIELDS, SERIES_FIELDS, SCALAR_FIELDS, FIELD_TYPE


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_series_last(s, ccy: str) -> tuple[str, str]:
    """Format the most recent annual value and return (formatted_value, year_label)."""
    if s is None or s.empty:
        return "n/a", ""
    val  = float(s.iloc[-1])
    year = str(s.index[-1].year) if hasattr(s.index[-1], "year") else ""
    if math.isnan(val):
        return "n/a", year
    b = val / 1e9
    return f"{ccy} {b:,.2f}B", year


def _fmt_scalar(field: str, val: float, ccy: str) -> str:
    if math.isnan(val):
        return "n/a"
    if field == "tax_rate":
        return f"{val:.2%}"
    if field == "diluted_shares":
        return f"{val/1e9:.3f}B"
    # total_debt, cash
    return f"{ccy} {val/1e9:.3f}B"


def _fmt_disagree(d: float, ftype: str) -> tuple[str, str]:
    """Return (text, rich_style) for the disagreement column."""
    if math.isnan(d):
        return "n/a", "dim"
    pct = d * 100
    if ftype == "DEFINITIONAL":
        # Definitional differences are resolved by convention — shown dim, never flagged
        return f"[dim]{pct:.1f}% conv[/dim]", ""
    if pct > 2.0:
        return f"[bold red]{pct:.1f}% ⚠[/bold red]", ""
    return f"{pct:.1f}%", "green"


# ── Label metadata ────────────────────────────────────────────────────────────

_LABELS = {
    "revenue":        "Revenue",
    "ebit":           "EBIT",
    "dep_amort":      "D&A",
    "capex":          "CapEx",
    "diluted_shares": "Diluted Shares",
    "total_debt":     "Total Debt",
    "cash":           "Cash",
    "tax_rate":       "Tax Rate (eff.)",
}


# ── Main ──────────────────────────────────────────────────────────────────────

def run_reconcile(ticker: str, skip_fmp: bool = False, skip_edgar: bool = False) -> None:
    console = Console()
    ticker  = ticker.upper()

    # ── 1. Collect sources ────────────────────────────────────────────────────
    sources: list[SourceData] = []

    console.rule(f"[bold]Reconciliation — {ticker}[/bold]")

    console.print("\n[bold cyan]▶ Yahoo Finance[/bold cyan]")
    try:
        yahoo = fetch_yahoo(ticker)
        sources.append(yahoo)
    except Exception as e:
        console.print(f"  [red]Yahoo fetch failed: {e}[/red]")
        return

    if not skip_fmp:
        console.print("\n[bold cyan]▶ Financial Modeling Prep (FMP)[/bold cyan]")
        try:
            fmp = fetch_fmp(ticker)
            sources.append(fmp)
            if fmp.missing_fields:
                console.print(f"  [yellow]FMP missing fields: {', '.join(fmp.missing_fields)}[/yellow]")
        except RuntimeError as e:
            console.print(f"  [yellow]FMP skipped: {e}[/yellow]")

    if not skip_edgar:
        console.print("\n[bold cyan]▶ SEC EDGAR[/bold cyan]")
        try:
            edgar = fetch_edgar(ticker)
            sources.append(edgar)
            if edgar.missing_fields:
                console.print(f"  [yellow]EDGAR missing fields: {', '.join(edgar.missing_fields)}[/yellow]")
        except RuntimeError as e:
            console.print(f"  [yellow]EDGAR skipped: {e}[/yellow]")

    if not sources:
        console.print("[red]No sources available — aborting.[/red]")
        return

    # ── 2. Reconcile ─────────────────────────────────────────────────────────
    result  = reconcile(sources)
    # Only DATA fields count as genuine disagreement flags
    n_flags = sum(
        1 for f, d in result.disagreement.items()
        if FIELD_TYPE.get(f) == "DATA" and math.isfinite(d) and d * 100 > 2.0
    )

    # ── 3. Build table ────────────────────────────────────────────────────────
    source_names = [s.source_name for s in sources]
    ccy = sources[0].currency

    tbl = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold",
        title=f"\n[bold]{ticker} — Side-by-Side ({', '.join(source_names)})[/bold]",
        title_style="bold white",
    )

    tbl.add_column("Field",  style="cyan",  no_wrap=True, min_width=18)
    tbl.add_column("Type",   style="dim",   no_wrap=True, min_width=12)
    for name in source_names:
        tbl.add_column(name, justify="right", min_width=18)
    if len(sources) > 1:
        tbl.add_column("Disagree", justify="right", min_width=14)

    n_extra = 1 + len(sources) + (1 if len(sources) > 1 else 0)  # Type col + source cols + disagree

    # ── Series fields ─────────────────────────────────────────────────────────
    tbl.add_row(*["[bold dim]── Annual series (most recent FY) ──[/bold dim]"]
                 + [""] * n_extra)

    for f in SERIES_FIELDS:
        ftype = FIELD_TYPE.get(f, "DATA")
        type_label = ("[dim]CONV[/dim]" if ftype == "DEFINITIONAL"
                      else "[green]DATA[/green]")
        cols = []
        for s in sources:
            val_str, year = _fmt_series_last(getattr(s, f), s.currency)
            label = f"{val_str}  [dim]{year}[/dim]" if year else val_str
            cols.append(label)

        d_txt, _ = _fmt_disagree(result.disagreement[f], ftype)
        row = [_LABELS[f], type_label] + cols
        if len(sources) > 1:
            row.append(d_txt)
        tbl.add_row(*row)

    # ── Scalar fields ─────────────────────────────────────────────────────────
    tbl.add_row(*["[bold dim]── Balance sheet / derived scalars ──[/bold dim]"]
                 + [""] * n_extra)

    for f in SCALAR_FIELDS:
        ftype = FIELD_TYPE.get(f, "DATA")
        type_label = ("[dim]CONV[/dim]" if ftype == "DEFINITIONAL"
                      else "[green]DATA[/green]")
        cols = [_fmt_scalar(f, getattr(s, f), s.currency) for s in sources]
        d_txt, _ = _fmt_disagree(result.disagreement[f], ftype)
        row = [_LABELS[f], type_label] + cols
        if len(sources) > 1:
            row.append(d_txt)
        tbl.add_row(*row)

    console.print(tbl)

    # ── 4. Summary ────────────────────────────────────────────────────────────
    console.print(f"  Sources used : {', '.join(source_names)}")
    console.print(f"  Preferred    : {result.preferred.source_name}  (used by DCF engine)")
    if len(sources) < 2:
        console.print(
            "  [yellow]Only one source available — disagreement cannot be computed.[/yellow]"
        )
    elif n_flags:
        console.print(
            f"  [bold red]⚠  {n_flags} DATA field(s) disagree by >2% — review before running DCF.[/bold red]"
        )
    else:
        console.print(
            "  [green]✓  All DATA fields agree within 2% across sources.[/green]"
        )

    if result.sigma_cross:
        console.print("\n  [bold]σ_cross (DATA fields → MC widening):[/bold]")
        for var, sig in sorted(result.sigma_cross.items()):
            console.print(f"    {var:<20}  σ = {sig:.4f}")
    console.print()


if __name__ == "__main__":
    args        = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags       = [a for a in sys.argv[1:] if a.startswith("--")]
    ticker      = args[0] if args else "MSFT"
    skip_fmp    = "--no-fmp"    in flags
    skip_edgar  = "--no-edgar"  in flags
    run_reconcile(ticker, skip_fmp=skip_fmp, skip_edgar=skip_edgar)
