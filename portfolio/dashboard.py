from rich.console import Console
from rich.table import Table
from rich import box
from .data import fetch_portfolio

console = Console(width=120)


def _color(val, text):
    c = "green" if val >= 0 else "red"
    return f"[{c}]{text}[/{c}]"


def main():
    console.print("\n[bold white]Archer Capital — Portfolio Dashboard[/bold white]\n")
    console.print("Fetching prices...", end="\r")

    rows, totals = fetch_portfolio()

    table = Table(box=box.SIMPLE_HEAVY, show_edge=True, padding=(0, 1))
    table.add_column("Ticker",     style="bold",    min_width=6,  no_wrap=True, overflow="fold")
    table.add_column("Last",       justify="right",  min_width=10, no_wrap=True, overflow="fold")
    table.add_column("Day %",      justify="right",  min_width=8,  no_wrap=True, overflow="fold")
    table.add_column("Mkt Value",  justify="right",  min_width=11, no_wrap=True, overflow="fold")
    table.add_column("Cost Basis", justify="right",  min_width=11, no_wrap=True, overflow="fold")
    table.add_column("Unreal P&L", justify="right",  min_width=11, no_wrap=True, overflow="fold")
    table.add_column("Day P&L",    justify="right",  min_width=10, no_wrap=True, overflow="fold")
    table.add_column("Weight",     justify="right",  min_width=7,  no_wrap=True, overflow="fold")

    for r in rows:
        table.add_row(
            r["ticker"],
            f"${r['last']:,.2f}",
            _color(r["daily_pct"],  f"{r['daily_pct']:+.2f}%"),
            f"${r['market_value']:,.0f}",
            f"${r['cost_basis']:,.0f}",
            _color(r["unreal_pnl"], f"${r['unreal_pnl']:+,.0f}"),
            _color(r["day_pnl"],    f"${r['day_pnl']:+,.0f}"),
            f"{r['weight']:.1f}%",
        )

    upnl_str = f"${totals['unreal_pnl']:+,.0f}"
    dpnl_str = f"${totals['day_pnl']:+,.0f}"

    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]", "", "",
        f"[bold]${totals['market_value']:,.0f}[/bold]",
        f"[bold]${totals['cost_basis']:,.0f}[/bold]",
        f"[bold]{_color(totals['unreal_pnl'], upnl_str)}[/bold]",
        f"[bold]{_color(totals['day_pnl'],    dpnl_str)}[/bold]",
        "[bold]100.0%[/bold]",
    )

    console.print(" " * 30, end="\r")
    console.print(table)
    console.print()


if __name__ == "__main__":
    main()
