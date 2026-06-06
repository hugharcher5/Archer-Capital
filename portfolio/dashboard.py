import time
import yaml
import yfinance as yf
from rich.console import Console
from rich.table import Table
from rich import box

CONFIG_PATH = "config/positions.yaml"
MAX_RETRIES = 3
RETRY_DELAY = 2

console = Console(width=120)


def load_positions(path):
    with open(path) as f:
        return yaml.safe_load(f)["positions"]


def fetch_price(ticker):
    for attempt in range(MAX_RETRIES):
        try:
            info = yf.Ticker(ticker).fast_info
            last = info.last_price
            prev = info.previous_close
            if last and prev:
                return last, prev
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    return None, None


def color(val, text):
    c = "green" if val >= 0 else "red"
    return f"[{c}]{text}[/{c}]"


def main():
    positions = load_positions(CONFIG_PATH)

    console.print("\n[bold white]Archer Capital — Portfolio Dashboard[/bold white]\n")
    console.print("Fetching prices...", end="\r")

    rows = []
    for pos in positions:
        ticker = pos["ticker"]
        shares = pos["shares"]
        avg_price = pos["avg_price"]

        last, prev = fetch_price(ticker)
        if last is None:
            console.print(f"[yellow]  Warning: could not fetch {ticker}, skipping[/yellow]")
            continue

        rows.append({
            "ticker":        ticker,
            "last":          last,
            "daily_pct":     (last - prev) / prev * 100,
            "market_value":  shares * last,
            "cost_basis":    shares * avg_price,
            "unreal_pnl":    shares * last - shares * avg_price,
            "day_pnl":       shares * (last - prev),
        })

    total_mv    = sum(r["market_value"] for r in rows)
    total_cb    = sum(r["cost_basis"]   for r in rows)
    total_upnl  = sum(r["unreal_pnl"]  for r in rows)
    total_dpnl  = sum(r["day_pnl"]     for r in rows)

    table = Table(box=box.SIMPLE_HEAVY, show_edge=True, padding=(0, 1))
    table.add_column("Ticker",     style="bold",  min_width=6,  no_wrap=True, overflow="fold")
    table.add_column("Last",       justify="right", min_width=10, no_wrap=True, overflow="fold")
    table.add_column("Day %",      justify="right", min_width=8,  no_wrap=True, overflow="fold")
    table.add_column("Mkt Value",  justify="right", min_width=11, no_wrap=True, overflow="fold")
    table.add_column("Cost Basis", justify="right", min_width=11, no_wrap=True, overflow="fold")
    table.add_column("Unreal P&L", justify="right", min_width=11, no_wrap=True, overflow="fold")
    table.add_column("Day P&L",    justify="right", min_width=10, no_wrap=True, overflow="fold")
    table.add_column("Weight",     justify="right", min_width=7,  no_wrap=True, overflow="fold")

    for r in rows:
        weight = r["market_value"] / total_mv * 100
        table.add_row(
            r["ticker"],
            f"${r['last']:,.2f}",
            color(r["daily_pct"],  f"{r['daily_pct']:+.2f}%"),
            f"${r['market_value']:,.0f}",
            f"${r['cost_basis']:,.0f}",
            color(r["unreal_pnl"], f"${r['unreal_pnl']:+,.0f}"),
            color(r["day_pnl"],    f"${r['day_pnl']:+,.0f}"),
            f"{weight:.1f}%",
        )

    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]",
        "",
        "",
        f"[bold]${total_mv:,.0f}[/bold]",
        f"[bold]${total_cb:,.0f}[/bold]",
        f"[bold]{color(total_upnl, f'${total_upnl:+,.0f}')}[/bold]",
        f"[bold]{color(total_dpnl, f'${total_dpnl:+,.0f}')}[/bold]",
        "[bold]100.0%[/bold]",
    )

    console.print(" " * 30, end="\r")
    console.print(table)
    console.print()


if __name__ == "__main__":
    main()
