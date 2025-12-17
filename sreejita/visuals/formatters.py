from matplotlib.ticker import FuncFormatter

def millions_formatter(x, pos):
    return f"${x/1_000_000:.1f}M"

def thousands_formatter(x, pos):
    return f"${x/1_000:.0f}K"

def plain_thousands_formatter(x, pos):
    return f"{int(x/1_000)}K"
