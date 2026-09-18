"""GUI entry point with Qt loaded only when opening the graphical interface."""


def main() -> None:
    from PASS.gui.app import main as launch

    launch()


__all__ = ["main"]
