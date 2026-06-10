"""Application entry point for the interactive meeting-memory demo."""

from src.ui.gradio_app import launch


def main() -> None:
    """Start the Gradio demo."""
    launch()


if __name__ == "__main__":
    main()
