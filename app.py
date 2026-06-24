"""Application entry point for the interactive meeting-memory demo."""

from src.ui.gradio_app import launch
from src.utils import load_dotenv


def main() -> None:
    """Start the Gradio demo."""
    load_dotenv()
    launch()


if __name__ == "__main__":
    main()
