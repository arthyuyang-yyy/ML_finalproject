"""Streamlit entry point for the meeting-memory demo."""

from src.ui.streamlit_app import main
from src.utils import load_dotenv


if __name__ == "__main__":
    load_dotenv()
    main()
