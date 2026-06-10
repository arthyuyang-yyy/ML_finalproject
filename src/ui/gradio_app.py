"""Gradio demo for the meeting-memory pipeline."""

from pathlib import Path

from src.pipeline import run_meeting_pipeline


def build_app():
    """Build the Gradio interface lazily so imports stay lightweight."""
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - depends on optional demo extra
        raise ImportError("Install demo dependencies with `pip install -r requirements-demo.txt`.") from exc

    def run(audio_path: str, meeting_id: str) -> dict:
        if not audio_path:
            return {"error": "Please upload an audio file."}
        return run_meeting_pipeline(audio_path, meeting_id or Path(audio_path).stem)

    with gr.Blocks(title="Meeting Memory Demo") as demo:
        gr.Markdown("# Meeting Memory Demo")
        audio = gr.Audio(label="Meeting audio", type="filepath")
        meeting_id = gr.Textbox(label="Meeting ID", value="meeting_001")
        run_button = gr.Button("Run pipeline")
        output = gr.JSON(label="Pipeline result")
        run_button.click(run, inputs=[audio, meeting_id], outputs=output)
    return demo


def launch() -> None:
    """Launch the Gradio app."""
    build_app().launch()
