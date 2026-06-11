"""Five-area Gradio demo for the meeting-memory pipeline."""

import re
from pathlib import Path
from typing import Any

from src.memory.retriever import retrieve_episodes
from src.llm.gemma_client import GemmaClient, create_gemma_client
from src.pipeline import run_meeting_pipeline
from src.pipeline.config import PipelineConfig
from src.qa.answerer import answer_question

TIMELINE_HEADERS = ["Time", "Speaker", "Path", "Overlap", "Text"]
MEMORY_HEADERS = ["Type", "Content", "Evidence", "Confidence"]
EMPTY_STATE: dict[str, Any] = {
    "meeting_id": "",
    "evidence_segments": [],
    "episodic_memory": [],
    "long_term_memory_path": "",
}


def prepare_demo_data(result: dict[str, Any]) -> dict[str, Any]:
    """Convert pipeline output into stable UI state and display rows."""
    evidence_segments = result.get("evidence_segments")
    if evidence_segments is None:
        evidence_segments = _read_artifact(result, "evidence_segments", default=[])
    episodic_memory = result.get("episodic_memory", [])
    high_segments = [
        segment
        for segment in evidence_segments
        if segment.get("processing_path") == "high_overlap_candidate"
    ]
    choices = [
        (f"{segment['segment_id']} ({_format_range(segment['start_time'], segment['end_time'])})", segment["segment_id"])
        for segment in high_segments
    ]
    state = {
        "meeting_id": str(result.get("meeting_id", "")),
        "evidence_segments": evidence_segments,
        "episodic_memory": episodic_memory,
        "long_term_memory_path": str(result.get("artifacts", {}).get("long_term_episodic_memory", "")),
    }
    return {
        "state": state,
        "timeline": build_timeline_rows(evidence_segments),
        "candidate_choices": choices,
        "selected_candidate": choices[0][1] if choices else None,
        "candidate_detail": candidate_detail(choices[0][1], state) if choices else {},
        "memory": build_memory_rows(episodic_memory),
    }


def build_timeline_rows(evidence_segments: list[dict[str, Any]]) -> list[list[Any]]:
    """Build compact timeline rows from canonical evidence segments."""
    rows: list[list[Any]] = []
    for segment in evidence_segments:
        processing_path = str(segment.get("processing_path", ""))
        text = str(segment.get("text", "")).strip()
        if processing_path == "high_overlap_candidate" and not text:
            text = "candidates available"
        rows.append([
            _format_range(segment.get("start_time", 0.0), segment.get("end_time", 0.0)),
            str(segment.get("speaker", "UNKNOWN")),
            _display_path(processing_path),
            round(float(segment.get("overlap_score", 0.0)), 3),
            text,
        ])
    return rows


def build_memory_rows(episodes: list[dict[str, Any]]) -> list[list[str]]:
    """Build the meeting-memory table from canonical episodes."""
    return [
        [
            str(episode.get("event_type", "")),
            str(episode.get("content", "")),
            ", ".join(str(value) for value in episode.get("evidence_ids", [])),
            str(episode.get("confidence", "")),
        ]
        for episode in episodes
    ]


def candidate_detail(segment_id: str | None, state: dict[str, Any] | None) -> dict[str, Any]:
    """Return one high-overlap segment in the presentation schema."""
    if not segment_id or not state:
        return {}
    for segment in state.get("evidence_segments", []):
        if segment.get("segment_id") != segment_id:
            continue
        if segment.get("processing_path") != "high_overlap_candidate":
            return {}
        candidates = [
            f"{candidate.get('speaker', 'UNKNOWN')}: {candidate.get('text', '')}".strip()
            for candidate in segment.get("candidates", [])
        ]
        return {
            "segment_id": str(segment["segment_id"]),
            "overlap_score": float(segment.get("overlap_score", 0.0)),
            "candidates": candidates,
            "uncertainty_note": str(segment.get("uncertainty_note", "")),
            "audio_clip_path": str(segment.get("audio_clip_path", "")),
        }
    return {}


def answer_demo_question(
    question: str,
    state: dict[str, Any] | None,
    top_k: int = 5,
    client: GemmaClient | None = None,
) -> tuple[str, list[list[Any]], dict[str, Any]]:
    """Retrieve across persisted episodic memory and return a validated QA answer."""
    if not isinstance(question, str) or not question.strip():
        return "请输入问题。", [], {}
    current_state = state or {}
    memory_path = str(current_state.get("long_term_memory_path", ""))
    episodes = None if memory_path and Path(memory_path).exists() else list(current_state.get("episodic_memory", []))
    retrieved = retrieve_episodes(question, episodes=episodes, path=memory_path, top_k=top_k)
    result = answer_question(question, retrieved, client=client)
    retrieval_rows = [
        [
            episode.get("episode_id", ""),
            episode.get("event_type", ""),
            round(float(episode.get("retrieval", {}).get("final_score", 0.0)), 4),
            ", ".join(str(value) for value in episode.get("evidence_ids", [])),
        ]
        for episode in retrieved
    ]
    return result["answer"], retrieval_rows, result


def run_demo_pipeline(
    audio_path: str | None,
    meeting_id: str,
    asr_backend: str = "auto",
    gemma_backend: str = "none",
    gemma_model: str = "gemma3:4b",
) -> dict[str, Any]:
    """Validate UI input and execute the shared application pipeline."""
    if not audio_path:
        raise ValueError("请先上传会议音频。")
    normalized_id = _meeting_id(meeting_id or Path(audio_path).stem)
    config = PipelineConfig(
        low_overlap_asr_model=asr_backend,
        gemma_backend=gemma_backend,
        gemma_model=gemma_model,
    )
    return run_meeting_pipeline(audio_path, normalized_id, config=config)


def build_app():
    """Build the five-area Gradio interface."""
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - optional demo dependency
        raise ImportError("Install demo dependencies with `pip install -r requirements-demo.txt`.") from exc

    def run(audio_path: str | None, meeting_id: str, asr_backend: str, gemma_backend: str, gemma_model: str):
        try:
            result = run_demo_pipeline(audio_path, meeting_id, asr_backend, gemma_backend, gemma_model)
            view = prepare_demo_data(result)
            selected = view["selected_candidate"]
            status = (
                f"处理完成：{result['num_evidence_segments']} 个证据片段，"
                f"其中 {result['num_high_overlap_segments']} 个高重叠片段；"
                f"生成 {len(result['episodic_memory'])} 条会议记忆。"
            )
            return (
                view["state"],
                status,
                view["timeline"],
                gr.Dropdown(choices=view["candidate_choices"], value=selected),
                view["candidate_detail"],
                view["memory"],
                "",
                [],
                {},
            )
        except Exception as exc:  # UI boundary: surface errors without losing the page
            return (
                dict(EMPTY_STATE),
                f"处理失败：{exc}",
                [],
                gr.Dropdown(choices=[], value=None),
                {},
                [],
                "",
                [],
                {},
            )

    def ask(question: str, state: dict[str, Any], gemma_backend: str, gemma_model: str):
        client = create_gemma_client(gemma_backend, model=gemma_model)
        return answer_demo_question(question, state, client=client)

    with gr.Blocks(title="Overlap-Aware Meeting Memory", theme=gr.themes.Soft()) as demo:
        state = gr.State(dict(EMPTY_STATE))
        gr.Markdown(
            "# Overlap-Aware Meeting Memory\n"
            "上传会议音频，查看重叠感知时间线、候选解释、结构化记忆和可追溯问答。"
        )

        with gr.Group():
            gr.Markdown("## 1. Upload Meeting Audio")
            with gr.Row():
                audio = gr.Audio(label="Meeting audio", type="filepath")
                with gr.Column():
                    meeting_id = gr.Textbox(label="Meeting ID", value="meeting_001")
                    asr_backend = gr.Dropdown(
                        label="ASR backend",
                        choices=["auto", "whisperx", "faster-whisper", "whisper", "funasr", "mock"],
                        value="auto",
                    )
                    gemma_backend = gr.Dropdown(
                        label="Gemma backend",
                        choices=["none", "ollama"],
                        value="none",
                    )
                    gemma_model = gr.Textbox(label="Gemma model", value="gemma3:4b")
                    run_button = gr.Button("Run Pipeline", variant="primary")
                    status = gr.Markdown("等待上传音频。")

        with gr.Group():
            gr.Markdown("## 2. Timeline")
            timeline = gr.Dataframe(
                headers=TIMELINE_HEADERS,
                datatype=["str", "str", "str", "number", "str"],
                value=[],
                interactive=False,
                wrap=True,
            )

        with gr.Group():
            gr.Markdown("## 3. High-overlap Candidates")
            candidate_selector = gr.Dropdown(
                label="High-overlap segment",
                choices=[],
                value=None,
            )
            candidate_json = gr.JSON(label="Candidate detail", value={})

        with gr.Group():
            gr.Markdown("## 4. Meeting Memory")
            memory_table = gr.Dataframe(
                headers=MEMORY_HEADERS,
                datatype=["str", "str", "str", "str"],
                value=[],
                interactive=False,
                wrap=True,
            )

        with gr.Group():
            gr.Markdown("## 5. QA")
            with gr.Row():
                question = gr.Textbox(
                    label="Question",
                    placeholder="例如：为什么说 Gemma 是后处理模块？",
                    scale=4,
                )
                ask_button = gr.Button("Ask", variant="primary", scale=1)
            answer = gr.Markdown("运行 Pipeline 后可以提问。")
            with gr.Accordion("Retrieved Episodes", open=False):
                retrieval_table = gr.Dataframe(
                    headers=["Episode", "Type", "Score", "Evidence"],
                    datatype=["str", "str", "number", "str"],
                    value=[],
                    interactive=False,
                )
                answer_json = gr.JSON(label="Validated QA result", value={})

        run_button.click(
            run,
            inputs=[audio, meeting_id, asr_backend, gemma_backend, gemma_model],
            outputs=[
                state,
                status,
                timeline,
                candidate_selector,
                candidate_json,
                memory_table,
                answer,
                retrieval_table,
                answer_json,
            ],
        )
        candidate_selector.change(
            candidate_detail,
            inputs=[candidate_selector, state],
            outputs=candidate_json,
        )
        ask_button.click(
            ask,
            inputs=[question, state, gemma_backend, gemma_model],
            outputs=[answer, retrieval_table, answer_json],
        )
        question.submit(
            ask,
            inputs=[question, state, gemma_backend, gemma_model],
            outputs=[answer, retrieval_table, answer_json],
        )

    return demo


def launch() -> None:
    """Launch the Gradio app."""
    build_app().launch()


def _read_artifact(result: dict[str, Any], name: str, default: Any) -> Any:
    path = result.get("artifacts", {}).get(name)
    if not path or not Path(path).exists():
        return default
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _format_range(start: Any, end: Any) -> str:
    return f"{_clock(float(start))}–{_clock(float(end))}"


def _clock(seconds: float) -> str:
    minutes, remainder = divmod(max(seconds, 0.0), 60.0)
    if abs(remainder - round(remainder)) < 1e-6:
        return f"{int(minutes):02d}:{int(round(remainder)):02d}"
    return f"{int(minutes):02d}:{remainder:04.1f}"


def _display_path(processing_path: str) -> str:
    return {
        "low_overlap_cluster": "low_overlap",
        "high_overlap_candidate": "high_overlap",
    }.get(processing_path, processing_path)


def _meeting_id(value: str) -> str:
    normalized = value.strip()
    if not normalized or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", normalized):
        raise ValueError("Meeting ID 只能包含字母、数字、点、下划线和连字符。")
    return normalized


__all__ = [
    "answer_demo_question",
    "build_app",
    "build_memory_rows",
    "build_timeline_rows",
    "candidate_detail",
    "launch",
    "prepare_demo_data",
    "run_demo_pipeline",
]
