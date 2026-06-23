"""Streamlit visual tester for the meeting-memory pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from src.ui.gradio_app import (
    EMPTY_STATE,
    MEMORY_HEADERS,
    TIMELINE_HEADERS,
    answer_demo_question,
    build_memory_rows,
    build_timeline_rows,
    candidate_audio_path,
    candidate_detail,
    prepare_demo_data,
    run_demo_pipeline,
)
from src.llm.gemma_client import create_gemma_client


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - exercised by launching the app
        raise ImportError("Install dependencies with `pip install -r requirements.txt`.") from exc

    st.set_page_config(page_title="Meeting Memory Tester", layout="wide")
    st.title("Meeting Memory Tester")

    if "meeting_state" not in st.session_state:
        st.session_state.meeting_state = dict(EMPTY_STATE)
    if "view" not in st.session_state:
        st.session_state.view = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    with st.sidebar:
        st.header("Run")
        uploaded = st.file_uploader("Meeting audio", type=["wav", "mp3", "m4a", "flac", "ogg"])
        meeting_id = st.text_input("Meeting ID", value="meeting_001")
        output_root = st.text_input("Output folder", value="outputs/ui")
        existing_output_dir = st.text_input(
            "Load existing output dir",
            value="outputs/experiments/fullflow_tuning/run_018_short_overlap_context/outputs/R8001_M8004_1110_1350_run_018_short_overlap_context",
        )
        load_existing_clicked = st.button("Load Existing Result", use_container_width=True)
        language = st.selectbox("Language", ["zh", "en", "und"], index=0)
        asr_backend = st.selectbox(
            "ASR backend",
            ["auto", "funasr", "faster-whisper", "whisperx", "whisper", "mock"],
            index=1,
        )
        gemma_backend = st.selectbox(
            "Gemma backend",
            ["none", "deepseek", "ollama", "openai", "transformers"],
            index=0,
        )
        gemma_model = st.text_input("Gemma model", value="gemma3:4b")
        st.subheader("Tuning")
        overlap_threshold = st.slider("Overlap threshold", 0.0, 1.0, 0.5, 0.05)
        suspected_overlap_threshold = st.slider("Suspected overlap threshold", 0.0, 1.0, 0.3, 0.05)
        high_overlap_min_segment_s = st.number_input("High-overlap min segment seconds", min_value=0.0, value=2.0)
        high_overlap_decode_context_s = st.number_input("Short-overlap decode context seconds", min_value=0.0, value=2.0)
        asr_context_padding_s = st.number_input("Low-overlap ASR context padding seconds", min_value=0.0, value=0.2)
        speech_separation_backend = st.selectbox("Speech separation", ["none", "nmf", "sepformer"], index=0)
        run_clicked = st.button("Run Pipeline", type="primary", use_container_width=True)

    if load_existing_clicked:
        try:
            result = _load_existing_result(Path(existing_output_dir))
            view = prepare_demo_data(result)
            st.session_state.last_result = result
            st.session_state.view = view
            st.session_state.meeting_state = view["state"]
            st.success("已加载已有输出。")
        except Exception as exc:
            st.exception(exc)

    if run_clicked:
        if uploaded is None:
            st.error("请先上传会议音频。")
        else:
            with st.status("Running pipeline...", expanded=True) as status:
                audio_path = _save_upload(uploaded)
                st.write(f"Saved upload: {audio_path}")
                try:
                    result = run_demo_pipeline(
                        str(audio_path),
                        meeting_id,
                        asr_backend=asr_backend,
                        gemma_backend=gemma_backend,
                        gemma_model=gemma_model,
                        output_root=output_root,
                        language=language,
                        overlap_threshold=overlap_threshold,
                        suspected_overlap_threshold=suspected_overlap_threshold,
                        high_overlap_min_segment_s=high_overlap_min_segment_s,
                        high_overlap_decode_context_s=high_overlap_decode_context_s,
                        asr_context_padding_s=asr_context_padding_s,
                        speech_separation_backend=speech_separation_backend,
                    )
                    view = prepare_demo_data(result)
                    st.session_state.last_result = result
                    st.session_state.view = view
                    st.session_state.meeting_state = view["state"]
                    status.update(label="Pipeline complete", state="complete", expanded=False)
                except Exception as exc:
                    status.update(label="Pipeline failed", state="error", expanded=True)
                    st.exception(exc)

    view = st.session_state.view
    result = st.session_state.last_result
    if not view or not result:
        st.info("可以上传音频运行 Pipeline，或在左侧加载已有输出目录来查看结果。")
        return

    _render_summary(result)
    timeline_tab, candidates_tab, memory_tab, qa_tab, files_tab = st.tabs(
        ["Timeline", "High-overlap Candidates", "Meeting Memory", "QA", "Artifacts"]
    )

    with timeline_tab:
        st.dataframe(
            pd.DataFrame(view["timeline"], columns=TIMELINE_HEADERS),
            use_container_width=True,
            hide_index=True,
        )

    with candidates_tab:
        choices = view["candidate_choices"]
        if not choices:
            st.info("No high-overlap candidate segments.")
        else:
            labels = [label for label, _value in choices]
            values = {label: value for label, value in choices}
            selected_label = st.selectbox("High-overlap segment", labels)
            segment_id = values[selected_label]
            detail = candidate_detail(segment_id, st.session_state.meeting_state)
            st.json(detail)
            audio_path = candidate_audio_path(segment_id, st.session_state.meeting_state)
            if audio_path and Path(audio_path).exists():
                st.audio(audio_path)

    with memory_tab:
        st.dataframe(
            pd.DataFrame(build_memory_rows(st.session_state.meeting_state.get("episodic_memory", [])), columns=MEMORY_HEADERS),
            use_container_width=True,
            hide_index=True,
        )

    with qa_tab:
        question = st.text_input("Question", placeholder="例如：谁负责测试 WhisperX？")
        if st.button("Ask", type="primary"):
            client = create_gemma_client(gemma_backend, model=gemma_model)
            answer, retrieval_rows, payload = answer_demo_question(question, st.session_state.meeting_state, client=client)
            st.markdown(answer)
            st.dataframe(
                pd.DataFrame(retrieval_rows, columns=["Episode", "Type", "Score", "Evidence"]),
                use_container_width=True,
                hide_index=True,
            )
            st.json(payload)

    with files_tab:
        st.write(result.get("output_dir", ""))
        st.json(result.get("artifacts", {}))


def _save_upload(uploaded: Any) -> Path:
    suffix = Path(uploaded.name).suffix or ".wav"
    target = Path(tempfile.gettempdir()) / "meeting_memory_uploads" / f"{Path(uploaded.name).stem}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(uploaded.getbuffer())
    return target


def _load_existing_result(output_dir: Path) -> dict[str, Any]:
    if not output_dir.exists():
        raise FileNotFoundError(output_dir)
    evidence = _read_json(output_dir / "evidence_segments.json", default=[])
    memory = _read_json(output_dir / "episodic_memory.json", default=[])
    events = _read_json(output_dir / "meeting_events.json", default={"events": []})
    meeting_id = output_dir.name
    high = [item for item in evidence if item.get("processing_path") == "high_overlap_candidate"]
    return {
        "meeting_id": meeting_id,
        "output_dir": str(output_dir),
        "evidence_segments": evidence,
        "episodic_memory": memory,
        "meeting_events": events,
        "num_evidence_segments": len(evidence),
        "num_high_overlap_segments": len(high),
        "artifacts": {
            "evidence_segments": str(output_dir / "evidence_segments.json"),
            "high_overlap_candidates": str(output_dir / "high_overlap_candidates.json"),
            "meeting_events": str(output_dir / "meeting_events.json"),
            "episodic_memory": str(output_dir / "episodic_memory.json"),
            "long_term_episodic_memory": str(output_dir.parent.parent / "memory" / "episodic_memory.json"),
        },
    }


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _render_summary(result: dict[str, Any]) -> None:
    cols = st_columns()
    cols[0].metric("Evidence", int(result.get("num_evidence_segments", 0)))
    cols[1].metric("High-overlap", int(result.get("num_high_overlap_segments", 0)))
    cols[2].metric("Memory", len(result.get("episodic_memory", [])))
    cols[3].metric("Output", Path(str(result.get("output_dir", ""))).name)


def st_columns():
    import streamlit as st

    return st.columns(4)


__all__ = ["main"]
