from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from analyzer import AnalysisResult, DetectionEvent, DroneVideoAnalyzer
from report import build_pdf_report


APP_ROOT = Path(__file__).parent
OUTPUT_ROOT = APP_ROOT / "outputs"
OUTPUT_ROOT.mkdir(exist_ok=True)

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

RESCUE_ESSENTIALS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "suitcase",
    "traffic light", "fire hydrant", "stop sign",
]

PRESETS = {
    "Rescue essentials": RESCUE_ESSENTIALS,
    "People only": ["person"],
    "People and vehicles": [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"
    ],
    "People, animals and bags": [
        "person", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
        "zebra", "giraffe", "backpack", "umbrella", "handbag", "suitcase"
    ],
    "All supported objects": COCO_CLASSES,
}

st.set_page_config(
    page_title="RescueEye // Aerial Intelligence",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
    --bg:#020604;
    --panel:#07140c;
    --green:#31ff70;
    --green2:#79ff8f;
    --cyan:#59efff;
    --amber:#ffbf5f;
    --text:#d9ffe3;
    --muted:#83a98d;
    --line:rgba(103,255,139,.22);
}

@keyframes pulse {
    0%,100% { opacity:.35; box-shadow:0 0 5px var(--green); }
    50% { opacity:1; box-shadow:0 0 18px var(--green); }
}

@keyframes scan {
    0% { transform:translateY(-120%); }
    100% { transform:translateY(900%); }
}

html, body, [class*="css"] {
    font-family:"Cascadia Code","JetBrains Mono",Consolas,monospace;
}

.stApp {
    color:var(--text);
    background-color:var(--bg);
    background-image:
        linear-gradient(rgba(69,255,111,.025) 1px,transparent 1px),
        linear-gradient(90deg,rgba(69,255,111,.025) 1px,transparent 1px),
        radial-gradient(circle at 80% 4%,rgba(33,255,111,.10),transparent 28%);
    background-size:34px 34px,34px 34px,auto;
}

.stApp::after {
    content:"";
    position:fixed;
    inset:0;
    pointer-events:none;
    z-index:9999;
    background:repeating-linear-gradient(
        0deg,
        rgba(255,255,255,.012),
        rgba(255,255,255,.012) 1px,
        transparent 1px,
        transparent 4px
    );
}

[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer { visibility:hidden; }
.block-container { max-width:1380px; padding:1.1rem 2rem 4rem; }

[data-testid="stSidebar"] {
    background:linear-gradient(180deg,rgba(5,18,11,.98),rgba(2,9,5,.98));
    border-right:1px solid var(--line);
}

.command-shell {
    position:relative;
    overflow:hidden;
    border:1px solid rgba(105,255,142,.36);
    border-radius:8px;
    padding:1rem 1.15rem 1.15rem;
    margin-bottom:1.2rem;
    background:linear-gradient(135deg,rgba(8,27,16,.96),rgba(2,10,6,.96));
    box-shadow:inset 0 0 0 1px rgba(89,239,255,.045),0 0 38px rgba(44,255,111,.055);
}

.command-shell::before,
.command-shell::after {
    content:"";
    position:absolute;
    width:38px;
    height:38px;
    pointer-events:none;
}

.command-shell::before {
    top:8px;
    left:8px;
    border-top:2px solid var(--green);
    border-left:2px solid var(--green);
}

.command-shell::after {
    right:8px;
    bottom:8px;
    border-right:2px solid var(--cyan);
    border-bottom:2px solid var(--cyan);
}

.scanbar {
    position:absolute;
    left:0;
    right:0;
    height:18%;
    pointer-events:none;
    background:linear-gradient(to bottom,transparent,rgba(72,255,118,.035),transparent);
    animation:scan 7s linear infinite;
}

.topline {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:1rem;
    padding-bottom:.7rem;
    border-bottom:1px solid var(--line);
    color:var(--muted);
    font-size:.72rem;
    letter-spacing:.12em;
    text-transform:uppercase;
}

.brand-lockup { color:var(--green2); font-weight:800; letter-spacing:.18em; }
.online { display:inline-flex; align-items:center; gap:.45rem; color:var(--green2); }
.online-dot {
    width:7px;
    height:7px;
    border-radius:50%;
    background:var(--green);
    animation:pulse 1.6s ease-in-out infinite;
}

.hero-grid {
    display:grid;
    grid-template-columns:1fr;
    align-items:center;
    gap:0;
    padding:1.35rem 0 .55rem;
}

.eyebrow {
    color:var(--cyan);
    font-size:.75rem;
    letter-spacing:.18em;
    text-transform:uppercase;
    margin-bottom:.65rem;
}

.command-shell h1 {
    margin:0;
    max-width:900px;
    color:#ecfff1;
    font-size:clamp(2.35rem,5vw,4.8rem);
    line-height:.94;
    letter-spacing:-.055em;
    text-shadow:0 0 25px rgba(89,255,126,.12);
}

.command-shell h1 span { color:var(--green); }
.hero-copy {
    max-width:820px;
    margin:.9rem 0 0;
    color:#93b99e;
    font-family:"Segoe UI",sans-serif;
    font-size:.98rem;
    line-height:1.65;
}

.status-row {
    display:grid;
    grid-template-columns:repeat(4,minmax(0,1fr));
    gap:.65rem;
    margin-top:.7rem;
}

.status-cell {
    border:1px solid rgba(102,255,139,.14);
    background:rgba(0,0,0,.23);
    padding:.65rem .75rem;
    min-height:58px;
}

.status-cell small {
    display:block;
    color:var(--muted);
    font-size:.63rem;
    letter-spacing:.1em;
    text-transform:uppercase;
}

.status-cell strong {
    display:block;
    margin-top:.25rem;
    color:var(--green2);
    font-size:.82rem;
    letter-spacing:.045em;
}

.protocol-alert {
    display:flex;
    gap:.8rem;
    align-items:flex-start;
    margin:0 0 1.4rem;
    padding:.85rem 1rem;
    border:1px solid rgba(255,191,95,.30);
    border-left:3px solid var(--amber);
    border-radius:4px;
    background:rgba(255,172,54,.045);
    color:#dfc795;
    font-size:.79rem;
    line-height:1.55;
}

.protocol-alert b { color:var(--amber); white-space:nowrap; }

.section-head {
    display:flex;
    align-items:center;
    gap:.75rem;
    margin:.4rem 0 .9rem;
    padding-bottom:.55rem;
    border-bottom:1px solid rgba(102,255,139,.15);
}
.section-index { color:var(--cyan); font-size:.71rem; letter-spacing:.12em; }
.section-title {
    color:var(--green2);
    font-size:.82rem;
    font-weight:800;
    letter-spacing:.11em;
    text-transform:uppercase;
}

.matrix-panel {
    min-height:100%;
    padding:1rem;
    border:1px solid rgba(89,239,255,.17);
    border-radius:6px;
    background:linear-gradient(180deg,rgba(5,20,14,.85),rgba(3,12,8,.88));
    box-shadow:inset 3px 0 0 rgba(89,239,255,.55);
}
.matrix-title {
    margin-bottom:.8rem;
    color:var(--cyan);
    font-size:.76rem;
    font-weight:800;
    letter-spacing:.12em;
    text-transform:uppercase;
}
.matrix-item {
    display:grid;
    grid-template-columns:22px 1fr;
    gap:.55rem;
    padding:.55rem 0;
    border-bottom:1px dashed rgba(116,255,149,.10);
    color:#accbb4;
    font-size:.77rem;
    line-height:1.42;
}
.matrix-code { color:var(--green); }

.terminal-readout {
    margin-top:.9rem;
    padding:.7rem .75rem;
    border:1px solid rgba(111,255,146,.15);
    background:#010503;
    color:#7fd993;
    font-size:.68rem;
    line-height:1.65;
}
.terminal-readout .cyan { color:var(--cyan); }
.terminal-readout .amber { color:var(--amber); }

.results-banner {
    margin-top:2rem;
    padding:.85rem 1rem;
    border-top:1px solid rgba(89,239,255,.35);
    border-bottom:1px solid rgba(49,255,112,.25);
    background:linear-gradient(90deg,rgba(49,255,112,.055),transparent);
    color:var(--green2);
    font-size:.84rem;
    font-weight:800;
    letter-spacing:.13em;
    text-transform:uppercase;
}

label,
.stTextInput label,
.stSelectbox label,
.stMultiSelect label,
.stSlider label,
.stFileUploader label {
    color:#a7cbb0 !important;
    font-size:.75rem !important;
    letter-spacing:.045em;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"],
[data-testid="stFileUploaderDropzone"] {
    border-color:rgba(106,255,143,.24) !important;
    border-radius:4px !important;
    background:rgba(2,10,6,.84) !important;
    box-shadow:none !important;
}

input, textarea { color:var(--text) !important; caret-color:var(--green) !important; }
[data-testid="stFileUploaderDropzone"] { border-style:dashed !important; }

.stButton > button,
.stDownloadButton > button {
    min-height:47px;
    border:1px solid rgba(49,255,112,.55);
    border-radius:3px;
    background:linear-gradient(90deg,rgba(49,255,112,.15),rgba(49,255,112,.04));
    color:var(--green2) !important;
    font-family:"Cascadia Code",Consolas,monospace;
    font-size:.77rem;
    font-weight:800;
    letter-spacing:.08em;
    text-transform:uppercase;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color:var(--green);
    background:rgba(49,255,112,.15);
    color:#effff2 !important;
    box-shadow:0 0 22px rgba(49,255,112,.13);
}

[data-testid="stMetric"] {
    min-height:122px;
    padding:1rem;
    border:1px solid rgba(103,255,139,.18);
    border-top:2px solid var(--green);
    border-radius:4px;
    background:linear-gradient(145deg,rgba(7,24,14,.92),rgba(2,10,6,.92));
}
[data-testid="stMetricLabel"] {
    color:#86a88e;
    font-size:.68rem;
    letter-spacing:.07em;
    text-transform:uppercase;
}
[data-testid="stMetricValue"] { color:var(--green2); }

[data-testid="stDataFrame"],
[data-testid="stImage"],
[data-testid="stVideo"] {
    overflow:hidden;
    border:1px solid rgba(105,255,142,.18);
    border-radius:5px;
    background:rgba(2,10,6,.74);
}

@media (max-width:900px) {
    .block-container { padding-left:1rem; padding-right:1rem; }
    .topline { align-items:flex-start; flex-direction:column; }
    .status-row { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .command-shell h1 { font-size:2.6rem; }
}

@media (max-width:560px) {
    .status-row { grid-template-columns:1fr; }
    .command-shell h1 { font-size:2.15rem; }
}
</style>
    """,
    unsafe_allow_html=True,
)

HERO_HTML = """<div class="command-shell">
<div class="scanbar"></div>
<div class="topline">
<span class="brand-lockup">RESCUE//EYE</span>
<span>CV SEARCH TERMINAL // BUILD 2.6</span>
<span class="online"><span class="online-dot"></span>SYSTEM ONLINE</span>
</div>
<div class="hero-grid">
<div>
<div class="eyebrow">Aerial intelligence and object tracking console</div>
<h1>SEARCH<span>.</span> TRACK<span>.</span><br>VERIFY<span>.</span></h1>
<p class="hero-copy">Process drone footage, identify rescue-relevant objects, preserve evidence, and turn detections into a structured mission record.</p>
</div>
</div>
<div class="status-row">
<div class="status-cell"><small>Detection Engine</small><strong>YOLO 11</strong></div>
<div class="status-cell"><small>Tracking Protocol</small><strong>BYTETRACK</strong></div>
<div class="status-cell"><small>Visual Grid</small><strong>3 × 3 SECTOR MAP</strong></div>
<div class="status-cell"><small>Output Layer</small><strong>VIDEO + CSV + PDF</strong></div>
</div>
</div>"""

st.markdown(HERO_HTML, unsafe_allow_html=True)

st.markdown(
    """<div class="protocol-alert">
<b>PROTOCOL 07</b>
<span>Decision-support prototype. All detections require human verification. The system does not diagnose injuries and must not replace trained rescue teams.</span>
</div>""",
    unsafe_allow_html=True,
)


def console_heading(index: str, title: str) -> None:
    st.markdown(
        f"""<div class="section-head">
<span class="section-index">{index}</span>
<span class="section-title">{title}</span>
</div>""",
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_analyzer(
    model_name: str,
    confidence: float,
    frame_step: int,
    low_movement_seconds: float,
    selected_class_names: tuple[str, ...],
) -> DroneVideoAnalyzer:
    return DroneVideoAnalyzer(
        model_name=model_name,
        confidence=confidence,
        frame_step=frame_step,
        low_movement_seconds=low_movement_seconds,
        selected_class_names=selected_class_names,
    )


def restore_result(data: dict) -> AnalysisResult:
    restored = data.copy()
    restored.setdefault("selected_classes", ["person"])
    restored.setdefault("class_counts", {})

    restored_events = []
    for event in restored.get("events", []):
        event_copy = event.copy()
        event_copy.setdefault("object_class", "person")
        restored_events.append(DetectionEvent(**event_copy))

    restored["events"] = restored_events
    return AnalysisResult(**restored)


with st.sidebar:
    st.markdown(
        """<div style="margin-bottom:1rem">
<div style="color:#59efff;font-size:.68rem;letter-spacing:.15em">CONTROL NODE</div>
<div style="color:#79ff8f;font-size:1.05rem;font-weight:800;letter-spacing:.08em">ANALYSIS CONFIG</div>
</div>""",
        unsafe_allow_html=True,
    )

    detection_preset = st.selectbox(
        "TARGET PROFILE",
        options=[*PRESETS.keys(), "Custom selection"],
        index=0,
    )

    if detection_preset == "Custom selection":
        selected_classes = st.multiselect(
            "OBJECT CLASS MATRIX",
            options=COCO_CLASSES,
            default=RESCUE_ESSENTIALS,
        )
    else:
        selected_classes = PRESETS[detection_preset]

    confidence = st.slider(
        "CONFIDENCE FLOOR",
        min_value=0.15,
        max_value=0.85,
        value=0.35,
        step=0.05,
    )

    frame_step = st.select_slider(
        "FRAME SAMPLING RATE",
        options=[1, 2, 3, 4, 5],
        value=2,
    )

    low_movement_seconds = st.slider(
        "LOW-MOVEMENT WINDOW",
        min_value=1.0,
        max_value=6.0,
        value=2.0,
        step=0.5,
    )

    model_name = st.selectbox(
        "DETECTION CORE",
        options=["yolo11n.pt", "yolo11s.pt"],
        index=0,
    )

    st.markdown(
        f"""<div class="terminal-readout">
<span class="cyan">&gt; target_profile</span> = {detection_preset}<br>
<span class="cyan">&gt; classes_loaded</span> = {len(selected_classes)}<br>
<span class="cyan">&gt; confidence</span> = {confidence:.2f}<br>
<span class="cyan">&gt; model</span> = {model_name}<br>
<span class="amber">&gt; weights auto-fetch on first launch</span>
</div>""",
        unsafe_allow_html=True,
    )

left, right = st.columns([1.12, 0.88], gap="large")

with left:
    console_heading("01 //", "Mission identity")

    mission_name = st.text_input(
        "MISSION NAME",
        value=f"Search Operation {datetime.now().strftime('%d %b')}",
    )
    location = st.text_input(
        "SEARCH SECTOR",
        placeholder="Example: Northern river bank, Sector B",
    )
    operator = st.text_input("OPERATOR ID", placeholder="Your name")

    console_heading("02 //", "Source footage")

    uploaded_video = st.file_uploader(
        "LOAD MP4, MOV, AVI OR MKV",
        type=["mp4", "mov", "avi", "mkv"],
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

with right:
    console_heading("03 //", "Detection matrix")

    st.markdown(
        f"""<div class="matrix-panel">
<div class="matrix-title">ACTIVE CAPABILITIES</div>
<div class="matrix-item"><span class="matrix-code">01</span><span>Detect and track people, with optional low apparent movement alerts.</span></div>
<div class="matrix-item"><span class="matrix-code">02</span><span>Recognize vehicles, bicycles, aircraft, trains and boats.</span></div>
<div class="matrix-item"><span class="matrix-code">03</span><span>Identify common animals, backpacks, handbags, umbrellas and suitcases.</span></div>
<div class="matrix-item"><span class="matrix-code">04</span><span>Assign tracking IDs, confidence values and 3 × 3 screen sectors.</span></div>
<div class="matrix-item"><span class="matrix-code">05</span><span>Export evidence frames, event logs, annotated video and a PDF report.</span></div>
<div class="terminal-readout">
<span class="cyan">&gt; selected_targets</span> = {len(selected_classes)}<br>
<span class="cyan">&gt; mission_state</span> = AWAITING_VIDEO<br>
<span class="amber">&gt; unsupported</span> = fire, smoke, injuries, structural damage
</div>
</div>""",
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:.45rem'></div>", unsafe_allow_html=True)

analyze_clicked = st.button(
    "[ EXECUTE MISSION ANALYSIS ]",
    type="primary",
    use_container_width=True,
    disabled=uploaded_video is None or not selected_classes,
)

if analyze_clicked and uploaded_video is not None:
    suffix = Path(uploaded_video.name).suffix or ".mp4"
    mission_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mission_folder = OUTPUT_ROOT / mission_stamp
    mission_folder.mkdir(parents=True, exist_ok=True)

    progress_bar = st.progress(0, text="BOOTING DETECTION PIPELINE...")
    preview_slot = st.empty()

    def update_progress(value: float, text: str) -> None:
        progress_bar.progress(
            min(max(value, 0.0), 1.0),
            text=f"CV_STREAM // {text.upper()}",
        )

    def update_preview(frame) -> None:
        preview_slot.image(
            frame,
            caption="LIVE // COMPUTER VISION FEED",
            use_container_width=True,
        )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_file.write(uploaded_video.getbuffer())
            temporary_path = Path(temporary_file.name)

        with st.spinner("INITIALIZING MODEL // SCANNING FRAME BUFFER"):
            analyzer = load_analyzer(
                model_name,
                confidence,
                frame_step,
                low_movement_seconds,
                tuple(selected_classes),
            )

            result = analyzer.analyze(
                video_path=temporary_path,
                output_folder=mission_folder,
                mission_name=mission_name.strip() or "Unnamed Mission",
                location=location.strip(),
                operator=operator.strip(),
                progress_callback=update_progress,
                preview_callback=update_preview,
            )

            report_path = build_pdf_report(
                result,
                mission_folder / "RescueEye_mission_report.pdf",
            )

            (mission_folder / "mission_result.json").write_text(
                json.dumps(result.to_dict(), indent=2),
                encoding="utf-8",
            )

            st.session_state["rescueeye_result"] = result.to_dict()
            st.session_state["rescueeye_report"] = str(report_path)

        st.success("MISSION COMPLETE // OUTPUT PACKAGE GENERATED")

    except Exception as error:
        st.error(f"PIPELINE HALTED // {error}")
        st.exception(error)

    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)

if "rescueeye_result" in st.session_state:
    result = restore_result(st.session_state["rescueeye_result"])
    report_path = Path(st.session_state["rescueeye_report"])

    st.markdown(
        '<div class="results-banner">04 // Mission intelligence recovered</div>',
        unsafe_allow_html=True,
    )

    metrics = st.columns(4)
    metrics[0].metric("Tracked objects", result.unique_track_ids)
    metrics[1].metric("Detection frames", result.total_detection_frames)
    metrics[2].metric("Low-movement alerts", result.low_movement_alerts)
    metrics[3].metric("Processed frames", result.processed_frames)

    if result.class_counts:
        console_heading("05 //", "Object count register")
        class_summary = pd.DataFrame(
            [
                {"OBJECT CLASS": name.upper(), "ESTIMATED TRACKS": count}
                for name, count in result.class_counts.items()
            ]
        )
        st.dataframe(class_summary, use_container_width=True, hide_index=True)

    if result.events:
        console_heading("06 //", "Detection event log")
        event_table = pd.DataFrame(
            [
                {
                    "EVENT": event.event_type.upper(),
                    "OBJECT": event.object_class.upper(),
                    "TRACK ID": event.track_id,
                    "TIME / SEC": event.timestamp_seconds,
                    "SECTOR": event.zone,
                    "CONFIDENCE": event.confidence,
                    "SYSTEM NOTE": event.note,
                }
                for event in result.events
            ]
        )
        st.dataframe(event_table, use_container_width=True, hide_index=True)
    else:
        st.info("NO SELECTED TARGETS DETECTED AT CURRENT CONFIDENCE FLOOR")

    annotated_path = Path(result.annotated_video)
    if annotated_path.exists():
        console_heading("07 //", "Annotated mission feed")
        st.video(str(annotated_path))

    evidence_events = [
        event for event in result.events if Path(event.evidence_path).exists()
    ]

    if evidence_events:
        console_heading("08 //", "Captured evidence frames")
        gallery = st.columns(3)
        for index, event in enumerate(evidence_events):
            with gallery[index % 3]:
                st.image(
                    event.evidence_path,
                    caption=(
                        f"{event.event_type.upper()} // {event.object_class.upper()} // "
                        f"ID {event.track_id} // {event.timestamp_seconds:.2f}s // SECTOR {event.zone}"
                    ),
                    use_container_width=True,
                )

    console_heading("09 //", "Export mission package")
    downloads = st.columns(3)

    csv_path = Path(result.events_csv)
    if csv_path.exists():
        downloads[0].download_button(
            "[ DOWNLOAD EVENT CSV ]",
            data=csv_path.read_bytes(),
            file_name="RescueEye_mission_events.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if report_path.exists():
        downloads[1].download_button(
            "[ DOWNLOAD PDF REPORT ]",
            data=report_path.read_bytes(),
            file_name="RescueEye_mission_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    if annotated_path.exists():
        downloads[2].download_button(
            "[ DOWNLOAD ANNOTATED VIDEO ]",
            data=annotated_path.read_bytes(),
            file_name="RescueEye_annotated_mission.mp4",
            mime="video/mp4",
            use_container_width=True,
        )

    st.markdown(
        """<div class="protocol-alert" style="margin-top:1rem">
<b>HUMAN REVIEW REQUIRED</b>
<span>Track counts are estimates. IDs may change when an object disappears, is blocked, or returns to the frame.</span>
</div>""",
        unsafe_allow_html=True,
    )