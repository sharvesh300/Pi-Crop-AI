import io
import json
import streamlit as st
import requests

# Base URL of your FastAPI backend
API_BASE = "http://localhost:8000"
API_URL = f"{API_BASE}/analyze"
ANALYZE_STREAM_URL = f"{API_BASE}/analyze/stream"

# URL that the *browser* will use to load the MJPEG stream.
# If you access Streamlit from another machine, replace localhost with the Pi's IP.
STREAM_URL = f"{API_BASE}/stream"

st.title("🌱 Pi-Crop-AI Diagnostic App")

# ── Sidebar: Ollama status ──────────────────────────────────────────────────
with st.sidebar:
    st.header("🧠 Ollama Status")
    try:
        _status = requests.get(f"{API_BASE}/ollama-status", timeout=5).json()
        if _status.get("reachable"):
            st.success(f"✅ Running at `{_status['host']}`")
            st.caption("Available models:")
            for _m in _status.get("models", []):
                st.markdown(f"- `{_m}`")
        else:
            st.error(f"❌ Unreachable — {_status.get('host')}")
            st.caption(_status.get("error", ""))
    except Exception:
        st.error("❌ Backend not reachable — start `python api.py`")
    st.markdown("---")
    st.caption("To change Ollama host: edit `config/model.yaml` → `ollama_host`")

crop_name = st.text_input("Enter Crop Name (Optional)", "Unknown")

# --- Image input: Pi Camera stream, browser camera, or file upload ---
tab_pi, tab_camera, tab_upload = st.tabs(["🎥 Pi Camera", "📷 Browser Camera", "📁 Upload Image"])

with tab_pi:
    st.markdown(
        f'<img src="{STREAM_URL}" style="width:100%; border-radius:8px;">',
        unsafe_allow_html=True,
    )
    st.caption("Live stream from the Pi camera")
    col_cap, col_clr = st.columns([2, 1])
    with col_cap:
        if st.button("📸 Capture Frame", key="capture_pi"):
            try:
                resp = requests.get(f"{API_BASE}/capture", timeout=10)
                if resp.status_code == 200:
                    st.session_state.pi_frame = resp.content
                    st.session_state.pop("camera_image", None)
                    st.session_state.pop("uploaded_file", None)
                    st.success("Frame captured — scroll down to analyze.")
                else:
                    st.error(f"❌ Camera returned {resp.status_code}: {resp.json().get('detail', '')}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to the Pi-Crop-AI server.")
    with col_clr:
        if st.button("🗑️ Clear", key="clear_pi"):
            st.session_state.pop("pi_frame", None)

with tab_camera:
    camera_image = st.camera_input("Take a picture of the diseased leaf")

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a leaf image", type=["jpg", "jpeg", "png"]
    )

# Resolve image source: Pi captured frame > browser camera > file upload
if st.session_state.get("pi_frame"):
    image_source = io.BytesIO(st.session_state["pi_frame"])
elif camera_image is not None:
    image_source = camera_image
else:
    image_source = uploaded_file

# --- No leaf detected state ---
if image_source is None:
    st.warning("🍃 No leaf image detected. Please stream and capture a frame from the Pi camera, take a photo with the browser camera, or upload an image to begin diagnosis.")
    st.stop()

# Display selected image
st.image(image_source, caption="Selected Leaf Image", use_container_width=True)

if st.button("Analyze Leaf"):
    files = {"image": ("leaf.jpg", image_source.getvalue(), "image/jpeg")}
    data = {"crop_name": crop_name}

    # ── Persistent result containers ───────────────────────────────────
    status_box      = st.empty()
    detection_box   = st.empty()
    dec_stream_box  = st.empty()   # live token stream — decision
    dec_result_box  = st.empty()   # structured decision card
    plan_stream_box = st.empty()   # live token stream — plan
    plan_result_box = st.empty()   # structured plan steps

    decision_text = ""
    plan_text     = ""

    try:
        response = requests.post(
            ANALYZE_STREAM_URL,
            files=files,
            data=data,
            stream=True,
            timeout=300,
        )
        response.raise_for_status()

        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            if not raw_line.startswith(b"data: "):
                continue

            event = json.loads(raw_line[6:])
            etype = event.get("type")

            # ── Progress phase banner ───────────────────────────────────
            if etype == "phase":
                status_box.info(event["message"])

            # ── Detection results ───────────────────────────────────────
            elif etype == "detection":
                severity_class = event.get("severity_class", "")
                with detection_box.container():
                    st.markdown("---")
                    st.subheader("🔬 Diagnosis Summary")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Crop", event.get("_crop", "Unknown"))
                    col2.metric("Disease", event.get("disease", "Unknown"))
                    col3.metric("Confidence", f"{event.get('confidence', 0) * 100:.1f}%")
                    col4, col5 = st.columns(2)
                    col4.metric("Severity Level", severity_class.capitalize() if severity_class else "—")
                    col5.metric("Affected Area", f"{event.get('severity_percent', 0):.1f}%")

            # ── Live decision tokens ────────────────────────────────────
            elif etype == "token" and event.get("phase") == "decision":
                decision_text += event["text"]
                dec_stream_box.markdown(
                    f"**🤖 Generating decision...**\n\n```\n{decision_text}▌\n```"
                )

            # ── Structured decision result ──────────────────────────────
            elif etype == "decision_result":
                dec_stream_box.empty()
                is_safe       = event.get("safe", True)
                overridden    = event.get("override", False)
                needs_confirm = event.get("requires_confirmation", False)
                with dec_result_box.container():
                    st.markdown("---")
                    st.subheader("🤖 Agent Decision")
                    if is_safe:
                        st.success(f"**Decision:** {event.get('decision', '')}")
                    else:
                        st.error(f"**Decision:** {event.get('decision', '')}")
                    st.info(f"**Reasoning:** {event.get('reason', '—')}")
                    flag_cols = st.columns(3)
                    flag_cols[0].metric("Safe", "✅ Yes" if is_safe else "⚠️ No")
                    flag_cols[1].metric("Safety Override", "Yes" if overridden else "No")
                    flag_cols[2].metric("Needs Confirmation", "Yes" if needs_confirm else "No")
                    if overridden:
                        st.warning("⚠️ Safety validator overrode the LLM decision.")
                    if needs_confirm:
                        st.warning("⚠️ This action requires human confirmation before proceeding.")

            # ── Live plan tokens ────────────────────────────────────────
            elif etype == "token" and event.get("phase") == "plan":
                plan_text += event["text"]
                plan_stream_box.markdown(
                    f"**💊 Generating treatment plan...**\n\n```\n{plan_text}▌\n```"
                )

            # ── Structured plan result ──────────────────────────────────
            elif etype == "plan_result":
                plan_stream_box.empty()
                plan = event.get("plan", [])
                if plan:
                    with plan_result_box.container():
                        st.markdown("---")
                        st.subheader("💊 Treatment Plan")
                        for step in plan:
                            with st.expander(
                                f"Step {step.get('step', '?')} — {step.get('action', '')}"
                            ):
                                st.write(step.get("details", ""))

            # ── All done ────────────────────────────────────────────────
            elif etype == "done":
                status_box.success("✅ Analysis complete!")
                break

            # ── Error from pipeline ─────────────────────────────────────
            elif etype == "error":
                status_box.error(f"❌ {event.get('message', 'Unknown error')}")
                break

    except requests.exceptions.ReadTimeout:
        st.error(
            "⏱️ Analysis timed out after 5 minutes. "
            "Check that Ollama is running (`ollama serve`) "
            "and the model is loaded (`ollama run qwen2.5:1.5b`)."
        )
    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Cannot connect to the Pi-Crop-AI server. "
            "Make sure the backend is running on port 8000."
        )
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Server error: {e}")