# api.py
import asyncio
import os
import shutil
import threading
import time
from typing import Optional
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from main import run_pipeline, stream_pipeline  # Import your existing pipeline

app = FastAPI(title="Pi-Crop-AI API")


# ---------------------------------------------------------------------------
# Pi Camera — shared background reader
# ---------------------------------------------------------------------------

class _PiCamera:
    """Opens the first available Pi/USB camera and captures frames in a
    background thread so /stream and /capture share one handle without
    blocking the async event loop.

    Root-cause fixes:
      1. Resolution set to match main.py (1280x720).
      2. 2-second warm-up sleep after open mirrors main.py capture_from_webcam.
      3. _read_loop throttled to ~30 fps so the driver is not hammered.
      4. Optional[bytes] replaces bytes | None (Python 3.9 compatible).
      5. Camera init runs in a thread so startup does not block the event loop.
    """

    _FRAME_INTERVAL = 1.0 / 30  # ~30 fps

    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[bytes] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False

    def start(self):
        """Probe camera devices exactly like main.py's capture_from_webcam."""
        print("Pi Camera: searching for camera device...")
        for idx in range(5):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"Pi Camera: found at /dev/video{idx}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                # Warm-up: let auto-exposure settle (mirrors main.py)
                time.sleep(2)
                self._cap = cap
                break
            cap.release()

        if self._cap is None:
            print("Pi Camera: no camera detected — /stream and /capture will return 503")
            return

        self._running = True
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _read_loop(self):
        """Read frames at ~30 fps. Throttle prevents camera driver thrashing."""
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                with self._lock:
                    self._frame = buf.tobytes()
            # Throttle regardless of success so we don't spin-wait
            time.sleep(self._FRAME_INTERVAL)

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._frame

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None


_camera = _PiCamera()

# Allow requests from your app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def _startup():
    # Run blocking camera init (including 2 s warm-up) in a thread pool
    # so the async event loop is never stalled during startup.
    await asyncio.to_thread(_camera.start)


@app.on_event("shutdown")
async def _shutdown():
    _camera.stop()


async def _mjpeg_generator():
    """Async generator — asyncio.sleep yields control every frame so the
    event loop is never blocked (fixes the original sync time.sleep bug)."""
    while True:
        frame = _camera.get_jpeg()
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        await asyncio.sleep(1.0 / 30)  # ~33 ms — keeps event loop responsive


@app.get("/stream")
async def pi_stream():
    """Live MJPEG stream from the Pi camera."""
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/capture")
async def pi_capture():
    """Return a single JPEG frame from the Pi camera."""
    frame = _camera.get_jpeg()
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not ready or not connected")
    return Response(content=frame, media_type="image/jpeg")


@app.get("/ollama-status")
async def ollama_status():
    """Report where Ollama is running and whether it is reachable."""
    import requests as _http
    from crop_agent.llm.model_backends import _OLLAMA_HOST
    try:
        resp = _http.get(f"{_OLLAMA_HOST}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        return {"host": _OLLAMA_HOST, "reachable": True, "models": models}
    except Exception as exc:
        return {"host": _OLLAMA_HOST, "reachable": False, "error": str(exc)}


def _with_cleanup(gen, path: str):
    """Wrap a sync generator to delete temp_file when exhausted or on error."""
    try:
        yield from gen
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.post("/analyze/stream")
async def analyze_stream(
    image: UploadFile = File(...),
    crop_name: str = Form("Unknown"),
):
    """Stream the full analysis pipeline as Server-Sent Events.

    Events are newline-delimited JSON prefixed with 'data: '.
    See stream_pipeline() in main.py for the full event schema.
    """
    temp_file = f"temp_stream_{image.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    return StreamingResponse(
        _with_cleanup(stream_pipeline(temp_file, crop_name), temp_file),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...), 
    crop_name: str = Form("Unknown")
):
    # 1. Save the uploaded image to a temporary file
    temp_file = f"temp_{image.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
        
    try:
        # run_pipeline makes two blocking Ollama subprocess calls; run it in
        # a thread pool so the async event loop (and camera stream) stay free.
        result = await asyncio.to_thread(run_pipeline, temp_file, crop_name)

        # 3. Return the results as JSON
        return {"status": "success", "data": result}
    except RuntimeError as e:
        # Includes Ollama-not-running and inference timeout messages
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)