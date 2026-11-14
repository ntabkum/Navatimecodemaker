import datetime
import os
import socket
import threading
import time
import webbrowser
from flask import Flask, jsonify, render_template_string, request
import numpy as np
import ntplib
import requests
import sounddevice as sd

app = Flask(__name__)

# Global Application State
state = {
    "running": False,
    "time_source": "pc",  # 'pc', 'ntp', 'server'
    "ntp_server": "pool.ntp.org",
    "time_server_url": "",
    "fps": 30.0,
    "drop_frame": False,  # only used at 29.97 FPS
    "manual_start": "00:00:00:00",  # used when time_source == 'manual'
    "audio_device": "",  # Empty string means Default PC Internal Audio
    "sample_rate": 48000,
    "offset_seconds": 0.0,
    "current_timecode": "00:00:00:00",
}

audio_thread = None

# Biphase level is kept between frames so there is always a transition
# at every frame boundary (required for a valid LTC signal).
_ltc_level = 1.0
AMPLITUDE = 0.5


def real_fps(fps):
    """Maps the UI frame rate to the exact rate (23.976 -> 24000/1001 etc.)."""
    if abs(fps - 23.976) < 0.01:
        return 24000.0 / 1001.0
    if abs(fps - 29.97) < 0.01:
        return 30000.0 / 1001.0
    return float(fps)


def frame_to_timecode(n, nominal, drop):
    """Converts an absolute frame count into h, m, s, f labels.

    drop=True applies SMPTE 29.97 drop-frame counting (frame numbers 0 and 1
    are skipped at the start of every minute except every tenth minute).
    """
    if drop:
        frames_per_day = 2589408
        n %= frames_per_day
        d, mm = divmod(n, 17982)  # 17982 frames per 10 minutes
        n += 18 * d + (2 * ((mm - 2) // 1798) if mm >= 2 else 0)
    else:
        n %= nominal * 86400
    f = n % nominal
    s = (n // nominal) % 60
    m = (n // (nominal * 60)) % 60
    h = (n // (nominal * 3600)) % 24
    return h, m, s, f


def timecode_to_frame(text, nominal, drop):
    """Parses 'HH:MM:SS:FF' (':' or ';' separators) into an absolute frame count.

    Raises ValueError for malformed or out-of-range values, including the
    frame numbers that do not exist in drop-frame counting.
    """
    parts = text.replace(";", ":").replace(".", ":").strip().split(":")
    if len(parts) != 4:
        raise ValueError("Start time must look like HH:MM:SS:FF")
    h, m, s, f = (int(p) for p in parts)
    if not (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60 and 0 <= f < nominal):
        raise ValueError(f"Start time out of range for {nominal} frames/sec")
    if drop:
        if s == 0 and f < 2 and m % 10 != 0:
            raise ValueError(
                "Frames 00 and 01 do not exist at the start of this minute in drop-frame"
            )
        total_minutes = 60 * h + m
        return (h * 3600 + m * 60 + s) * 30 + f - 2 * (total_minutes - total_minutes // 10)
    return (h * 3600 + m * 60 + s) * nominal + f


def encode_smpte_frame(h, m, s, f, fps=30.0, sample_rate=48000,
                       drop_frame=False, samples_per_frame=None):
    """Encodes h:m:s:f into one 80-bit SMPTE LTC biphase-mark audio frame."""
    global _ltc_level
    nominal = int(round(fps))  # 23.976 -> 24, 29.97 -> 30
    if samples_per_frame is None:
        samples_per_frame = int(round(sample_rate / real_fps(fps)))

    bits = [0] * 80

    def put(start, count, value):
        for i in range(count):
            bits[start + i] = (value >> i) & 1

    # Correct LTC bit layout (units + tens fields, user bits left at 0)
    put(0, 4, f % 10)
    put(8, 2, f // 10)
    bits[10] = 1 if drop_frame else 0  # drop-frame flag
    put(16, 4, s % 10)
    put(24, 3, s // 10)
    put(32, 4, m % 10)
    put(40, 3, m // 10)
    put(48, 4, h % 10)
    put(56, 2, h // 10)

    # Sync word 0011 1111 1111 1101 (bits 64..79)
    bits[64:80] = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]

    # Polarity correction bit: keep the number of zero bits even
    parity_pos = 59 if nominal == 25 else 27
    if bits.count(0) % 2 == 1:
        bits[parity_pos] = 1

    # Biphase mark: transition at start of every bit, extra one mid-bit for 1s
    edges = (np.arange(81) * samples_per_frame) // 80
    out = np.empty(samples_per_frame, dtype=np.float32)
    level = _ltc_level
    for i, bit in enumerate(bits):
        start, end = int(edges[i]), int(edges[i + 1])
        n = end - start
        level = -level  # transition at bit boundary
        seg = np.full(n, level, dtype=np.float32)
        if bit == 1:
            seg[n // 2:] = -level
            level = -level
        out[start:end] = seg
    _ltc_level = level
    return out * AMPLITUDE


def audio_generator_worker():
    try:
        sample_rate = state["sample_rate"]
        fps = state["fps"]
        rfps = real_fps(fps)
        nominal = int(round(fps))
        drop = bool(state["drop_frame"]) and nominal == 30 and rfps != 30.0
        sample_acc = 0.0  # carries fractional samples (e.g. 1601.6/frame)

        # Manual mode: start counting from the user's start time
        manual_start_sod = 0.0
        manual_t0 = time.time()
        if state["time_source"] == "manual":
            start_frame = timecode_to_frame(state["manual_start"], nominal, drop)
            manual_start_sod = start_frame / rfps

        # Empty/'default' -> None so sounddevice uses the system default output
        device_val = state["audio_device"]
        if device_val in ("", "default", None):
            device = None
        else:
            device = int(device_val)

        stream = sd.OutputStream(
            samplerate=sample_rate, channels=1, dtype="float32", device=device
        )
        stream.start()

        ntp_client = ntplib.NTPClient()
        time_offset_cache = 0.0
        last_sync = 0.0

        while state["running"]:
            now = time.time()

            # Time Source Management
            if state["time_source"] == "ntp":
                if now - last_sync > 30:
                    try:
                        response = ntp_client.request(
                            state["ntp_server"], version=3, timeout=1.5
                        )
                        time_offset_cache = response.tx_time - time.time()
                        last_sync = now
                    except Exception:
                        pass
                current_time = time.time() + time_offset_cache
            elif state["time_source"] == "server" and state["time_server_url"]:
                if now - last_sync > 10:
                    try:
                        resp = requests.get(state["time_server_url"], timeout=1.0)
                        data = resp.json()
                        server_time = float(
                            data.get("time", data.get("timestamp", time.time()))
                        )
                        time_offset_cache = server_time - time.time()
                        last_sync = now
                    except Exception:
                        pass
                current_time = time.time() + time_offset_cache
            else:
                current_time = time.time()

            current_time += state["offset_seconds"]
            # Compensate for audio output buffering so the frame sounds on time
            current_time += stream.latency

            if state["time_source"] == "manual":
                seconds_of_day = (
                    manual_start_sod
                    + (time.time() - manual_t0)
                    + state["offset_seconds"]
                    + stream.latency
                ) % 86400.0
            else:
                dt = datetime.datetime.fromtimestamp(current_time)
                seconds_of_day = (
                    dt.hour * 3600 + dt.minute * 60 + dt.second
                    + dt.microsecond / 1_000_000.0
                )
            frame_no = int(seconds_of_day * rfps)
            h, m, s, f = frame_to_timecode(frame_no, nominal, drop)

            sep = ";" if drop else ":"
            state["current_timecode"] = f"{h:02d}:{m:02d}:{s:02d}{sep}{f:02d}"

            # Whole samples for this frame, carrying the remainder forward
            sample_acc += sample_rate / rfps
            spf = int(sample_acc)
            sample_acc -= spf

            # write() blocks until the device consumes audio, which paces the loop
            frame_audio = encode_smpte_frame(
                h, m, s, f, fps=fps, sample_rate=sample_rate,
                drop_frame=drop, samples_per_frame=spf,
            )
            stream.write(frame_audio)

        stream.stop()
        stream.close()
    except Exception as e:
        print(f"Audio Stream Error: {e}")
        state["running"] = False


# HTML Frontend Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Nava Timecode Maker</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 40px; display: flex; justify-content: center; }
        .container { width: 100%; max-width: 650px; background: #1e293b; padding: 30px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
        h1 { font-size: 1.5rem; margin-bottom: 20px; color: #38bdf8; text-align: center; }
        .tc-display { background: #020617; border: 2px solid #334155; font-family: monospace; font-size: 3.5rem; text-align: center; padding: 20px; border-radius: 8px; color: #4ade80; letter-spacing: 2px; margin-bottom: 25px; box-shadow: inset 0 2px 6px rgba(0,0,0,0.8); }
        .status-badge { display: inline-block; padding: 4px 10px; border-radius: 4px; font-size: 0.85rem; font-weight: bold; text-transform: uppercase; margin-bottom: 20px; }
        .stopped { background: #ef4444; color: white; }
        .running { background: #22c55e; color: white; }
        .form-group { margin-bottom: 15px; }
        label { display: block; font-size: 0.9rem; margin-bottom: 5px; color: #cbd5e1; }
        select, input { width: 100%; padding: 10px; background: #0f172a; border: 1px solid #475569; color: white; border-radius: 6px; box-sizing: border-box; }
        .actions { display: flex; gap: 10px; margin-top: 25px; }
        button { flex: 1; padding: 12px; font-size: 1rem; font-weight: bold; border: none; border-radius: 6px; cursor: pointer; transition: background 0.2s; }
        #startBtn { background: #22c55e; color: white; }
        #startBtn:hover { background: #16a34a; }
        #stopBtn { background: #ef4444; color: white; }
        #stopBtn:hover { background: #dc2626; }
        .conditional { display: none; margin-top: 8px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Nava Timecode Maker</h1>
        
        <div style="text-align: center;">
            <span id="statusBadge" class="status-badge stopped">Stopped</span>
        </div>
        
        <div class="tc-display" id="tcDisplay">00:00:00:00</div>

        <form id="configForm">
            <div class="form-group">
                <label>Time Source</label>
                <select id="time_source" name="time_source" onchange="toggleSourceFields()">
                    <option value="pc">PC Internal Clock</option>
                    <option value="ntp">NTP Server</option>
                    <option value="server">Custom Time Server (REST JSON)</option>
                    <option value="manual">Manual Start Time</option>
                </select>
            </div>

            <div class="form-group conditional" id="manualField">
                <label>Start Time (HH:MM:SS:FF)</label>
                <input type="text" id="manual_start" name="manual_start" value="00:00:00:00" placeholder="01:00:00:00" maxlength="11">
            </div>

            <div class="form-group conditional" id="ntpField">
                <label>NTP Server Address</label>
                <input type="text" id="ntp_server" name="ntp_server" value="pool.ntp.org">
            </div>

            <div class="form-group conditional" id="serverField">
                <label>Time Server URL</label>
                <input type="text" id="time_server_url" name="time_server_url" placeholder="http://your-server/time">
            </div>

            <div class="form-group">
                <label>Frame Rate (FPS)</label>
                <select id="fps" name="fps" onchange="toggleSourceFields()">
                    <option value="23.976">23.976 FPS</option>
                    <option value="24">24 FPS</option>
                    <option value="25">25 FPS (PAL)</option>
                    <option value="29.97">29.97 FPS</option>
                    <option value="30" selected>30 FPS</option>
                </select>
            </div>

            <div class="form-group conditional" id="dfField">
                <label>29.97 Counting Mode</label>
                <select id="drop_frame" name="drop_frame">
                    <option value="false">NDF (Non-Drop Frame)</option>
                    <option value="true">DF (Drop Frame)</option>
                </select>
            </div>

            <div class="form-group">
                <label>Audio Output Device</label>
                <select id="audio_device" name="audio_device">
                    <option value="default">Default System Audio (PC Internal Out)</option>
                </select>
            </div>

            <div class="form-group">
                <label>Time Offset (Seconds)</label>
                <input type="number" step="0.1" id="offset_seconds" name="offset_seconds" value="0">
            </div>

            <div class="actions">
                <button type="button" id="startBtn" onclick="startGenerator()">Start Generator</button>
                <button type="button" id="stopBtn" onclick="stopGenerator()">Stop</button>
            </div>
        </form>
    </div>

    <script>
        function toggleSourceFields() {
            const src = document.getElementById('time_source').value;
            document.getElementById('ntpField').style.display = (src === 'ntp') ? 'block' : 'none';
            document.getElementById('serverField').style.display = (src === 'server') ? 'block' : 'none';
            document.getElementById('manualField').style.display = (src === 'manual') ? 'block' : 'none';
            const fps = parseFloat(document.getElementById('fps').value);
            document.getElementById('dfField').style.display = (fps === 29.97) ? 'block' : 'none';
        }

        async function loadDevices() {
            const res = await fetch('/api/devices');
            const devices = await res.json();
            const select = document.getElementById('audio_device');
            
            select.innerHTML = '<option value="default">Default System Audio (PC Internal Out)</option>';
            
            devices.forEach(d => {
                const opt = document.createElement('option');
                opt.value = d.id;
                opt.textContent = `${d.name} (${d.hostapi})`;
                select.appendChild(opt);
            });
        }

        async function startGenerator() {
            const data = {
                time_source: document.getElementById('time_source').value,
                ntp_server: document.getElementById('ntp_server').value,
                time_server_url: document.getElementById('time_server_url').value,
                fps: parseFloat(document.getElementById('fps').value),
                drop_frame: document.getElementById('drop_frame').value === 'true',
                manual_start: document.getElementById('manual_start').value,
                audio_device: document.getElementById('audio_device').value,
                offset_seconds: parseFloat(document.getElementById('offset_seconds').value) || 0
            };

            const res = await fetch('/api/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await res.json();
            if (!res.ok) {
                alert(result.error || 'Could not start the generator');
                return;
            }
            updateStateUI(result);
        }

        async function stopGenerator() {
            const res = await fetch('/api/stop', { method: 'POST' });
            const result = await res.json();
            updateStateUI(result);
        }

        function updateStateUI(st) {
            const badge = document.getElementById('statusBadge');
            if (st.running) {
                badge.className = 'status-badge running';
                badge.textContent = 'Running (Audio Active)';
            } else {
                badge.className = 'status-badge stopped';
                badge.textContent = 'Stopped';
            }
        }

        async function pollStatus() {
            try {
                const res = await fetch('/api/status');
                const st = await res.json();
                updateStateUI(st);
                document.getElementById('tcDisplay').textContent = st.current_timecode;
            } catch (e) {}
        }

        loadDevices();
        setInterval(pollStatus, 250);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/devices")
def get_devices():
    devices = sd.query_devices()
    output_devices = []
    for i, dev in enumerate(devices):
        if dev["max_output_channels"] > 0:
            output_devices.append({
                "id": i,
                "name": dev["name"],
                "hostapi": sd.query_hostapis(dev["hostapi"])["name"],
            })
    return jsonify(output_devices)


@app.route("/api/status")
def get_status():
    return jsonify(state)


@app.route("/api/start", methods=["POST"])
def start_generator():
    global audio_thread
    if state["running"]:
        return jsonify(state)

    req_data = request.json or {}
    state["time_source"] = req_data.get("time_source", "pc")
    state["ntp_server"] = req_data.get("ntp_server", "pool.ntp.org")
    state["time_server_url"] = req_data.get("time_server_url", "")
    state["fps"] = float(req_data.get("fps", 30.0))
    state["drop_frame"] = bool(req_data.get("drop_frame", False))
    state["manual_start"] = str(req_data.get("manual_start", "00:00:00:00"))

    # Validate the manual start time before launching the audio thread
    if state["time_source"] == "manual":
        nominal = int(round(state["fps"]))
        drop = state["drop_frame"] and abs(state["fps"] - 29.97) < 0.01
        try:
            timecode_to_frame(state["manual_start"], nominal, drop)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
    state["audio_device"] = req_data.get("audio_device", "default")
    state["offset_seconds"] = float(req_data.get("offset_seconds", 0.0))
    state["running"] = True

    audio_thread = threading.Thread(target=audio_generator_worker, daemon=True)
    audio_thread.start()
    return jsonify(state)


@app.route("/api/stop", methods=["POST"])
def stop_generator():
    state["running"] = False
    return jsonify(state)


# Local-only by default. To allow other devices on the network to open the
# page, set the environment variable NAVA_HOST=0.0.0.0 before launching.
HOST = os.environ.get("NAVA_HOST", "127.0.0.1")
PREFERRED_PORT = int(os.environ.get("NAVA_PORT", "5000"))
PORT_TRIES = 20  # try PREFERRED_PORT .. PREFERRED_PORT + 19


def port_is_free(host, port):
    """True if we can bind to host:port right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def pick_port(host, preferred, tries):
    """Returns the first free port from preferred upward; falls back to an
    OS-assigned free port if all of them are taken."""
    for port in range(preferred, preferred + tries):
        if port_is_free(host, port):
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def open_browser_when_ready(port, timeout=15.0):
    """Waits for the server to accept connections, then opens the browser."""
    def _worker():
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.2)
        try:
            webbrowser.open(f"http://127.0.0.1:{port}/")
        except Exception:
            pass  # the URL is printed in the console as a fallback

    threading.Thread(target=_worker, daemon=True).start()


if __name__ == "__main__":
    port = pick_port(HOST, PREFERRED_PORT, PORT_TRIES)
    url = f"http://127.0.0.1:{port}/"

    print("=" * 60)
    print(" Nava Timecode Maker")
    if port != PREFERRED_PORT:
        print(f" (port {PREFERRED_PORT} was busy, using {port} instead)")
    print(f" Open in your browser: {url}")
    print(" Keep this window open while the generator is in use.")
    print(" Close this window (or press Ctrl+C) to quit.")
    print("=" * 60)

    open_browser_when_ready(port)

    try:
        from waitress import serve
    except ImportError:
        print("Waitress is not installed; using Flask's development server.")
        app.run(host=HOST, port=port, debug=False)
    else:
        # Keep this a single process: the generator state lives in memory.
        # Threads are fine; extra processes would each get their own state.
        serve(app, host=HOST, port=port, threads=4)
