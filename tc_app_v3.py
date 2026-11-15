import datetime
import os
import socket
import sys
import threading
import time
import webbrowser
from flask import Flask, jsonify, render_template_string, request
import numpy as np
import ntplib
import requests

try:
    import psutil  # optional: gives adapter names; falls back to sockets if missing
except ImportError:
    psutil = None

try:
    import sounddevice as sd
except OSError as _e:
    sd = None
    print(f"WARNING: server audio library unavailable ({_e}). Browser audio still works.")

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

app = Flask(__name__)

state = {
    "running": False,
    "time_source": "pc",
    "ntp_server": "pool.ntp.org",
    "time_server_url": "",
    "fps": 29.97,
    "timezone": "local",
    "drop_frame": True,
    "manual_start": "00:00:00:00",
    "audio_device": "browser",
    "sample_rate": 48000,
    "offset_seconds": 0.0,
    "current_timecode": "00:00:00:00",
    "clock_offset": 0.0,
}

audio_thread = None
_ltc_level = 1.0
AMPLITUDE = 0.5


def get_tz(name):
    if not name or name == "local":
        return None
    hours = int(name)
    if not -12 <= hours <= 12:
        raise ValueError("Time zone offset must be between UTC-12 and UTC+12")
    return datetime.timezone(datetime.timedelta(hours=hours))


TZ_OPTIONS = '<option value="local">Local (Server PC\'s time zone)</option>' + "".join(
    f'<option value="{h}">{"UTC" if h == 0 else f"UTC{h:+d}"}</option>'
    for h in range(12, -13, -1)
)


def real_fps(fps):
    if abs(fps - 23.976) < 0.01:
        return 24000.0 / 1001.0
    if abs(fps - 29.97) < 0.01:
        return 30000.0 / 1001.0
    return float(fps)


def frame_to_timecode(n, nominal, drop):
    if drop:
        frames_per_day = 2589408
        n %= frames_per_day
        d, mm = divmod(n, 17982)
        n += 18 * d + (2 * ((mm - 2) // 1798) if mm >= 2 else 0)
    else:
        n %= nominal * 86400
    f = n % nominal
    s = (n // nominal) % 60
    m = (n // (nominal * 60)) % 60
    h = (n // (nominal * 3600)) % 24
    return h, m, s, f


def timecode_to_frame(text, nominal, drop):
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
    global _ltc_level
    nominal = int(round(fps))
    if samples_per_frame is None:
        samples_per_frame = int(round(sample_rate / real_fps(fps)))

    bits = [0] * 80

    def put(start, count, value):
        for i in range(count):
            bits[start + i] = (value >> i) & 1

    put(0, 4, f % 10)
    put(8, 2, f // 10)
    bits[10] = 1 if drop_frame else 0
    put(16, 4, s % 10)
    put(24, 3, s // 10)
    put(32, 4, m % 10)
    put(40, 3, m // 10)
    put(48, 4, h % 10)
    put(56, 2, h // 10)
    bits[64:80] = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]

    parity_pos = 59 if nominal == 25 else 27
    if bits.count(0) % 2 == 1:
        bits[parity_pos] = 1

    edges = (np.arange(81) * samples_per_frame) // 80
    out = np.empty(samples_per_frame, dtype=np.float32)
    level = _ltc_level
    for i, bit in enumerate(bits):
        start, end = int(edges[i]), int(edges[i + 1])
        n = end - start
        level = -level
        seg = np.full(n, level, dtype=np.float32)
        if bit == 1:
            seg[n // 2:] = -level
            level = -level
        out[start:end] = seg
    _ltc_level = level
    return out * AMPLITUDE


def sync_external_clock():
    if state["time_source"] == "ntp":
        try:
            ntp_client = ntplib.NTPClient()
            response = ntp_client.request(state["ntp_server"], version=3, timeout=1.5)
            state["clock_offset"] = response.tx_time - time.time()
        except Exception:
            pass
    elif state["time_source"] == "server" and state["time_server_url"]:
        try:
            resp = requests.get(state["time_server_url"], timeout=1.0)
            data = resp.json()
            server_time = float(data.get("time", data.get("timestamp", time.time())))
            state["clock_offset"] = server_time - time.time()
        except Exception:
            pass
    else:
        state["clock_offset"] = 0.0


def audio_generator_worker():
    try:
        sample_rate = state["sample_rate"]
        fps = state["fps"]
        rfps = real_fps(fps)
        nominal = int(round(fps))
        drop = bool(state["drop_frame"]) and nominal == 30 and rfps != 30.0
        sample_acc = 0.0

        tz = get_tz(state["timezone"]) if state["time_source"] == "ntp" else None

        manual_start_sod = 0.0
        manual_t0 = time.time()
        if state["time_source"] == "manual":
            start_frame = timecode_to_frame(state["manual_start"], nominal, drop)
            manual_start_sod = start_frame / rfps

        device_val = state["audio_device"]
        device = None if device_val in ("", "default", None) else int(device_val)

        stream = sd.OutputStream(
            samplerate=sample_rate, channels=1, dtype="float32", device=device
        )
        stream.start()

        last_sync = 0.0
        while state["running"] and state["audio_device"] != "browser":
            now = time.time()
            if state["time_source"] in ("ntp", "server") and now - last_sync > 15:
                sync_external_clock()
                last_sync = now

            current_time = time.time() + state["clock_offset"] + state["offset_seconds"] + stream.latency

            if state["time_source"] == "manual":
                seconds_of_day = (
                    manual_start_sod
                    + (time.time() - manual_t0)
                    + state["offset_seconds"]
                    + stream.latency
                ) % 86400.0
            else:
                dt = datetime.datetime.fromtimestamp(current_time, tz=tz)
                seconds_of_day = (
                    dt.hour * 3600 + dt.minute * 60 + dt.second
                    + dt.microsecond / 1_000_000.0
                )
            frame_no = int(seconds_of_day * rfps)
            h, m, s, f = frame_to_timecode(frame_no, nominal, drop)

            sep = ";" if drop else ":"
            state["current_timecode"] = f"{h:02d}:{m:02d}:{s:02d}{sep}{f:02d}"

            sample_acc += sample_rate / rfps
            spf = int(sample_acc)
            sample_acc -= spf

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


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="theme-color" content="#0f172a">
    <title>Nava Timecode Maker</title>
    <style>
        html { -webkit-text-size-adjust: 100%; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 40px; display: flex; justify-content: center; min-height: 100vh; box-sizing: border-box; }
        .container { width: 100%; max-width: 650px; background: #1e293b; padding: 30px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
        h1 { font-size: 1.5rem; margin-bottom: 20px; color: #38bdf8; text-align: center; }
        .tc-display { background: #020617; border: 2px solid #334155; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: clamp(1.6rem, 9.5vw, 3.5rem); white-space: nowrap; overflow: hidden; text-align: center; padding: 20px; border-radius: 8px; color: #4ade80; letter-spacing: 2px; margin-bottom: 25px; box-shadow: inset 0 2px 6px rgba(0,0,0,0.8); }
        .status-badge { display: inline-block; padding: 4px 10px; border-radius: 4px; font-size: 0.85rem; font-weight: bold; text-transform: uppercase; margin-bottom: 20px; }
        .stopped { background: #ef4444; color: white; }
        .running { background: #22c55e; color: white; }
        .form-group { margin-bottom: 15px; }
        label { display: block; font-size: 0.9rem; margin-bottom: 5px; color: #cbd5e1; }
        select, input { width: 100%; padding: 10px; font-size: 16px; background: #0f172a; border: 1px solid #475569; color: white; border-radius: 6px; box-sizing: border-box; -webkit-appearance: none; appearance: none; }
        select { background-image: linear-gradient(45deg, transparent 50%, #94a3b8 50%), linear-gradient(135deg, #94a3b8 50%, transparent 50%); background-position: calc(100% - 18px) 50%, calc(100% - 13px) 50%; background-size: 5px 5px, 5px 5px; background-repeat: no-repeat; padding-right: 34px; }
        button { touch-action: manipulation; -webkit-tap-highlight-color: transparent; }
        .actions { display: flex; gap: 10px; margin-top: 25px; }
        button { flex: 1; padding: 12px; font-size: 1rem; font-weight: bold; border: none; border-radius: 6px; cursor: pointer; transition: background 0.2s; }
        #startBtn { background: #22c55e; color: white; }
        #startBtn:hover { background: #16a34a; }
        #stopBtn { background: #ef4444; color: white; }
        #stopBtn:hover { background: #dc2626; }
        .conditional { display: none; margin-top: 8px; }
        @media (max-width: 600px) {
            body { padding: max(12px, env(safe-area-inset-top)) max(12px, env(safe-area-inset-right)) max(12px, env(safe-area-inset-bottom)) max(12px, env(safe-area-inset-left)); }
            .container { padding: 18px 14px; border-radius: 10px; }
            h1 { font-size: 1.3rem; margin-top: 0; }
            .tc-display { padding: 14px 6px; letter-spacing: 0; margin-bottom: 18px; }
            select, input { padding: 12px; min-height: 46px; }
            button { min-height: 50px; font-size: 1.05rem; }
        }
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
                    <option value="pc">Clock (Synced)</option>
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

            <div class="form-group conditional" id="tzField">
                <label>Time Zone</label>
                <select id="timezone" name="timezone">
                    {{ tz_options|safe }}
                </select>
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
                    <option value="29.97" selected>29.97 FPS</option>
                    <option value="30">30 FPS</option>
                </select>
            </div>

            <div class="form-group conditional" id="dfField">
                <label>29.97 Counting Mode</label>
                <select id="drop_frame" name="drop_frame">
                    <option value="true" selected>DF (Drop Frame)</option>
                    <option value="false">NDF (Non-Drop Frame)</option>
                </select>
            </div>

            <div class="form-group">
                <label>Audio Output Device</label>
                <select id="audio_device" name="audio_device">
                    <option value="browser">This Device / Browser Audio</option>
                    <option value="default">Host Server Default Audio</option>
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
        const isRemoteClient = !['localhost', '127.0.0.1', '::1'].includes(window.location.hostname);

        let audioCtx = null;
        let scriptNode = null;
        let browserRunning = false;
        let serverClockDelta = 0;

        let ltcLevel = 1.0;
        let sampleAcc = 0.0;
        let frameBuffer = new Float32Array(0);
        let frameReadPos = 0;
        let manualStartSod = 0;
        let manualT0 = 0;

        function toggleSourceFields() {
            const src = document.getElementById('time_source').value;
            document.getElementById('ntpField').style.display = (src === 'ntp') ? 'block' : 'none';
            document.getElementById('serverField').style.display = (src === 'server') ? 'block' : 'none';
            document.getElementById('tzField').style.display = (src === 'ntp') ? 'block' : 'none';
            document.getElementById('manualField').style.display = (src === 'manual') ? 'block' : 'none';
            const fps = parseFloat(document.getElementById('fps').value);
            document.getElementById('dfField').style.display = (fps === 29.97) ? 'block' : 'none';
        }

        async function loadDevices() {
            const res = await fetch('/api/devices');
            const devices = await res.json();
            const select = document.getElementById('audio_device');

            select.innerHTML = `
                <option value="browser">This Device / Browser Audio (Recommended for Remote)</option>
                <option value="default">Host Server Default Audio (Server PC Out)</option>
            `;

            devices.forEach(d => {
                const opt = document.createElement('option');
                opt.value = d.id;
                opt.textContent = `Host Server: ${d.name} (${d.hostapi})`;
                select.appendChild(opt);
            });

            select.value = 'browser';
        }

        function realFps(fps) {
            if (Math.abs(fps - 23.976) < 0.01) return 24000.0 / 1001.0;
            if (Math.abs(fps - 29.97) < 0.01) return 30000.0 / 1001.0;
            return fps;
        }

        function frameToTimecode(n, nominal, drop) {
            if (drop) {
                n = ((n % 2589408) + 2589408) % 2589408;
                const d = Math.floor(n / 17982);
                const mm = n % 17982;
                n += 18 * d + (mm >= 2 ? 2 * Math.floor((mm - 2) / 1798) : 0);
            } else {
                const dayFrames = nominal * 86400;
                n = ((n % dayFrames) + dayFrames) % dayFrames;
            }
            const f = n % nominal;
            const s = Math.floor(n / nominal) % 60;
            const m = Math.floor(n / (nominal * 60)) % 60;
            const h = Math.floor(n / (nominal * 3600)) % 24;
            return [h, m, s, f];
        }

        function timecodeToFrame(text, nominal, drop) {
            const parts = text.replace(/;/g, ':').replace(/[.]/g, ':').trim().split(':');
            if (parts.length !== 4) throw new Error('Start time must look like HH:MM:SS:FF');
            const [h, m, s, f] = parts.map(Number);
            if ([h, m, s, f].some(isNaN) || h < 0 || h >= 24 || m < 0 || m >= 60 || s < 0 || s >= 60 || f < 0 || f >= nominal) {
                throw new Error(`Start time out of range for ${nominal} frames/sec`);
            }
            if (drop) {
                if (s === 0 && f < 2 && m % 10 !== 0) {
                    throw new Error('Frames 00 and 01 do not exist at the start of this minute in drop-frame');
                }
                const totalMinutes = 60 * h + m;
                return (h * 3600 + m * 60 + s) * 30 + f - 2 * (totalMinutes - Math.floor(totalMinutes / 10));
            }
            return (h * 3600 + m * 60 + s) * nominal + f;
        }

        function encodeSmpteFrame(h, m, s, f, nominal, drop, spf) {
            const bits = new Array(80).fill(0);
            const put = (start, count, val) => {
                for (let i = 0; i < count; i++) bits[start + i] = (val >> i) & 1;
            };
            put(0, 4, f % 10);
            put(8, 2, Math.floor(f / 10));
            bits[10] = drop ? 1 : 0;
            put(16, 4, s % 10);
            put(24, 3, Math.floor(s / 10));
            put(32, 4, m % 10);
            put(40, 3, Math.floor(m / 10));
            put(48, 4, h % 10);
            put(56, 2, Math.floor(h / 10));

            const sync = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1];
            for (let i = 0; i < 16; i++) bits[64 + i] = sync[i];

            const parityPos = (nominal === 25) ? 59 : 27;
            const zeros = bits.filter(b => b === 0).length;
            if (zeros % 2 === 1) bits[parityPos] = 1;

            const out = new Float32Array(spf);
            let level = ltcLevel;
            for (let i = 0; i < 80; i++) {
                const start = Math.floor((i * spf) / 80);
                const end = Math.floor(((i + 1) * spf) / 80);
                const mid = start + Math.floor((end - start) / 2);
                level = -level;
                if (bits[i] === 1) {
                    for (let j = start; j < mid; j++) out[j] = level * 0.5;
                    level = -level;
                    for (let j = mid; j < end; j++) out[j] = level * 0.5;
                } else {
                    for (let j = start; j < end; j++) out[j] = level * 0.5;
                }
            }
            ltcLevel = level;
            return out;
        }

        function startBrowserAudio(config, serverClockOffset) {
            stopBrowserAudio();
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            audioCtx = new AudioContextClass({ sampleRate: 48000 });
            if (audioCtx.state === 'suspended') audioCtx.resume();

            const sampleRate = audioCtx.sampleRate;
            const fps = config.fps;
            const rfps = realFps(fps);
            const nominal = Math.round(fps);
            const drop = Boolean(config.drop_frame) && nominal === 30 && rfps !== 30.0;

            ltcLevel = 1.0;
            sampleAcc = 0.0;
            frameBuffer = new Float32Array(0);
            frameReadPos = 0;

            if (config.time_source === 'manual') {
                const startFrame = timecodeToFrame(config.manual_start, nominal, drop);
                manualStartSod = startFrame / rfps;
                manualT0 = Date.now() / 1000.0;
            }

            scriptNode = audioCtx.createScriptProcessor(2048, 0, 1);
            scriptNode.onaudioprocess = (e) => {
                if (!browserRunning) return;
                const out = e.outputBuffer.getChannelData(0);
                const latency = audioCtx.outputLatency || (2048 / sampleRate);
                const offsetSec = config.offset_seconds || 0;

                for (let i = 0; i < out.length; i++) {
                    if (frameReadPos >= frameBuffer.length) {
                        const nowSec = (Date.now() / 1000.0) + serverClockDelta + serverClockOffset + offsetSec + latency;
                        let sod;
                        if (config.time_source === 'manual') {
                            sod = (manualStartSod + ((Date.now() / 1000.0) - manualT0) + offsetSec + latency) % 86400.0;
                        } else if (config.time_source === 'ntp' && config.timezone !== 'local') {
                            const tzHours = parseInt(config.timezone, 10) || 0;
                            const shifted = nowSec + tzHours * 3600;
                            sod = ((shifted % 86400) + 86400) % 86400;
                        } else {
                            const d = new Date(nowSec * 1000);
                            sod = d.getHours() * 3600 + d.getMinutes() * 60 + d.getSeconds() + d.getMilliseconds() / 1000.0;
                        }

                        const frameNo = Math.floor(sod * rfps);
                        const [h, m, s, f] = frameToTimecode(frameNo, nominal, drop);

                        const sep = drop ? ';' : ':';
                        const pad = (n) => String(n).padStart(2, '0');
                        document.getElementById('tcDisplay').textContent = `${pad(h)}:${pad(m)}:${pad(s)}${sep}${pad(f)}`;

                        sampleAcc += sampleRate / rfps;
                        const spf = Math.floor(sampleAcc);
                        sampleAcc -= spf;

                        frameBuffer = encodeSmpteFrame(h, m, s, f, nominal, drop, spf);
                        frameReadPos = 0;
                    }
                    out[i] = frameBuffer[frameReadPos++];
                }
            };

            scriptNode.connect(audioCtx.destination);
            browserRunning = true;
        }

        function stopBrowserAudio() {
            browserRunning = false;
            if (scriptNode) {
                scriptNode.disconnect();
                scriptNode = null;
            }
            if (audioCtx) {
                audioCtx.close();
                audioCtx = null;
            }
        }

        async function startGenerator() {
            const data = {
                time_source: document.getElementById('time_source').value,
                ntp_server: document.getElementById('ntp_server').value,
                time_server_url: document.getElementById('time_server_url').value,
                fps: parseFloat(document.getElementById('fps').value),
                drop_frame: document.getElementById('drop_frame').value === 'true',
                manual_start: document.getElementById('manual_start').value,
                timezone: document.getElementById('timezone').value,
                audio_device: document.getElementById('audio_device').value,
                offset_seconds: parseFloat(document.getElementById('offset_seconds').value) || 0
            };

            const t0 = Date.now() / 1000.0;
            const res = await fetch('/api/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const t1 = Date.now() / 1000.0;
            const result = await res.json();

            if (!res.ok) {
                alert(result.error || 'Could not start the generator');
                return;
            }

            if (result.server_time) {
                serverClockDelta = result.server_time - ((t0 + t1) / 2.0);
            }

            if (data.audio_device === 'browser') {
                try {
                    startBrowserAudio(data, result.clock_offset || 0);
                } catch (err) {
                    alert(err.message);
                    await stopGenerator();
                    return;
                }
            } else {
                stopBrowserAudio();
            }

            updateStateUI(result);
        }

        async function stopGenerator() {
            stopBrowserAudio();
            const res = await fetch('/api/stop', { method: 'POST' });
            const result = await res.json();
            updateStateUI(result);
        }

        // Automatically turn off generator and close socket/streams when browser/tab exits
        window.addEventListener('pagehide', () => {
            stopBrowserAudio();
            navigator.sendBeacon('/api/stop');
        });

        function updateStateUI(st) {
            const badge = document.getElementById('statusBadge');
            const active = st.running || browserRunning;
            if (active) {
                badge.className = 'status-badge running';
                badge.textContent = browserRunning ? 'Running (This Device Audio)' : 'Running (Server Audio)';
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
                if (!browserRunning) {
                    document.getElementById('tcDisplay').textContent = st.current_timecode;
                }
            } catch (e) {}
        }

        toggleSourceFields();
        loadDevices();
        setInterval(pollStatus, 250);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, tz_options=TZ_OPTIONS)


@app.route("/api/devices")
def get_devices():
    if sd is None:
        return jsonify([])
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
    req_data = request.json or {}
    audio_device = req_data.get("audio_device", "default")

    if state["running"] and audio_device != "browser":
        return jsonify({**state, "server_time": time.time()})

    if audio_device != "browser" and sd is None:
        return jsonify({
            "error": "Audio library (PortAudio) is not available on this server. Select 'This Device / Browser Audio' instead."
        }), 500

    state["time_source"] = req_data.get("time_source", "pc")
    state["ntp_server"] = req_data.get("ntp_server", "pool.ntp.org")
    state["time_server_url"] = req_data.get("time_server_url", "")
    state["fps"] = float(req_data.get("fps", 30.0))
    state["drop_frame"] = bool(req_data.get("drop_frame", False))
    state["manual_start"] = str(req_data.get("manual_start", "00:00:00:00"))
    state["timezone"] = str(req_data.get("timezone", "local"))
    state["audio_device"] = audio_device
    state["offset_seconds"] = float(req_data.get("offset_seconds", 0.0))

    if state["time_source"] == "ntp":
        try:
            get_tz(state["timezone"])
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    if state["time_source"] == "manual":
        nominal = int(round(state["fps"]))
        drop = state["drop_frame"] and abs(state["fps"] - 29.97) < 0.01
        try:
            timecode_to_frame(state["manual_start"], nominal, drop)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    sync_external_clock()
    state["running"] = True

    if audio_device != "browser":
        audio_thread = threading.Thread(target=audio_generator_worker, daemon=True)
        audio_thread.start()

    return jsonify({**state, "server_time": time.time()})


@app.route("/api/stop", methods=["POST"])
def stop_generator():
    state["running"] = False
    return jsonify(state)


def port_is_free(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def find_free_port(host, preferred, tries=20):
    for port in range(preferred, preferred + tries):
        if port_is_free(host, port):
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def get_lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("10.255.255.255", 1))
            return sock.getsockname()[0]
    except OSError:
        return None


def get_bind_addresses(host):
    """Returns [(interface_name, ipv4_address), ...] the server is reachable on.

    A specific host binds only that address. '0.0.0.0' binds every IPv4
    interface on this computer, so all of them are listed.
    """
    if host not in ("0.0.0.0", ""):
        return [("", host)]

    found = []

    def add(name, ip):
        if ip and not any(ip == existing for _, existing in found):
            found.append((name, ip))

    if psutil is not None:
        try:
            for name, addrs in psutil.net_if_addrs().items():
                for a in addrs:
                    if a.family == socket.AF_INET:
                        add(name, a.address)
        except Exception:
            pass

    if not found:  # fallback without psutil: no adapter names
        try:
            for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
                add("", ip)
        except OSError:
            pass
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                add("", info[4][0])
        except OSError:
            pass
        add("", get_lan_ip())
        add("Loopback", "127.0.0.1")

    # Loopback last, everything else in discovery order
    found.sort(key=lambda item: item[1].startswith("127."))
    return found


def open_browser_when_ready(port, url, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.2)
    webbrowser.open(url)


if __name__ == "__main__":
    host = os.environ.get("NAVA_HOST", "0.0.0.0")
    try:
        preferred_port = int(os.environ.get("NAVA_PORT", "5000"))
    except ValueError:
        preferred_port = 5000

    port = find_free_port(host, preferred_port)
    url = f"http://127.0.0.1:{port}"

    print("=" * 56)
    print(" Nava Timecode Maker")
    if port != preferred_port:
        print(f" Port {preferred_port} is in use, using port {port} instead.")
    addresses = get_bind_addresses(host)
    if host in ("0.0.0.0", ""):
        print(f" Listening on ALL network interfaces, port {port}:")
    else:
        print(f" Listening on {host}, port {port}:")
    width = max((len(name) for name, _ in addresses), default=0)
    for name, ip in addresses:
        label = f"{name:<{width}}  " if width else ""
        print(f"   {label}http://{ip}:{port}")
    if host in ("0.0.0.0", ""):
        print(" Other devices on the same network can use any non-127 address above.")
    print(" Close this window (or press Ctrl+C) to quit.")
    print("=" * 56)

    if os.environ.get("NAVA_NO_BROWSER") != "1":
        threading.Thread(
            target=open_browser_when_ready, args=(port, url), daemon=True
        ).start()

    try:
        from waitress import serve
    except ImportError:
        print("Waitress is not installed; using Flask's development server.")
        app.run(host=host, port=port, debug=False)
    else:
        serve(app, host=host, port=port, threads=4)