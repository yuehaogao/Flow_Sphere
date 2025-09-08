# read_mindmonitor.py — EEG + Spectrum + PPG + Local BPM (Allo-ready)
# Yuehao Gao / Flow Sphere — 2025-09-06

from datetime import datetime, timedelta
from pythonosc import dispatcher, osc_server, udp_client
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import threading
import signal
import sys
import csv
import os
from collections import deque, defaultdict

# ============ CONFIG ============
# OSC listening (from Mind Monitor)
IP_LISTEN        = "0.0.0.0"
PORT_LISTEN      = 5089     # set the same port in Mind Monitor

# OSC forward (to Mock_EEG / Flow_Sphere)
IP_FORWARD       = "127.0.0.1"
PORT_FORWARD     = 9000

# EEG settings
N_CHANNELS       = 4
CH_LABELS        = ["TP9", "AF7", "AF8", "TP10"]
EEG_WIN          = 500            # samples shown in time plot / spectrum window
EEG_FS           = 256            # Muse 2 typical; adjust if needed

# If your EEG numbers are super large due to phone config, you can divide for plotting & forwarding.
# 1 means no forced scaling. Try 1000 or 10000 temporarily if needed.
EEG_FORCE_DIVISOR = 1.0

# Spectrum range
F_MIN, F_MAX     = 1.0, 60.0

# PPG / Pulse
PPG_WIN          = 1000           # samples in pulse plot
PPG_FS_EST       = 50             # approximate PPG sampling rate (Mind Monitor ~ 50Hz)
PPG_BUFFER_SEC   = 10             # seconds of PPG kept for detection (rolling)
PPG_REFRACTORY   = 0.35           # sec; avoid double counts
PPG_THRESH_K     = 0.8            # threshold = mean + k*std (adaptive)
BPM_SMOOTH_N     = 5              # smooth the last N beats (median of RR)
BPM_MIN, BPM_MAX = 30.0, 200.0    # plausible bpm
SHOW_DEBUG_UNKNOWN = True         # print unknown OSC paths (limited)

# Plot autoscale (only affects visuals)
AUTO_SCALE_EEG   = True
EEG_PLOT_TARGET  = 150.0          # map 95th percentile to ±150 (display only)
EEG_MIN_CLIP_UV  = 5.0

AUTO_SCALE_PPG   = True
PPG_PLOT_TARGET  = 1.0

# ============ STATE ============
alive = True
server = None
file_lock = threading.Lock()

# EEG buffers
eeg_raw   = np.zeros((N_CHANNELS, EEG_WIN), dtype=float)
eeg_plot  = np.zeros((N_CHANNELS, EEG_WIN), dtype=float)
time_lbls = [""] * EEG_WIN

# Spectrum precompute
freqs_full = np.fft.rfftfreq(EEG_WIN, d=1.0/EEG_FS)
freq_mask  = (freqs_full >= F_MIN) & (freqs_full <= F_MAX)
freqs      = freqs_full[freq_mask]
N_FREQS    = len(freqs)
spec_db    = np.zeros((N_CHANNELS, N_FREQS), dtype=float)

# Bands cache
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta" : (13.0, 30.0),
    "gamma": (30.0, 60.0),
}
band_masks_cached = {name: (freqs_full >= lo) & (freqs_full <= hi) for name, (lo, hi) in BANDS.items()}

# PPG buffers
pulse_raw  = np.zeros(PPG_WIN, dtype=float)   # for plotting (raw)
pulse_plot = np.zeros(PPG_WIN, dtype=float)   # for plotting (scaled)
pulse_lbls = [""] * PPG_WIN

ppg_ring   = deque(maxlen=int(PPG_BUFFER_SEC * max(PPG_FS_EST, 1)))
last_peak_t   = None
last_crossing = False
rr_deque      = deque(maxlen=BPM_SMOOTH_N)
bpm_value     = 0.0

# OSC sender
osc_out = udp_client.SimpleUDPClient(IP_FORWARD, PORT_FORWARD)

# Files
start_ts     = datetime.now()
fname_eeg    = start_ts.strftime("EEG_%Y_%m_%d_%H-%M-%S.csv")
fname_spec   = start_ts.strftime("EEG_SPECTRUM_%Y_%m_%d_%H-%M-%S.csv")
fname_pulse  = start_ts.strftime("EEG_PULSE_%Y_%m_%d_%H-%M-%S.csv")

f_eeg  = open(fname_eeg, 'w', newline='')
w_eeg  = csv.writer(f_eeg);  w_eeg.writerow(["Timestamp"] + CH_LABELS)

f_spec = open(fname_spec, 'w', newline='')
w_spec = csv.writer(f_spec); w_spec.writerow(["Timestamp", "Channel"] + [f"f_{f:.2f}Hz_dB" for f in freqs])

f_pul  = open(fname_pulse, 'w', newline='')
w_pul  = csv.writer(f_pul);  w_pul.writerow(["Timestamp", "Pulse", "BPM"])

# ============ EEG HANDLER ============
def eeg_handler(addr: str, *args):
    """
    /muse/eeg -> 4 floats
    CSV: raw values
    Forward: /eeg/raw [4], /eeg/bands [ch, d,t,a,b,g], /eeg/dominant [ch, freq]
    """
    global eeg_raw, eeg_plot, spec_db, time_lbls

    now = datetime.now()
    ts  = now.strftime("%Y-%m-%d %H:%M:%S.%f")

    vals = []
    for i in range(N_CHANNELS):
        try:
            v = float(args[i])
        except Exception:
            v = 0.0
        if EEG_FORCE_DIVISOR != 1.0:
            v = v / EEG_FORCE_DIVISOR
        vals.append(v)

    # CSV
    with file_lock:
        if alive and f_eeg and not f_eeg.closed:
            w_eeg.writerow([ts] + vals)

    # Forward raw EEG
    osc_out.send_message("/eeg/raw", vals)

    # Update ring buffers
    for ch in range(N_CHANNELS):
        eeg_raw[ch, :-1] = eeg_raw[ch, 1:]
        eeg_raw[ch, -1]  = vals[ch]
    time_lbls[:-1] = time_lbls[1:]
    time_lbls[-1]  = now.strftime("%M:%S")

    # Display scaling (only visuals)
    if AUTO_SCALE_EEG:
        for ch in range(N_CHANNELS):
            x = eeg_raw[ch]
            p95 = np.percentile(np.abs(x), 95)
            scale = EEG_PLOT_TARGET / max(p95, EEG_MIN_CLIP_UV)
            eeg_plot[ch] = x * scale
    else:
        eeg_plot[:] = eeg_raw

    # Spectrum per channel
    window = np.hanning(EEG_WIN)
    win_energy = np.sum(window**2) + 1e-12

    for ch in range(N_CHANNELS):
        x = eeg_raw[ch] - np.mean(eeg_raw[ch])
        X = np.fft.rfft(x * window)
        P = (np.abs(X)**2) / win_energy               # linear power
        P_band = P[freq_mask]
        P_db   = 10.0 * np.log10(P_band + 1e-12)
        spec_db[ch, :] = P_db

        # Spectrum CSV row
        with file_lock:
            if alive and f_spec and not f_spec.closed:
                w_spec.writerow([ts, CH_LABELS[ch]] + P_db.tolist())

        # Bands (linear sums)
        band_lin = {name: float(np.sum(P[mask])) for name, mask in band_masks_cached.items()}

        # Dominant freq in [F_MIN, F_MAX]
        dom_idx  = int(np.argmax(P[freq_mask])) if np.any(freq_mask) else 0
        dom_freq = float(freqs[dom_idx]) if N_FREQS else 0.0

        osc_out.send_message("/eeg/bands", [
            ch,
            band_lin["delta"], band_lin["theta"], band_lin["alpha"],
            band_lin["beta"], band_lin["gamma"]
        ])
        osc_out.send_message("/eeg/dominant", [ch, dom_freq])

# ============ PPG + local BPM ============
def ppg_handler(addr: str, *args):
    """
    Compatible with /muse/ppg, /muse/pulse, /pulse
    We estimate BPM locally from PPG peaks with adaptive threshold & refractory.
    """
    global pulse_raw, pulse_plot, pulse_lbls, bpm_value, last_peak_t, last_crossing

    now = datetime.now()
    ts  = now.strftime("%Y-%m-%d %H:%M:%S.%f")

    try:
        val = float(args[0])
    except Exception:
        return

    # Plot ring
    pulse_raw[:-1] = pulse_raw[1:]
    pulse_raw[-1]  = val
    pulse_lbls[:-1] = pulse_lbls[1:]
    pulse_lbls[-1]  = now.strftime("%M:%S")

    # Keep detection buffer (time, value)
    ppg_ring.append((now, val))

    # Adaptive threshold: mean + k*std over last ~2s data
    # (make sure enough samples exist)
    det_len = max(1, int(2.0 * PPG_FS_EST))
    if len(ppg_ring) >= det_len + 2:
        vals = np.array([v for (_, v) in list(ppg_ring)[-det_len:]], dtype=float)
        mu   = float(np.mean(vals))
        sd   = float(np.std(vals))
        high = mu + PPG_THRESH_K * sd

        # crossing up detection (Schmitt-like with refractory)
        t2, v2 = ppg_ring[-2]
        t3, v3 = ppg_ring[-1]

        crossed_up = (v2 < high) and (v3 >= high)
        # refractory window
        can_trigger = True
        if last_peak_t is not None:
            if (now - last_peak_t).total_seconds() < PPG_REFRACTORY:
                can_trigger = False

        if crossed_up and can_trigger:
            # refine peak: look back small neighborhood (~0.3s) for local max
            neigh = max(1, int(0.3 * PPG_FS_EST))
            window_pts = list(ppg_ring)[-neigh:]
            peak_t, peak_v = max(window_pts, key=lambda tv: tv[1])

            if last_peak_t is not None:
                rr = (peak_t - last_peak_t).total_seconds()
                if rr > 0:
                    bpm_inst = 60.0 / rr
                    if BPM_MIN <= bpm_inst <= BPM_MAX:
                        rr_deque.append(rr)
                        # median rr → smoother bpm
                        rr_med = float(np.median(rr_deque))
                        if rr_med > 0:
                            bpm_value = 60.0 / rr_med
            last_peak_t = peak_t

    # Forward OSC (raw + bpm)
    osc_out.send_message("/pulse/raw", [val])
    osc_out.send_message("/pulse/bpm", [float(bpm_value)])

    # CSV
    with file_lock:
        if alive and f_pul and not f_pul.closed:
            w_pul.writerow([ts, val, float(bpm_value)])

    # Plot scaling
    if AUTO_SCALE_PPG:
        p95 = np.percentile(np.abs(pulse_raw), 95)
        scale = (PPG_PLOT_TARGET) / max(p95, 1e-6)
        pulse_plot[:] = pulse_raw * scale
    else:
        pulse_plot[:] = pulse_raw

# ============ PLOTTING ============
# EEG time
fig_time, ax_time = plt.subplots()
time_lines = []
colors = ['C0','C1','C2','C3']
for i in range(N_CHANNELS):
    (ln,) = ax_time.plot(eeg_plot[i], label=CH_LABELS[i], color=colors[i])
    time_lines.append(ln)
ax_time.set_xlim(0, EEG_WIN)
ax_time.set_ylim(-EEG_PLOT_TARGET, EEG_PLOT_TARGET)
ax_time.set_title("Real-Time EEG (display auto-scaled)")
ax_time.set_ylabel("scaled μV"); ax_time.set_xlabel("Time")
ax_time.legend(loc="upper left")

def _update_time(_):
    for i in range(N_CHANNELS):
        time_lines[i].set_ydata(eeg_plot[i])
    xt = np.linspace(0, EEG_WIN-1, 6, dtype=int)
    ax_time.set_xticks(xt)
    labels = []
    step = max(1, EEG_WIN // 6)
    for k in range(0, EEG_WIN, step):
        labels.append(time_lbls[k]); 
        if len(labels) == len(xt): break
    while len(labels) < len(xt): labels.append("")
    ax_time.set_xticklabels(labels)
    return time_lines

ani_time = animation.FuncAnimation(fig_time, _update_time, interval=50, cache_frame_data=False)

# Spectrum heatmap
fig_spec, ax_spec = plt.subplots()
im = ax_spec.imshow(spec_db, aspect="auto", origin="lower",
                    extent=[F_MIN, F_MAX, 0, N_CHANNELS], cmap="RdYlGn_r")
ax_spec.set_title("Real-Time Spectrum Power (Channel × Frequency, dB)")
ax_spec.set_xlabel("Frequency (Hz)"); ax_spec.set_ylabel("Channel")
ax_spec.set_yticks(np.arange(N_CHANNELS) + 0.5)
ax_spec.set_yticklabels(CH_LABELS)
cbar = plt.colorbar(im, ax=ax_spec); cbar.set_label("Power (dB)")

def _update_spec(_):
    im.set_data(spec_db)
    im.set_clim(vmin=np.min(spec_db), vmax=np.max(spec_db) + 1e-12)
    return [im]

ani_spec = animation.FuncAnimation(fig_spec, _update_spec, interval=100, cache_frame_data=False)

# Pulse window
fig_pulse, ax_pulse = plt.subplots()
(pulse_ln,) = ax_pulse.plot(pulse_plot, label="Pulse")
ax_pulse.set_xlim(0, PPG_WIN)
ax_pulse.set_ylim(-PPG_PLOT_TARGET, PPG_PLOT_TARGET)
ax_pulse.set_title("Pulse (PPG) — BPM: --")
ax_pulse.set_xlabel("Time"); ax_pulse.set_ylabel("scaled pulse")
ax_pulse.legend(loc="upper left")

def _update_pulse(_):
    y = pulse_plot
    # gentle autoscale
    ymax = max(1e-6, np.percentile(np.abs(y), 98))
    ax_pulse.set_ylim(-1.2*ymax, 1.2*ymax)
    pulse_ln.set_ydata(y)
    ax_pulse.set_title(f"Pulse (PPG) — BPM: {bpm_value:.1f}")
    xt = np.linspace(0, PPG_WIN-1, 6, dtype=int)
    ax_pulse.set_xticks(xt)
    labels = []
    step = max(1, PPG_WIN // 6)
    for k in range(0, PPG_WIN, step):
        labels.append(pulse_lbls[k]); 
        if len(labels) == len(xt): break
    while len(labels) < len(xt): labels.append("")
    ax_pulse.set_xticklabels(labels)
    return [pulse_ln]

ani_pulse = animation.FuncAnimation(fig_pulse, _update_pulse, interval=100, cache_frame_data=False)

# ============ CLEANUP ============
def _cleanup_and_exit():
    with file_lock:
        try:
            if f_eeg  and not f_eeg.closed:  f_eeg.close()
            if f_spec and not f_spec.closed: f_spec.close()
            if f_pul  and not f_pul.closed:  f_pul.close()
        except Exception as e:
            print("Close file error:", e)
    print(f"✅ Saved EEG:      {os.path.abspath(fname_eeg)}")
    print(f"✅ Saved Spectrum: {os.path.abspath(fname_spec)}")
    print(f"✅ Saved Pulse:    {os.path.abspath(fname_pulse)}")
    plt.close('all'); sys.exit(0)

def _handle_exit(sig, frame):
    global alive, server
    print("\n🧠 Stopping... Saving files.")
    alive = False
    try:
        if server is not None:
            server.shutdown()
            server.server_close()
    except Exception as e:
        print("Server stop error:", e)
    _cleanup_and_exit()

signal.signal(signal.SIGINT, _handle_exit)
signal.signal(signal.SIGTERM, _handle_exit)

# ============ SERVER ============
_unknown_count = defaultdict(int)
_MAX_PRINT = 60
def default_handler(address, *args):
    if not SHOW_DEBUG_UNKNOWN: return
    c = _unknown_count[address]
    if c < _MAX_PRINT:
        # print(f"[OSC?] {address} -> {args}")
        _unknown_count[address] += 1

if __name__ == "__main__":
    disp = dispatcher.Dispatcher()
    disp.set_default_handler(default_handler)

    # EEG
    disp.map("/muse/eeg", eeg_handler)

    # PPG / Pulse (cover common variants)
    for addr in ["/muse/ppg", "/muse/pulse", "/pulse"]:
        disp.map(addr, ppg_handler)

    server = osc_server.ThreadingOSCUDPServer((IP_LISTEN, PORT_LISTEN), disp)
    print(f"✅ Listening on {IP_LISTEN}:{PORT_LISTEN}, forwarding to {IP_FORWARD}:{PORT_FORWARD}")
    print("• PPG addrs watched: /muse/ppg, /muse/pulse, /pulse")
    threading.Thread(target=server.serve_forever, daemon=True).start()

    plt.show()
