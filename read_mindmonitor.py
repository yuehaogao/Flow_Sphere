from pythonosc import dispatcher, osc_server, udp_client
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import threading
import time
from scipy.signal import butter, filtfilt, find_peaks
import atexit

import pandas as pd

log_data = []  # Dictionary
start_time = time.time()


waves = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'flow']
buffer_size = 200
eeg_channels = 4
sampling_rate = 256

data = {wave: deque([0.0]*buffer_size, maxlen=buffer_size) for wave in waves}
eeg_raw = deque([[0.0]*eeg_channels for _ in range(buffer_size)], maxlen=buffer_size)

flow_history = deque(maxlen=20)

# -------------------
# OSC Client（Sending to Flow_Sphere.cpp）
# -------------------
client_ip = "127.0.0.1"
client_port = 9000  # Flow_Sphere.cpp should listen to this port
osc_client = udp_client.SimpleUDPClient(client_ip, client_port)

# -------------------
# OSC Receiving
# -------------------
def handle_band(wave):
    def handler(address, *args):
        value = sum(args) / len(args)
        data[wave].append(value)
        update_flow()
        timestamp = time.time() - start_time
        log_data.append({
            'timestamp': timestamp,
            'type': 'band',
            'wave': wave,
            'value': value
        })
    return handler

def handle_eeg(address, *args):
    timestamp = time.time() - start_time
    eeg_raw.append(list(args))
    
    # Trying to send raw EEG to Allolib
    osc_client.send_message("/eeg/raw", list(args))
    
    
    # print("EEG µV:", ", ".join([f"Ch{i+1}: {v:7.1f}" for i, v in enumerate(args)]))
    
    log_data.append({
        'timestamp': timestamp,
        'type': 'eeg_raw',
        'ch1': args[0], 'ch2': args[1], 'ch3': args[2], 'ch4': args[3]
    })

def update_flow():
    alpha = data['alpha'][-1]
    theta = data['theta'][-1]
    beta = data['beta'][-1]
    gamma = data['gamma'][-1]

    epsilon = 1e-6
    raw_flow = (alpha + 0.8 * theta + 0.5 * beta) / (0.5 * gamma + epsilon)

    flow_history.append(raw_flow)
    smoothed_flow = sum(flow_history) / len(flow_history)

    data['flow'].append(smoothed_flow)

    # If sending OSC：
    # osc_client.send_message("/muse/flow", smoothed_flow)


# -------------------
# Filtering Curve
# -------------------
def bandpass_filter(data, lowcut=0.8, highcut=3.0, fs=256, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# -------------------
# Visualization
# -------------------
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
x = np.arange(buffer_size)

# 1. Brain signals（单位 dB）
lines = {wave: ax1.plot(x, list(data[wave]), label=wave)[0] for wave in waves}
ax1.set_ylim(0, 1)
ax1.set_title("Brainwave Band Powers")
ax1.set_ylabel("Power (dB)")
ax1.legend()

# 2. Spectrum（in dB）
fft_lines = {wave: ax2.plot([], [], label=wave + ' FFT')[0] for wave in waves}
ax2.set_ylim(0, 100)
ax2.set_xlim(0, buffer_size // 2)
ax2.set_title("FFT Spectrum")
ax2.set_ylabel("Amplitude (dB)")
ax2.legend()

# 3. Raw EEG（in μV）
eeg_lines = [ax3.plot(x, [row[i] for row in eeg_raw], label=f"EEG {i+1}")[0] for i in range(eeg_channels)]
ax3.set_ylim(-1000, 1000)
ax3.set_title("Raw EEG (4 channels)")
ax3.set_ylabel("Voltage (µV)")
ax3.legend()

# 4. Heart Beat
heartbeat_line, = ax4.plot(x, [0]*buffer_size, label="Heartbeat Signal", color="red")
heartbeat_peaks, = ax4.plot([], [], 'go', label="Peaks")
ax4.set_ylim(-20, 20)
ax4.set_title("Estimated Heartbeat Waveform (from EEG TP9)")
ax4.set_ylabel("Voltage (µV)")
ax4.legend()

# -------------------
# FFT Calculation
# -------------------
def compute_fft(signal):
    y = np.array(signal)
    fft_vals = np.abs(np.fft.rfft(y))
    return fft_vals

# -------------------
# Refresh Visualization
# -------------------
def animate(i):
    # 1. Brain Signal
    for wave in waves:
        lines[wave].set_ydata(list(data[wave]))
        fft_y = compute_fft(list(data[wave]))
        fft_x = np.linspace(0, 50, len(fft_y))
        fft_lines[wave].set_data(fft_x, fft_y)

    # 2. Raw EEG
    for i in range(eeg_channels):
        eeg_lines[i].set_ydata([row[i] for row in eeg_raw])

    # 3. Heart Beat
    eeg_channel = [row[0] for row in eeg_raw]
    filtered = bandpass_filter(eeg_channel)
    heartbeat_line.set_ydata(filtered)

    peaks, _ = find_peaks(filtered, distance=sampling_rate/2)
    peak_x = peaks[-10:]  # only showing the latest
    peak_y = [filtered[i] for i in peak_x]
    heartbeat_peaks.set_data(peak_x, peak_y)

    return list(lines.values()) + list(fft_lines.values()) + eeg_lines + [heartbeat_line, heartbeat_peaks]

ani = animation.FuncAnimation(fig, animate, interval=100, blit=True)

# -------------------
# Start OSC Server
# -------------------
dispatcher = dispatcher.Dispatcher()
for wave in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
    dispatcher.map(f"/muse/elements/{wave}_absolute", handle_band(wave))
dispatcher.map("/muse/eeg", handle_eeg)

ip = "0.0.0.0"
port = 5089
server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
threading.Thread(target=server.serve_forever, daemon=True).start()

print(f"✅ Listening on {ip}:{port} ... Forwarding EEG to {client_ip}:{client_port}")



def save_log():
    if log_data:
        print("Program Exited, Now Saving EEG Data...")
        df = pd.DataFrame(log_data)
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        filename = f"eeg_log_{timestamp_str}.csv"
        df.to_csv(filename, index=False)
        print(f"成功保存, Data has been saved into {filename}")

atexit.register(save_log)

plt.tight_layout()
plt.show()



