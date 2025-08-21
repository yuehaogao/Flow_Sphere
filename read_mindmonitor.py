from datetime import datetime
from pythonosc import dispatcher, osc_server, udp_client
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import threading
import signal
import sys
import csv
import os

import threading
alive = True
file_lock = threading.Lock()
server = None  # 提前声明，后面会赋值


# ==== CONFIG ====
ip = "0.0.0.0"
receive_port = 5089
send_ip = "127.0.0.1"
send_port = 9000

# 数据窗长度（样本点）
window_size = 500
# Muse 2 原始EEG常见采样率（按你的实际值改：256 / 220 等）
sampling_rate = 256

channel_labels = ["TP9", "AF7", "AF8", "TP10"]
n_channels = 4

# 实时缓冲
eeg_data = np.zeros((n_channels, window_size), dtype=float)
time_labels = [""] * window_size

# ==== 频谱配置 ====
# 频率范围固定为 1~60 Hz（线性）
fmin, fmax = 1.0, 60.0

# 基于窗口与采样率的频率轴（中心频率，线性）
freqs_full = np.fft.rfftfreq(window_size, d=1.0 / sampling_rate)  # [0..Nyquist]
freq_mask = (freqs_full >= fmin) & (freqs_full <= fmax)
freqs = freqs_full[freq_mask]
n_freqs = len(freqs)

# 用于绘制/写入的频谱矩阵（通道×频率，单位 dB）
spectrum_db = np.zeros((n_channels, n_freqs), dtype=float)

# ==== INIT OSC SENDER ====
osc_sender = udp_client.SimpleUDPClient(send_ip, send_port)

# ==== FILE SETUP ====
start_time = datetime.now()

# 1) 原始EEG CSV
eeg_filename = start_time.strftime("EEG_%Y_%m_%d_%H-%M-%S.csv")
eeg_csv_file = open(eeg_filename, 'w', newline='')
eeg_csv_writer = csv.writer(eeg_csv_file)
eeg_csv_writer.writerow(["Timestamp"] + channel_labels)  # Header

# 2) 频谱CSV（dB）
spec_filename = start_time.strftime("EEG_SPECTRUM_%Y_%m_%d_%H-%M-%S.csv")
spec_csv_file = open(spec_filename, 'w', newline='')
spec_csv_writer = csv.writer(spec_csv_file)
spec_header = ["Timestamp", "Channel"] + [f"f_{f:.2f}Hz_dB" for f in freqs]
spec_csv_writer.writerow(spec_header)

# ==== EEG HANDLER ====
def eeg_handler(address: str, *args):
    """
    每次收到 4 通道原始EEG：
    - 写原始EEG CSV
    - 转发到 /eeg/raw
    - 更新环形缓冲
    - 计算当前窗的频谱功率（转 dB）并写入频谱CSV
    """
    global eeg_data, time_labels, spectrum_db

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f")

    # --- 写原始EEG到CSV ---
    vals = [float(args[i]) for i in range(n_channels)]
 
    with file_lock:
        if alive and eeg_csv_file and (not eeg_csv_file.closed):
            eeg_csv_writer.writerow([timestamp] + vals)

    # --- 通过OSC转发原始EEG（保留你现有功能）---
    osc_sender.send_message("/eeg/raw", vals)

    # --- 更新实时缓冲（用于时域与频域绘制）---
    for i in range(n_channels):
        eeg_data[i, :-1] = eeg_data[i, 1:]
        eeg_data[i, -1] = vals[i]
    time_labels[:-1] = time_labels[1:]
    time_labels[-1] = now.strftime("%M:%S")

    # --- 计算当前窗频谱（去均值 + 汉宁窗 + rFFT → 功率 → dB）---
    window = np.hanning(window_size)
    win_energy = np.sum(window ** 2) + 1e-12
    
    # 频段边界（只需初始化一次缓存）
    bands = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta":  (13.0, 30.0),
        "gamma": (30.0, 60.0),
    }

    if "band_masks_cached" not in globals():
        global band_masks_cached
        band_masks_cached = {name: (freqs_full >= lo) & (freqs_full <= hi)
                             for name, (lo, hi) in bands.items()}
    
    for ch in range(n_channels):
        x = eeg_data[ch]
        x = x - np.mean(x)
        xw = x * window
        X = np.fft.rfft(xw)
        P = (np.abs(X) ** 2) / win_energy             # 线性功率谱
        P_band = P[freq_mask]
        P_db = 10.0 * np.log10(P_band + 1e-12)        # 仅用于画图/CSV
        spectrum_db[ch, :] = P_db
    
        # --- 写频谱（dB）到CSV：逐通道一行 ---
        row = [timestamp, channel_labels[ch]] + P_db.tolist()
        with file_lock:
            if alive and spec_csv_file and (not spec_csv_file.closed):
                spec_csv_writer.writerow(row)

    
        # === 新增：按频段积分（线性功率），并发送到 Allo ===
        band_lin = {name: float(np.sum(P[mask])) for name, mask in band_masks_cached.items()}

        # 主频（1–60 Hz 内的最大频点）
        dom_idx = np.argmax(P[freq_mask])
        dom_freq = float(freqs[dom_idx]) if len(freqs) else 0.0

        # /eeg/bands: [ch, delta, theta, alpha, beta, gamma]
        osc_sender.send_message("/eeg/bands", [
            ch,
            band_lin["delta"],
            band_lin["theta"],
            band_lin["alpha"],
            band_lin["beta"],
            band_lin["gamma"],
        ])

        # /eeg/dominant: [ch, dom_freq]
        osc_sender.send_message("/eeg/dominant", [ch, dom_freq])
       




# ==== PLOTTING ====
# 窗口1：时域波形
fig_time, ax_time = plt.subplots()
time_lines = []
colors = ['blue', 'orange', 'green', 'red']
for i in range(n_channels):
    (line,) = ax_time.plot(eeg_data[i], label=channel_labels[i], color=colors[i])
    time_lines.append(line)
ax_time.set_ylim(-100, 1000)  # 依据你的信号范围调整
ax_time.set_xlim(0, window_size)
ax_time.set_title("Real-Time EEG (μV)")
ax_time.set_ylabel("μV")
ax_time.set_xlabel("Time")
ax_time.legend(loc="upper left")

def update_time_plot(_):
    for i in range(n_channels):
        time_lines[i].set_ydata(eeg_data[i])
    # 更新时间刻度标签
    xticks = np.linspace(0, window_size - 1, 6, dtype=int)
    ax_time.set_xticks(xticks)
    labels = []
    step = max(1, window_size // 6)
    for k in range(0, window_size, step):
        labels.append(time_labels[k])
        if len(labels) == len(xticks):
            break
    while len(labels) < len(xticks):
        labels.append("")
    ax_time.set_xticklabels(labels)
    return time_lines

ani_time = animation.FuncAnimation(fig_time, update_time_plot, interval=50, cache_frame_data=False)

# 窗口2：频谱（通道 × 频率，线性 1–60 Hz，dB 色图）
fig_spec, ax_spec = plt.subplots()

# 用 imshow + extent（线性频率轴）
# imshow 期望图像 shape: (Y, X) = (n_channels, n_freqs)
im = ax_spec.imshow(
    spectrum_db,
    aspect="auto",
    origin="lower",
    extent=[fmin, fmax, 0, n_channels],  # 线性 1–60 Hz
    cmap="RdYlGn_r"  # 低=绿，高=红
)

ax_spec.set_title("Real-Time Spectrum Power (Channel × Frequency, dB)")
ax_spec.set_xlabel("Frequency (Hz)")
ax_spec.set_ylabel("Channel")
ax_spec.set_yticks(np.arange(n_channels) + 0.5)
ax_spec.set_yticklabels(channel_labels)

cbar = plt.colorbar(im, ax=ax_spec)
cbar.set_label("Power (dB)")

def update_spec_plot(_):
    # 更新图像数据
    im.set_data(spectrum_db)
    # 自动色阶（更稳定可改为固定范围：im.set_clim(vmin=-90, vmax=10)）
    im.set_clim(vmin=np.min(spectrum_db), vmax=np.max(spectrum_db) + 1e-12)
    return [im]

ani_spec = animation.FuncAnimation(fig_spec, update_spec_plot, interval=100, cache_frame_data=False)

# ==== EXIT CLEANUP ====
def handle_exit(sig, frame):
    global alive, server
    print("\n🧠 Stopping... Saving files.")
    alive = False
    try:
        if server is not None:
            server.shutdown()     # 让 serve_forever 退出
            server.server_close()
    except Exception as e:
        print(f"Server stop error: {e}")
    with file_lock:
        try:
            if eeg_csv_file and not eeg_csv_file.closed:
                eeg_csv_file.close()
            if spec_csv_file and not spec_csv_file.closed:
                spec_csv_file.close()
        except Exception as e:
            print(f"Close file error: {e}")
    print(f"✅ Saved EEG:      {os.path.abspath(eeg_filename)}")
    print(f"✅ Saved Spectrum: {os.path.abspath(spec_filename)}")
    plt.close('all')
    sys.exit(0)


signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

# ==== SERVER START ====
if __name__ == "__main__":
    disp = dispatcher.Dispatcher()
    # Mind Monitor 默认原始EEG地址：/muse/eeg
    disp.map("/muse/eeg", eeg_handler)

    server = osc_server.ThreadingOSCUDPServer((ip, receive_port), disp)
    print(f"✅ Listening on {ip}:{receive_port}, forwarding EEG to {send_ip}:{send_port}")
    threading.Thread(target=server.serve_forever, daemon=True).start()

    # 同时显示两个窗口
    plt.show()
