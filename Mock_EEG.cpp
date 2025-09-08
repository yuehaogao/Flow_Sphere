// Mock_EEG.cpp - With Granular | live OSC + CSV playback (EEG / Bands / Dominant / Pulse / BPM)
// 2025-09-06

#pragma once

#include <lo/lo.h>
#include <vector>
#include <array>
#include <string>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <chrono>
#include <unordered_map>

#include <unistd.h>
#include <limits.h>

class Mock_EEG {
public:
  using BandArray = std::array<float, 5>; // delta, theta, alpha, beta, gamma

  // ====== 可调默认：当不传入 stamp 时使用它 ======
  static constexpr const char* kDefaultStamp = "2025_08_28_15-27-52";

  Mock_EEG(int channels, const std::string& oscPort)
  : mNumChannels(channels),
    mSrv(nullptr),
    mValues(channels, 0.0f),
    mDominantFreq(channels, 0.0f),
    mBands(channels, BandArray{0,0,0,0,0}),
    mBandsEMA(channels, BandArray{0,0,0,0,0}),
    mPlayback(false),
    mStopPlayback(false)
  {
    // —— 实时 OSC 服务器（一直开着）——
    mSrv = lo_server_thread_new(oscPort.c_str(), nullptr);
    if (!mSrv) {
      std::cerr << "[Mock_EEG] Failed to open OSC port " << oscPort << std::endl;
    } else {
      lo_server_thread_add_method(mSrv, "/eeg/raw",      NULL, &Mock_EEG::on_raw_static,      this);
      lo_server_thread_add_method(mSrv, "/eeg/bands",    NULL, &Mock_EEG::on_bands_static,    this);
      lo_server_thread_add_method(mSrv, "/eeg/dominant", NULL, &Mock_EEG::on_dominant_static, this);
      lo_server_thread_add_method(mSrv, "/pulse/raw",    NULL, &Mock_EEG::on_pulse_raw_static, this);
      lo_server_thread_add_method(mSrv, "/pulse/bpm",    NULL, &Mock_EEG::on_pulse_bpm_static, this);
      lo_server_thread_start(mSrv);
    }
  }

  ~Mock_EEG() {
    disablePlayback();
    if (mSrv) {
      lo_server_thread_stop(mSrv);
      lo_server_thread_free(mSrv);
      mSrv = nullptr;
    }
  }

  // ===================== Live getters（两种模式统一访问） =====================
  std::vector<float> getLatestValues() {
    std::lock_guard<std::mutex> lk(mMtx);
    return mValues;
  }
  std::vector<float> getLatestFrequencies() {
    std::lock_guard<std::mutex> lk(mMtx);
    return mDominantFreq;
  }
  std::vector<BandArray> getLatestBandPowers() {
    std::lock_guard<std::mutex> lk(mMtx);
    return mBands;
  }
  float getLatestPulse() {
    std::lock_guard<std::mutex> lk(mMtx);
    return mPulseRaw;
  }
  float getLatestBPM() {
    std::lock_guard<std::mutex> lk(mMtx);
    return mBPM;
  }

  // ===================== Playback 控制 =====================
  bool enablePlayback(const std::string& stamp) {
    disablePlayback(); // 先停掉旧线程
    mPlayback = true;
    mStopPlayback = false;
    mStamp = stamp.empty() ? std::string(kDefaultStamp) : stamp;

    // 预加载 CSV（若任何一份失败，回退实时模式）
    if (!loadAllCsv()) {
      std::cerr << "[Mock_EEG] Playback load failed. Reverting to live mode.\n";
      mPlayback = false;
      mStopPlayback = true;
      return false;
    }

    mPlaybackThread = std::thread([this]() { this->playbackLoop(); });
    return true;
  }

  bool enablePlayback() { return enablePlayback(kDefaultStamp); }

  void disablePlayback() {
    if (mPlaybackThread.joinable()) {
      mStopPlayback = true;
      {
        std::lock_guard<std::mutex> lk(mCvMtx);
        mCv.notify_all();
      }
      mPlaybackThread.join();
    }
    mPlayback = false;
    mStopPlayback = false;
  }

  bool isPlaybackEnabled() const { return mPlayback; }

private:
  // ===================== 实时模式（OSC） =====================
  int mNumChannels;
  lo_server_thread mSrv;
  std::mutex mMtx;

  std::vector<float>     mValues;       // last EEG sample per channel
  std::vector<float>     mDominantFreq; // last dominant frequency per channel
  std::vector<BandArray> mBands;        // band powers (linear)
  std::vector<BandArray> mBandsEMA;     // simple EMA for stabler visuals

  float mPulseRaw = 0.0f;
  float mBPM      = 0.0f;

  static constexpr float kEmaAlpha = 0.2f;

  static int on_raw_static(const char* path, const char* types, lo_arg** argv,
                           int argc, lo_message msg, void* user)
  { return reinterpret_cast<Mock_EEG*>(user)->on_raw(path, types, argv, argc, msg); }
  static int on_bands_static(const char* path, const char* types, lo_arg** argv,
                             int argc, lo_message msg, void* user)
  { return reinterpret_cast<Mock_EEG*>(user)->on_bands(path, types, argv, argc, msg); }
  static int on_dominant_static(const char* path, const char* types, lo_arg** argv,
                                int argc, lo_message msg, void* user)
  { return reinterpret_cast<Mock_EEG*>(user)->on_dominant(path, types, argv, argc, msg); }
  static int on_pulse_raw_static(const char* path, const char* types, lo_arg** argv,
                                 int argc, lo_message msg, void* user)
  { return reinterpret_cast<Mock_EEG*>(user)->on_pulse_raw(path, types, argv, argc, msg); }
  static int on_pulse_bpm_static(const char* path, const char* types, lo_arg** argv,
                                 int argc, lo_message msg, void* user)
  { return reinterpret_cast<Mock_EEG*>(user)->on_pulse_bpm(path, types, argv, argc, msg); }

  int on_raw(const char* /*path*/, const char* types, lo_arg** argv, int argc, lo_message /*msg*/) {
    if (argc < mNumChannels) return 0;
    std::lock_guard<std::mutex> lk(mMtx);
    for (int i = 0; i < mNumChannels; ++i) {
      if      (types[i] == 'f') mValues[i] = argv[i]->f;
      else if (types[i] == 'd') mValues[i] = static_cast<float>(argv[i]->d);
      else if (types[i] == 'i') mValues[i] = static_cast<float>(argv[i]->i);
    }
    return 0;
  }
  int on_bands(const char* /*path*/, const char* types, lo_arg** argv, int argc, lo_message /*msg*/) {
    if (argc < 6) return 0;
    int ch = 0;
    if      (types[0] == 'i') ch = argv[0]->i;
    else if (types[0] == 'f') ch = static_cast<int>(argv[0]->f);
    else if (types[0] == 'd') ch = static_cast<int>(argv[0]->d);
    if (ch < 0 || ch >= mNumChannels) return 0;

    float v[5]{};
    for (int k = 0; k < 5; ++k) {
      char t = types[1 + k];
      if      (t == 'f') v[k] = argv[1 + k]->f;
      else if (t == 'd') v[k] = static_cast<float>(argv[1 + k]->d);
      else if (t == 'i') v[k] = static_cast<float>(argv[1 + k]->i);
    }
    std::lock_guard<std::mutex> lk(mMtx);
    for (int k = 0; k < 5; ++k) {
      mBandsEMA[ch][k] = kEmaAlpha * v[k] + (1.0f - kEmaAlpha) * mBandsEMA[ch][k];
      mBands[ch][k]    = mBandsEMA[ch][k];
    }
    return 0;
  }
  int on_dominant(const char* /*path*/, const char* types, lo_arg** argv, int argc, lo_message /*msg*/) {
    if (argc < 2) return 0;
    int ch = 0;
    if      (types[0] == 'i') ch = argv[0]->i;
    else if (types[0] == 'f') ch = static_cast<int>(argv[0]->f);
    else if (types[0] == 'd') ch = static_cast<int>(argv[0]->d);
    float f = 0.0f;
    if      (types[1] == 'f') f = argv[1]->f;
    else if (types[1] == 'd') f = static_cast<float>(argv[1]->d);
    else if (types[1] == 'i') f = static_cast<float>(argv[1]->i);
    if (ch < 0 || ch >= mNumChannels) return 0;
    std::lock_guard<std::mutex> lk(mMtx);
    mDominantFreq[ch] = f;
    return 0;
  }
  int on_pulse_raw(const char* /*path*/, const char* types, lo_arg** argv, int argc, lo_message /*msg*/) {
    if (argc < 1) return 0;
    float v = 0.0f;
    if      (types[0] == 'f') v = argv[0]->f;
    else if (types[0] == 'd') v = static_cast<float>(argv[0]->d);
    else if (types[0] == 'i') v = static_cast<float>(argv[0]->i);
    std::lock_guard<std::mutex> lk(mMtx);
    mPulseRaw = v;
    return 0;
  }
  int on_pulse_bpm(const char* /*path*/, const char* types, lo_arg** argv, int argc, lo_message /*msg*/) {
    if (argc < 1) return 0;
    float v = 0.0f;
    if      (types[0] == 'f') v = argv[0]->f;
    else if (types[0] == 'd') v = static_cast<float>(argv[0]->d);
    else if (types[0] == 'i') v = static_cast<float>(argv[0]->i);
    std::lock_guard<std::mutex> lk(mMtx);
    mBPM = v;
    return 0;
  }

  // ===================== 读取模式（CSV 播放） =====================
  struct EEGRow { double t; std::vector<float> vals; };           // t: 秒（相对首帧）
  struct PulseRow { double t; float pulse; float bpm; };
  struct SpecRow  { double t; int ch; std::vector<float> db; };   // 每行一个channel的dB谱

  // 预加载容器
  std::vector<EEGRow>   mEEG;
  std::vector<PulseRow> mPULSE;
  std::vector<SpecRow>  mSPEC;

  // 频谱头（Hz 列表）与频段掩码
  std::vector<double> mFreqsHz;
  std::array<std::vector<int>,5> mBandIdx; // Δ/Θ/Α/Β/Γ 在 mFreqsHz 的下标集合

  // 播放控制
  std::atomic<bool> mPlayback;
  std::atomic<bool> mStopPlayback;
  std::thread       mPlaybackThread;
  std::string       mStamp;

  std::mutex mCvMtx;
  std::condition_variable mCv;

  // —— 小工具：解析 “YYYY-.. HH:MM:SS.micro” 的时间成“秒（只用时分秒）” ——
  static double parseHMSsec(const std::string& ts) {
    // 提取最后一个空格后的 "HH:MM:SS.micro"
    auto pos = ts.find_last_of(' ');
    std::string hms = (pos == std::string::npos) ? ts : ts.substr(pos + 1);
    int h=0, m=0; double s=0.0;
    // 支持带小数秒
    // 例如 12:34:56.123456
    char dummy;
    std::istringstream iss(hms);
    iss >> h >> dummy >> m >> dummy >> s;
    return h*3600.0 + m*60.0 + s;
  }

  static bool splitCSVLine(const std::string& line, std::vector<std::string>& out) {
    out.clear();
    std::string cur; bool inq=false;
    for (char c: line) {
      if (c=='"') { inq=!inq; continue; }
      if (c==',' && !inq) { out.push_back(cur); cur.clear(); }
      else cur.push_back(c);
    }
    out.push_back(cur);
    return true;
  }

  static int channelIndexFromName(const std::string& name) {
    if (name == "TP9")  return 0;
    if (name == "AF7")  return 1;
    if (name == "AF8")  return 2;
    if (name == "TP10") return 3;
    // 兜底：尝试数字
    try { return std::stoi(name); } catch (...) { return -1; }
  }

  static bool fileExists(const std::string& path) {
    std::ifstream f(path); return f.good();
  }

  bool loadAllCsv() {

    char buf[PATH_MAX];
    if (getcwd(buf, sizeof(buf))) {
        std::cerr << "[Mock_EEG] Current working directory: " << buf << std::endl;
    }

    const std::string eeg = "EEG_" + mStamp + ".csv";
    const std::string pul = "EEG_PULSE_" + mStamp + ".csv";
    const std::string spc = "EEG_SPECTRUM_" + mStamp + ".csv";


    if (!fileExists(eeg) || !fileExists(pul) || !fileExists(spc)) {
      std::cerr << "[Mock_EEG] Missing CSV files for stamp " << mStamp << "\n";
      std::cerr << "  -> " << eeg << "\n  -> " << pul << "\n  -> " << spc << "\n";
      return false;
    }

    return loadEEG(eeg) && loadPULSE(pul) && loadSPEC(spc);
  }

  bool loadEEG(const std::string& path) {
    mEEG.clear();
    std::ifstream fin(path);
    if (!fin.is_open()) return false;

    std::string line;
    if (!std::getline(fin, line)) return false; // header

    // 以首行时间为零点
    double t0 = -1.0;
    while (std::getline(fin, line)) {
      if (line.empty()) continue;
      std::vector<std::string> cols; splitCSVLine(line, cols);
      if (cols.size() < 1 + mNumChannels) continue;
      double t = parseHMSsec(cols[0]);
      if (t0 < 0) t0 = t;
      EEGRow row; row.t = t - t0;
      row.vals.resize(mNumChannels, 0.0f);
      for (int i = 0; i < mNumChannels; ++i) {
        try { row.vals[i] = std::stof(cols[1+i]); } catch (...) { row.vals[i] = 0.0f; }
      }
      mEEG.push_back(std::move(row));
    }
    return !mEEG.empty();
  }

  bool loadPULSE(const std::string& path) {
    mPULSE.clear();
    std::ifstream fin(path);
    if (!fin.is_open()) return false;
    std::string line;
    if (!std::getline(fin, line)) return false; // header

    double t0 = -1.0;
    while (std::getline(fin, line)) {
      if (line.empty()) continue;
      std::vector<std::string> cols; splitCSVLine(line, cols);
      if (cols.size() < 3) continue;
      double t = parseHMSsec(cols[0]);
      if (t0 < 0) t0 = t;
      PulseRow r; r.t = t - t0;
      try { r.pulse = std::stof(cols[1]); } catch (...) { r.pulse = 0.0f; }
      try { r.bpm   = std::stof(cols[2]); } catch (...) { r.bpm   = 0.0f; }
      mPULSE.push_back(std::move(r));
    }
    return true; // 脉搏为空也不致命
  }

  bool loadSPEC(const std::string& path) {
    mSPEC.clear();
    mFreqsHz.clear();
    for (auto& v: mBandIdx) v.clear();

    std::ifstream fin(path);
    if (!fin.is_open()) return false;

    std::string header;
    if (!std::getline(fin, header)) return false;
    // header: "Timestamp,Channel,f_1.00Hz_dB,f_1.12Hz_dB,..."
    {
      std::vector<std::string> cols; splitCSVLine(header, cols);
      if (cols.size() < 3) return false;
      // 解析频率列
      for (size_t i = 2; i < cols.size(); ++i) {
        // 去掉前缀 f_ 和后缀 Hz_dB
        std::string s = cols[i];
        // 可能是 "f_8.00Hz_dB"
        size_t p1 = s.find('_');
        size_t p2 = s.find("Hz");
        if (p1 != std::string::npos && p2 != std::string::npos && p2 > p1+1) {
          std::string num = s.substr(p1+1, p2 - (p1+1));
          try { mFreqsHz.push_back(std::stod(num)); } catch (...) { mFreqsHz.push_back(0.0); }
        } else {
          mFreqsHz.push_back(0.0);
        }
      }
      // 预生成频段掩码（索引）
      auto mask = [&](double lo, double hi) {
        std::vector<int> idx;
        for (int i = 0; i < (int)mFreqsHz.size(); ++i) {
          double f = mFreqsHz[i];
          if (f >= lo && f <= hi) idx.push_back(i);
        }
        return idx;
      };
      mBandIdx[0] = mask(1.0, 4.0);   // Δ
      mBandIdx[1] = mask(4.0, 8.0);   // Θ
      mBandIdx[2] = mask(8.0, 13.0);  // Α
      mBandIdx[3] = mask(13.0, 30.0); // Β
      mBandIdx[4] = mask(30.0, 60.0); // Γ
    }

    // 读取每行 —— 每行一个 channel 的谱
    std::string line;
    double t0 = -1.0;
    while (std::getline(fin, line)) {
      if (line.empty()) continue;
      std::vector<std::string> cols; splitCSVLine(line, cols);
      if (cols.size() < 3) continue;

      double t = parseHMSsec(cols[0]);
      if (t0 < 0) t0 = t;

      int ch = channelIndexFromName(cols[1]);
      if (ch < 0 || ch >= mNumChannels) continue;

      SpecRow r; r.t = t - t0; r.ch = ch;
      r.db.reserve(mFreqsHz.size());
      for (size_t i = 2; i < cols.size(); ++i) {
        try { r.db.push_back(std::stof(cols[i])); } catch (...) { r.db.push_back(-120.0f); }
      }
      mSPEC.push_back(std::move(r));
    }
    return !mSPEC.empty();
  }

  // —— 把 dB 光谱转为（Δ..Γ）线性功率和主频 ——
  void spectrumToBandsAndDominant(const std::vector<float>& db,
                                  BandArray& bands_out, float& domFreq_out) {
    // dB -> linear power
    const int N = (int)db.size();
    if (N == 0 || (int)mFreqsHz.size() != N) {
      bands_out = BandArray{0,0,0,0,0};
      domFreq_out = 0.0f;
      return;
    }
    // 找主频（最大线性功率）
    int argmax = 0;
    double maxP = -1.0;
    // 先把线性谱缓存（避免多次 pow）
    static thread_local std::vector<double> P;
    P.resize(N);
    for (int i = 0; i < N; ++i) {
      // dB 可能极小，保底
      double p = std::pow(10.0, db[i] / 10.0);
      P[i] = p;
      if (p > maxP) { maxP = p; argmax = i; }
    }
    domFreq_out = (float)mFreqsHz[argmax];

    // 每个频段积分
    for (int b = 0; b < 5; ++b) {
      double sum = 0.0;
      for (int idx : mBandIdx[b]) {
        if (idx >= 0 && idx < N) sum += P[idx];
      }
      bands_out[b] = (float)sum;
    }
  }

  void playbackLoop() {
    if (mEEG.empty() && mPULSE.empty() && mSPEC.empty()) {
      std::cerr << "[Mock_EEG] Nothing to play.\n";
      return;
    }

    // 各流的索引
    size_t ie = 0, ip = 0, is = 0;

    auto tStart = std::chrono::steady_clock::now();

    while (!mStopPlayback.load()) {
      auto now = std::chrono::steady_clock::now();
      double t = std::chrono::duration<double>(now - tStart).count(); // 已经播放的秒数

      // —— EEG：推进至所有 r.t <= t 的行 ——（逐行覆盖）
      while (ie < mEEG.size() && mEEG[ie].t <= t) {
        const auto& r = mEEG[ie];
        {
          std::lock_guard<std::mutex> lk(mMtx);
          for (int i = 0; i < mNumChannels && i < (int)r.vals.size(); ++i) {
            mValues[i] = r.vals[i];
          }
        }
        ++ie;
      }

      // —— PULSE：推进至 r.t <= t ——（覆盖脉搏与 BPM）
      while (ip < mPULSE.size() && mPULSE[ip].t <= t) {
        const auto& r = mPULSE[ip];
        {
          std::lock_guard<std::mutex> lk(mMtx);
          mPulseRaw = r.pulse;
          mBPM      = r.bpm;
        }
        ++ip;
      }

      // —— SPECTRUM：每行一个通道 ——（覆盖该通道的 bands 与 domHz）
      while (is < mSPEC.size() && mSPEC[is].t <= t) {
        const auto& r = mSPEC[is];
        BandArray b{}; float dom=0.0f;
        spectrumToBandsAndDominant(r.db, b, dom);
        {
          std::lock_guard<std::mutex> lk(mMtx);
          // EMA 平滑，保持与实时模式一致的观感
          for (int k = 0; k < 5; ++k) {
            mBandsEMA[r.ch][k] = kEmaAlpha * b[k] + (1.0f - kEmaAlpha) * mBandsEMA[r.ch][k];
            mBands[r.ch][k]    = mBandsEMA[r.ch][k];
          }
          mDominantFreq[r.ch] = dom;
        }
        ++is;
      }

      // 全部结束则退出
      if (ie >= mEEG.size() && ip >= mPULSE.size() && is >= mSPEC.size()) {
        break;
      }

      // 轻睡眠以降低 CPU；也支持提前唤醒
      std::unique_lock<std::mutex> lk(mCvMtx);
      mCv.wait_for(lk, std::chrono::milliseconds(2), [this]{ return mStopPlayback.load(); });
    }

    // 播放结束：保持最后一帧在缓存里，静默退出
  }
};
