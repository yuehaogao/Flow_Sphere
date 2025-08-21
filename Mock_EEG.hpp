#pragma once
#include <lo/lo.h>
#include <array>
#include <vector>
#include <string>
#include <mutex>
#include <cmath>
#include <algorithm>

class Mock_EEG {
public:
  using BandArray = std::array<float, 5>; // delta, theta, alpha, beta, gamma

  // 例：Mock_EEG eeg(4, "9000");
  inline Mock_EEG(int channels, const std::string& oscPort)
  : numChannels(channels) {
    values.assign(numChannels, 0.0f);
    dominantFreq.assign(numChannels, 0.0f);
    bands.assign(numChannels, BandArray{0,0,0,0,0});
    bandsEMA.assign(numChannels, BandArray{0,0,0,0,0});

    // 注意：typespec 传 NULL => 允许 float/double 混合
    srv = lo_server_thread_new(oscPort.c_str(), nullptr);
    lo_server_thread_add_method(srv, "/eeg/raw",      nullptr, &Mock_EEG::on_raw_eeg,     this);
    lo_server_thread_add_method(srv, "/eeg/bands",    nullptr, &Mock_EEG::on_bands,       this);
    lo_server_thread_add_method(srv, "/eeg/dominant", nullptr, &Mock_EEG::on_dominant,    this);
    lo_server_thread_start(srv);
  }

  inline ~Mock_EEG() {
    if (srv) {
      lo_server_thread_stop(srv);
      lo_server_thread_free(srv);
      srv = nullptr;
    }
  }

  // 拿到最近一帧 4 通道原始值（来自 /eeg/raw）
  inline std::vector<float> getLatestValues() {
    std::lock_guard<std::mutex> lk(mtx);
    return values;
  }

  // 拿到最近一帧 4 通道“主频”（来自 /eeg/dominant）
  inline std::vector<float> getLatestFrequencies() {
    std::lock_guard<std::mutex> lk(mtx);
    return dominantFreq;
  }

  // 拿到最近一帧 4 通道五段功率（线性能量，来自 /eeg/bands）
  inline std::vector<BandArray> getLatestBandPowers() {
    std::lock_guard<std::mutex> lk(mtx);
    return bands;
  }

private:
  int numChannels = 0;
  lo_server_thread srv = nullptr;
  std::mutex mtx;

  std::vector<float>     values;        // /eeg/raw
  std::vector<float>     dominantFreq;  // /eeg/dominant
  std::vector<BandArray> bands;         // /eeg/bands
  std::vector<BandArray> bandsEMA;

  const float emaAlpha = 0.2f;          // EMA 平滑系数

  // ---------- 回调：/eeg/raw ----------
  static int on_raw_eeg(const char* /*path*/, const char* types, lo_arg** argv,
                        int argc, lo_message /*msg*/, void* user) {
    auto* self = reinterpret_cast<Mock_EEG*>(user);
    if (!self || argc < self->numChannels) return 0;
    std::lock_guard<std::mutex> lk(self->mtx);
    for (int i = 0; i < self->numChannels; ++i) {
      if (types[i] == 'f')      self->values[i] = argv[i]->f;
      else if (types[i] == 'd') self->values[i] = static_cast<float>(argv[i]->d);
    }
    return 0;
  }

  // ---------- 回调：/eeg/bands ----------
  static int on_bands(const char* /*path*/, const char* types, lo_arg** argv,
                      int argc, lo_message /*msg*/, void* user) {
    auto* self = reinterpret_cast<Mock_EEG*>(user);
    if (!self || argc < 6) return 0; // 期望: ch + 5 段
    int ch = 0;
    if (types[0] == 'i')      ch = argv[0]->i;
    else if (types[0] == 'f') ch = static_cast<int>(argv[0]->f);
    if (ch < 0 || ch >= self->numChannels) return 0;

    float v[5]{};
    for (int k = 0; k < 5; ++k) {
      if (types[1 + k] == 'f')      v[k] = argv[1 + k]->f;
      else if (types[1 + k] == 'd') v[k] = static_cast<float>(argv[1 + k]->d);
    }

    std::lock_guard<std::mutex> lk(self->mtx);
    for (int k = 0; k < 5; ++k) {
      self->bandsEMA[ch][k] = self->emaAlpha * v[k] + (1.0f - self->emaAlpha) * self->bandsEMA[ch][k];
      self->bands[ch][k]    = self->bandsEMA[ch][k];
    }
    return 0;
  }

  // ---------- 回调：/eeg/dominant ----------
  static int on_dominant(const char* /*path*/, const char* types, lo_arg** argv,
                         int argc, lo_message /*msg*/, void* user) {
    auto* self = reinterpret_cast<Mock_EEG*>(user);
    if (!self || argc < 2) return 0;
    int ch = 0;
    if (types[0] == 'i')      ch = argv[0]->i;
    else if (types[0] == 'f') ch = static_cast<int>(argv[0]->f);
    float f = 0.0f;
    if (types[1] == 'f')      f = argv[1]->f;
    else if (types[1] == 'd') f = static_cast<float>(argv[1]->d);
    if (ch < 0 || ch >= self->numChannels) return 0;

    std::lock_guard<std::mutex> lk(self->mtx);
    self->dominantFreq[ch] = f;
    return 0;
  }
};
