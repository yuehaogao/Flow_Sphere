// Mock_EEG.cpp - 0827

// — single-file, header-free version
// Drop-in: include this file directly from Flow_Sphere.cpp:
//     #include "Mock_EEG.cpp"
//
// Notes:
// - Do NOT also compile/link this file separately if you include it like above.
// - Constructor: Mock_EEG(int channels, const std::string& oscPort)
//   Expects read_mindmonitor.py to be sending to that port (e.g., "9000").
// - Endpoints received:
//   /eeg/raw      -> payload: 4 floats (one per channel)  [you can send any #channels]
//   /eeg/bands    -> payload: int ch, 5 floats (delta..gamma) (linear power)
//   /eeg/dominant -> payload: int ch, float dom_freq
//
// Thread-safe getters:
//   getLatestValues()       -> std::vector<float> size=channels
//   getLatestFrequencies()  -> std::vector<float> size=channels
//   getLatestBandPowers()   -> std::vector<std::array<float,5>> size=channels

#pragma once

#include <lo/lo.h>
#include <vector>
#include <array>
#include <string>
#include <mutex>
#include <cmath>
#include <cstring>
#include <iostream>

class Mock_EEG {
public:
  using BandArray = std::array<float, 5>; // delta, theta, alpha, beta, gamma

  Mock_EEG(int channels, const std::string& oscPort)
  : mNumChannels(channels),
    mSrv(nullptr),
    mValues(channels, 0.0f),
    mDominantFreq(channels, 0.0f),
    mBands(channels, BandArray{0,0,0,0,0}),
    mBandsEMA(channels, BandArray{0,0,0,0,0})
  {
    // Create liblo server thread on the given port.
    mSrv = lo_server_thread_new(oscPort.c_str(), nullptr);
    if (!mSrv) {
      std::cerr << "[Mock_EEG] Failed to open OSC port " << oscPort << std::endl;
      return;
    }
    // Register handlers (typespec = NULL to accept float/double mix, etc.)
    lo_server_thread_add_method(mSrv, "/eeg/raw",      NULL, &Mock_EEG::on_raw_static,      this);
    lo_server_thread_add_method(mSrv, "/eeg/bands",    NULL, &Mock_EEG::on_bands_static,    this);
    lo_server_thread_add_method(mSrv, "/eeg/dominant", NULL, &Mock_EEG::on_dominant_static, this);
    lo_server_thread_start(mSrv);
  }

  ~Mock_EEG() {
    if (mSrv) {
      lo_server_thread_stop(mSrv);
      lo_server_thread_free(mSrv);
      mSrv = nullptr;
    }
  }

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

private:
  int mNumChannels;
  lo_server_thread mSrv;
  std::mutex mMtx;
  std::vector<float>     mValues;       // latest raw value per channel (last sample)
  std::vector<float>     mDominantFreq; // latest dominant freq per channel
  std::vector<BandArray> mBands;        // latest band powers (linear) per channel
  std::vector<BandArray> mBandsEMA;     // EMA smoother for band powers

  static constexpr float kEmaAlpha = 0.2f; // simple smoothing for colors etc.

  // ---- Static trampolines required by liblo C callbacks ----
  static int on_raw_static(const char* path, const char* types, lo_arg** argv,
                           int argc, lo_message msg, void* user)
  {
    return reinterpret_cast<Mock_EEG*>(user)->on_raw(path, types, argv, argc, msg);
  }

  static int on_bands_static(const char* path, const char* types, lo_arg** argv,
                             int argc, lo_message msg, void* user)
  {
    return reinterpret_cast<Mock_EEG*>(user)->on_bands(path, types, argv, argc, msg);
  }

  static int on_dominant_static(const char* path, const char* types, lo_arg** argv,
                                int argc, lo_message msg, void* user)
  {
    return reinterpret_cast<Mock_EEG*>(user)->on_dominant(path, types, argv, argc, msg);
  }

  // ---- Instance handlers ----
  int on_raw(const char* /*path*/, const char* types, lo_arg** argv,
             int argc, lo_message /*msg*/)
  {
    // Accept at least mNumChannels elements; extra are ignored.
    if (argc < mNumChannels) return 0;
    std::lock_guard<std::mutex> lk(mMtx);
    for (int i = 0; i < mNumChannels; ++i) {
      if      (types[i] == 'f') mValues[i] = argv[i]->f;
      else if (types[i] == 'd') mValues[i] = static_cast<float>(argv[i]->d);
      else if (types[i] == 'i') mValues[i] = static_cast<float>(argv[i]->i);
    }
    return 0;
  }

  int on_bands(const char* /*path*/, const char* types, lo_arg** argv,
               int argc, lo_message /*msg*/)
  {
    // Expect: ch + 5 band floats (delta..gamma)
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

  int on_dominant(const char* /*path*/, const char* types, lo_arg** argv,
                  int argc, lo_message /*msg*/)
  {
    if (argc < 2) return 0; // ch, f
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
};
