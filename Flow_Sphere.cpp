

// Flow_Sphere.cpp | Yuehao Gao, 2025 - 0906
// Ribbon mesh implemented
// Granular Synthesis V1 implemented
// problem: grains neve stop


// Synthesis designed on Myungin Lee(2022) Sine Envelope with Visuals
// Inspired by:
// https://www.media.mit.edu/projects/flower-eeg-visualization-with-the-aid-of-machine-learning/overview/


/*
 * Purpose:
 
 * This file implements the main functionalities for the "Flow Sphere" project, 
 * an immersive EEG visualization system designed for the UCSB Allosphere. The 
 * goal is to analyze and visually represent group flow experiences during musical 
 * performances using real-time or recorded EEG data.

 * Background:
 * The concept of "Flow Experiences," as introduced by Mihaly Csikszentmihalyi, 
 * describes a state of optimal engagement, focus, and creativity that occurs 
 * when individuals are fully immersed in an activity. While flow is often studied 
 * at the individual level, "Group Flow" occurs when multiple participants achieve 
 * synchronization and heightened collaboration, such as in musical performances.

 * Significance:
 * Understanding group flow during music-making offers profound insights into 
 * human collaboration, creativity, and emotional well-being. By visualizing EEG 
 * signals of multiple participants, this project aims to:
 *   - Identify neural patterns associated with group flow states.
 *   - Foster a deeper understanding of how music facilitates cognitive and emotional 
 *     synchronization among individuals.
 *   - Provide participants with feedback on their neural activity to enhance 
 *     personal and group-level musical experiences.

 * Contributions:
 * This project will generate 3D visualizations of EEG data within the Allosphere, 
 * where each participant's neural activity is represented as a dynamic, interactive 
 * object (e.g., spinning flower shapes). Spectral components such as theta and 
 * alpha waves, which are critical markers for flow states, are color-coded for 
 * intuitive analysis. These visualizations will:
 *   - Help participants and lab members study group neural synchronization.
 *   - Provide a platform for real-time feedback and experimental exploration.
 *   - Contribute to the broader understanding of how music fosters connectivity 
 *     and mental well-being.

 * This main.cpp file includes the core logic for:
 *   - Reading EEG signals from input files or live streams.
 *   - Processing and mapping EEG data to visual elements.
 *   - Rendering interactive 3D visualizations in the Allosphere environment.
 */

// ----------------------------------------------------------------
// Press '=' to enable/disable navigation
// Press '[' or ']' to turn on & off GUI 
// ----------------------------------------------------------------


#include <cmath>
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <time.h> 
#include <vector>
#include <deque>
#include <unordered_map>
#include <array>
#include <algorithm>
#include <string>
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif
#define PI 3.1415926535

//#include "al/app/al_App.hpp"

#include "Mock_EEG.cpp" 

#include "al/app/al_DistributedApp.hpp"
#include "al/app/al_GUIDomain.hpp"
#include "al_ext/statedistribution/al_CuttleboneDomain.hpp"
#include "al_ext/statedistribution/al_CuttleboneStateSimulationDomain.hpp"
#include "al/graphics/al_Shapes.hpp"
#include "al/io/al_MIDI.hpp"
#include "al/math/al_Functions.hpp"
#include "al/scene/al_PolySynth.hpp"
#include "al/scene/al_SynthSequencer.hpp"
#include "al/ui/al_ControlGUI.hpp"
#include "al/ui/al_Parameter.hpp"
#include "Gamma/Analysis.h"
#include "Gamma/Effects.h"
#include "Gamma/Envelope.h"
#include "Gamma/Oscillator.h"
#include "Gamma/DFT.h"
#include "Gamma/rnd.h"
#include "Gamma/SoundFile.h"
#include "Gamma/Filter.h"   // (optional) for a simple HPF if you want


using namespace gam;
using namespace al;
using namespace std;

// ===== 在你现有的宏附近，替换为 =====
#define FFT_SIZE 4096
#define HOP_SIZE (FFT_SIZE/4)   // 典型 25% hop，平衡时频分辨率



const int NUM_FLOWERS = 1;                        // How many flowers are there, meaning how many participants are observed
const int NUM_CHANNELS = 4;                      // How many EEG channels are there for one participant
const int WAVE_BUFFER_LENGTH = 600;               // The length of the buffer storing the wave values
const float DENSITY = 0.2;                        // How dense the samples are on each channel
const float REFRESH_ANGLE = PI * -0.5;            // The refreshing point in the circle
                                                  // ** CHANGE TO VARIABLE LATER
const float CENTRAL_RADIUS = 10.0;                // The radius of the middle channel of each EEG Flower Mesh
const float CHANNEL_DISTANCE = 10.0 / NUM_CHANNELS;          
                                                  // The distance between each channel
                                                  // ** CHANGE TO VARIABLE LATER
const float OSC_AMP = 0.3 * CHANNEL_DISTANCE;     // The oscilation amplitude of each channel



// ===== Silky Ribbon parameters =====
const int   RIBBON_SLICES     = 8000;    // 可见的切片数量（长度）
const int   RIBBON_WIDTH      = 64;      // 截面采样点数（越大越丝滑）
const float RIBBON_THICKNESS  = 2.0f;    // 丝绸“带宽”（半幅）
const float SPIRAL_RADIUS     = 12.0f;   // 围绕“廊桥”的半径（与你Processing一致）
const float SPIRAL_SPEED      = 0.02f;   // 围绕中心的旋转速度
const float ADVANCE_PER_FRAME = 0.05f;   // 沿 z 轴缓缓前进速度（Processing 里≈2）
const float LIFT_GAIN   = 3.80f;         // 出平面（z）位移强度：越大上下起伏越明显
const float TWIST_GAIN  = 0.55f;         // 切向（tHat）扭拧强度：越大越像“拧毛巾”
const float LIFT_SIGMA  = 0.22f;         // 4 通道高斯核的宽度（0~0.5），越小越“分区”，越大越“融合”
const float RIBBON_THICKNESS_3D = 0.60f; // 体厚（单位世界坐标），可 0.4~1.2 调

// 三基色锚：暗红(低频 delta)、青蓝(中频 theta/alpha/低beta=心流)、暗紫(高频 gamma)
const float HUE_LOW   = 0.96f;  // red
const float HUE_MID   = 0.53f;  // cyan-blue
const float HUE_HIGH  = 0.78f;  // purple

struct RibbonSlice {
  std::vector<Vec3f> v;   // 截面顶点（沿宽度方向）
  std::vector<Color> c;   // 对应颜色（含 alpha）
  float z0 = 0.0f;        // 该截面生成时的 z 基准，用于做“整体前推”滚动
};

std::deque<RibbonSlice> gRibbon;  // 环形队列，持续推进，超长则丢尾
Mesh gRibbonMesh;                 // 真正绘制的三角形网格
Mesh Flowers;
float gScrollZ = 0.0f;            // 全局“相机相对滚动”，实现向前推进的错觉


// --- Stars (pulse-reactive) -----------------------------
const int   STAR_MAX       = 200;   // 上限，仅用于 state 数组大小
const int   STAR_COUNT     = 180;    // 实际生成的星星数量（可调 300~1500）
const float STAR_RMIN      = 20.0f;  // 星场内半径（根据你的场景单位调）
const float STAR_RMAX      = 22.0f;  // 星场外半径
const float STAR_PULSE_AMP = 3.2f;   // 心跳引起的半径摆动幅度（单位=世界坐标）
//const float STAR_SIZE_PX   = 1.26f;   // 点精灵大小（point shader 的 pointSize）
const float STAR_SIZE_WORLD = 0.1f;   // 小四边形半边长（世界单位）。想更大就 2.5/3.5


// ** CHANGE TO VARIABLE LATER
const float BASE_RADIUS = CENTRAL_RADIUS - (0.5 * CHANNEL_DISTANCE * NUM_FLOWERS);               
                                                  // The radius of the inner-most channel
const float FLOWER_DIST = 70.0;                   // The distance of each flower mesh to the origin
                                                  // ** CHANGE TO VARIABLE LATER
//const float MAX_HUE = 0.5;                        // The range of colors, which is set to:
                                                  // Dark red: furthest to central Lower Betta (16Hz)   
                                                  // Light blue: closest to central Lower Betta (16Hz)                                                              
//const float HUE_CONTRAST = 1.5;                   // How obvious are the color contrast between non-flow and flow frequencies
//const float MIN_BRIGHNESS = 0.3;                  // The lowest brightness of frequencies furthest to central lower Betta
const float FLOWER_DYNAMIC = 0.01;               // The "thickness" of the flower
//const float CENTRAL_LOWER_BETTA = 16.0;           // As "lower Betta waves" are mostly correlated with focused flow states
                                                  // Frequencies 12-20Hz are considered "music-induced flow"



// This example shows how to use SynthVoice and SynthManagerto create an audio
// visual synthesizer. In a class that inherits from SynthVoice you will
// define the synth's voice parameters and the sound and graphic generation
// processes in the onProcess() functions.
class SineEnv : public SynthVoice
{
public:
  // Unit generators
  gam::Pan<> mPan;
  gam::Sine<> mOsc;
  gam::Env<3> mAmpEnv;
  // envelope follower to connect audio output to graphics
  gam::EnvFollow<> mEnvFollow;
  // Draw parameters
  Mesh mMesh;
  double a;
  double b;

  double spin = gam::rnd::uniS(1.0);

  double timepose = 0;
  Vec3f note_position;
  Vec3f note_direction;

  // Additional members
  // Initialize voice. This function will only be called once per voice when
  // it is created. Voices will be reused if they are idle.
  void init() override
  {
    // Intialize envelope
    mAmpEnv.curve(0); // make segments lines
    mAmpEnv.levels(0, 1, 1, 0);
    mAmpEnv.sustainPoint(2); // Make point 2 sustain until a release is issued

    // We have the mesh be a sphere
    addSphere(mMesh, 0.3, 50, 50);
    mMesh.decompress();
    mMesh.generateNormals();

    createInternalTriggerParameter("amplitude", 0.16, 0.0, 0.5);
    createInternalTriggerParameter("frequency", 60, 20, 5000);
    createInternalTriggerParameter("attackTime", 0.66, 0.01, 3.0);
    createInternalTriggerParameter("releaseTime", 1.5, 0.1, 10.0);
    createInternalTriggerParameter("pan", 0.0, -1.0, 1.0);

    // Initalize MIDI device input
  }


  // The audio processing function
  void onProcess(AudioIOData &io) override
  {
    // Get the values from the parameters and apply them to the corresponding
    // unit generators. You could place these lines in the onTrigger() function,
    // but placing them here allows for realtime prototyping on a running
    // voice, rather than having to trigger a new voice to hear the changes.
    // Parameters will update values once per audio callback because they
    // are outside the sample processing loop.
    mOsc.freq(getInternalParameterValue("frequency"));
    mAmpEnv.lengths()[0] = getInternalParameterValue("attackTime");
    mAmpEnv.lengths()[2] = getInternalParameterValue("releaseTime");
    mPan.pos(getInternalParameterValue("pan"));
    while (io())
    {
      float s1 = mOsc() * mAmpEnv() * getInternalParameterValue("amplitude");
      float s2;
      mEnvFollow(s1);
      mPan(s1, s1, s2);
      
      io.out(0) += s1;
      io.out(1) += s2;
    }
    // We need to let the synth know that this voice is done
    // by calling the free(). This takes the voice out of the
    // rendering chain
    if (mAmpEnv.done() && (mEnvFollow.value() < 0.001f))
      free();
  }

  // The graphics processing function
  void onProcess(Graphics &g) override
  {}

  // The triggering functions just need to tell the envelope to start or release
  // The audio processing function checks when the envelope is done to remove
  // the voice from the processing chain.
  void onTriggerOn() override
  {
    float angle = getInternalParameterValue("frequency") / 200;
    mAmpEnv.reset();

    // a = al::rnd::uniform();
    // b = al::rnd::uniform();
    
    a = gam::rnd::uni(0.0, 1.0);
    b = gam::rnd::uni(0.0, 1.0);

    timepose = 0;
    note_position = {0, 0, 0};
    note_direction = {sin(angle), cos(angle), 0};
  }

  void onTriggerOff() override { mAmpEnv.release(); }
};



// The shared state between local and the Allosphere Terminal
struct CommonState {
  // General
  Pose pose;
  float baseRadius;    
  float channelDistance;
  float oscillationAmp;
  int numSamplesInEachChannel[NUM_FLOWERS][NUM_CHANNELS];

  // Flowers' Colors and Line Locations
  HSV flowersRealTimeColors[NUM_FLOWERS][NUM_CHANNELS][WAVE_BUFFER_LENGTH];
  Vec3f flowersRealTimePositions[NUM_FLOWERS][NUM_CHANNELS][WAVE_BUFFER_LENGTH];
  Vec3f flowersRealTimeStartingPositions[NUM_FLOWERS][NUM_CHANNELS][WAVE_BUFFER_LENGTH];
  Vec3f flowersRealTimeEndingPositions[NUM_FLOWERS][NUM_CHANNELS][WAVE_BUFFER_LENGTH];

  // --- Silky Ribbon (state-replicated) ---
  int ribbonNumSlices;               // 实际使用的切片数（<= RIBBON_SLICES）
  int ribbonWidthUsed;               // 实际使用的宽度（= RIBBON_WIDTH）
  Vec3f ribbonPos[RIBBON_SLICES][RIBBON_WIDTH];   // 已应用 scrollZ 的最终绘制坐标
  Color ribbonCol[RIBBON_SLICES][RIBBON_WIDTH];   // 每顶点颜色（含 alpha）

  // --- Pulse-reactive starfield (state replicated) ---
  int   numStars;
  Vec3f starPos[STAR_MAX];
  Color starCol[STAR_MAX];
};




// To slurp a file
string slurp(string fileName);


static inline float clampf(float x, float a, float b){ return std::max(a, std::min(b, x)); }
static inline float lerpf(float a, float b, float t){ return a + (b - a) * t; }

// 频段功率 → HSV，再转 Color（含透明度）
// 频段功率 → 颜色（亮度由“中频占比”控制；RGB 空间混色避免黄绿）
static Color bandsToColor(const Mock_EEG::BandArray& B){
  const float d = std::max(0.0f, B[0]);  // delta
  const float t = std::max(0.0f, B[1]);  // theta
  const float a = std::max(0.0f, B[2]);  // alpha
  const float b = std::max(0.0f, B[3]);  // beta
  const float g = std::max(0.0f, B[4]);  // gamma

  // 心流“中频” = θ + α + 0.5β；高频 = 0.5β + γ；低频 = δ
  const float w_low  = d;
  const float w_mid  = t + a + 0.5f * b;
  const float w_high = 0.5f * b + g;
  const float sum    = w_low + w_mid + w_high + 1e-12f;

  // 基础亮度：中频占比越高越亮（开方压缩 + 抬底）
  float midFrac = w_mid / sum;
  float Vbase   = clampf(0.35f + 0.90f * std::sqrt(midFrac), 0.35f, 1.00f);

  // ★ 三个锚点用不同的亮度（红/紫更暗一点，蓝更亮一点）
  float Vlow  = Vbase * 0.85f;                                // 暗红更暗
  float Vmid  = clampf(Vbase * 1.18f, 0.0f, 1.0f);            // 河水蓝更亮
  float Vhigh = Vbase * 0.88f;                                // 暗紫更暗

  Color cLow  = Color(HSV(HUE_LOW,  1.0f, Vlow));   // red    (delta)
  Color cMid  = Color(HSV(HUE_MID,  1.0f, Vmid));   // blue   (θ+α+低β)
  Color cHigh = Color(HSV(HUE_HIGH, 1.0f, Vhigh));  // purple (高β+γ)

  // 在 RGB 空间按权重混合，避免经过黄/绿区
  float inv = 1.0f / sum;
  Color c;
  c.r = (w_low * cLow.r  + w_mid * cMid.r  + w_high * cHigh.r) * inv;
  c.g = (w_low * cLow.g  + w_mid * cMid.g  + w_high * cHigh.g) * inv;
  c.b = (w_low * cLow.b  + w_mid * cMid.b  + w_high * cHigh.b) * inv;
  c.a = 0.78f; // 半透明丝感
  return c;
}


// 宽度方向 j∈[0..RIBBON_WIDTH-1] 映射到通道索引（0..3）做线性插值，得到“折皱驱动值”
// 你当前 raw μV 偏大，我们延续你在旧代码中的缩放方式：((v/1000)-0.5)*3
static float foldFromChannels(const std::vector<float>& chVals, int j){
  if (chVals.empty()) return 0.0f;
  float t = (RIBBON_WIDTH <= 1) ? 0.0f : (float)j / float(RIBBON_WIDTH - 1); // 0..1
  float x = t * (chVals.size()-1);
  int i0 = (int)std::floor(x);
  int i1 = std::min(i0 + 1, (int)chVals.size()-1);
  float ft = x - i0;
  float v  = lerpf(chVals[i0], chVals[i1], ft);
  v = ((v / 1000.0f) - 0.5f) * 3.0f;  // 兼容你之前的显示缩放（避免一下子“飞出画面”）
  return v; // 这个 v 将作为半径微扰的一部分
}



// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// The main "app" structure
struct MyApp : public DistributedAppWithState<CommonState>, public MIDIMessageHandler
{
public:

  Mock_EEG Signal_0;
  MyApp() : Signal_0(NUM_CHANNELS, "9000") {}

  // ---- Silky Ribbon ----
  float ribbonTheta = 0.0f;   // 螺旋角（随时间累积）
  float ribbonZHead = 0.0f;   // 新切片生成的“头部 z”
  
  // ---- GUI Manager -----
  SynthGUIManager<SineEnv> synthManager{"SineEnv"};                                      // GUI manager for SineEnv voices
  RtMidiIn midiIn;            // MIDI input carrier
  Mesh mSpectrogram;
  vector<float> spectrum;
  bool showGUI = true;
  bool showSpectro = false;   // SPECTRO is not used
  bool zoomOut = true;
  bool navi = true;
  int frameCount;
  float ribbonNoisePhase = 0.0f;  // for gentle radius drift
  float chAmpEMA[NUM_CHANNELS] = {0,0,0,0};  // 4 通道“施力”平滑值
  int  printEveryNFrames = 60;
  bool debugPrint = false;      // 按 P 开关
  bool readingMode = false;   // 按 'o' 切换 CSV 回放 / 实时
  std::string playbackStamp = "2025_08_28_15-27-52"; // 你想播放的时间戳


  bool openWavForStamp(const std::string& stamp) {
    // Expect "<stamp>.wav" in CWD/bin (same place as CSVs ended up)
    std::string fname = stamp + ".wav";
    if (!mWav.openRead(fname.c_str())) {
      std::printf("[AUDIO] Can't open %s (wav)\n", fname.c_str());
      mWavLoaded = false; return false;
    }
    mWavSR = mWav.frameRate();
    analysisSR = mWavSR;

    const int channels = mWav.channels();
    const size_t frames = mWav.frames();
    // read whole file and downmix to mono
    std::vector<float> tmp(frames * channels);
    mWav.read(tmp.data(), frames);
    mWavBuf.resize(frames);
    for (size_t i = 0; i < frames; ++i) {
      double s = 0.0;
      for (int c = 0; c < channels; ++c) s += tmp[i*channels + c];
      mWavBuf[i] = float(s / std::max(1, channels));
    }
    mWavPos = 0;
    mWavLoaded = true;
    std::printf("[AUDIO] Loaded %s | sr=%.1fHz, frames=%zu, ch=%d\n",
                fname.c_str(), mWavSR, frames, channels);
    return true;
  }



  // ---- Stars ----
  Mesh Stars;
  Mesh StarsQuads;   // 用 TRIANGLES 画出的“方块星星”
  std::vector<Vec3f> starDir;    // 每颗星的单位方向
  std::vector<float> starBaseR;  // 每颗星的基准半径
  std::vector<float> starPhase;  // 每颗星的相位偏移
  float bpmEMA   = 72.0f;        // 平滑后的 BPM
  float pulsePhi = 0.0f;         // 全局心跳相位

  
  vector<vector<float>> flowersLatestValues;
  vector<vector<vector<float>>> flowersAllShownValues;

  // --------- FLOWER TOPOLOGICAL ----------
  // ** SAVED FOR LATER
  // List of triggered MIDI Notes
  // vector<int> MIDINoteTriggeredLastTime;

  // STFT variables
  gam::STFT stft{FFT_SIZE, HOP_SIZE, 0, gam::HANN, gam::COMPLEX};

  // ----- Audio file playback (read-mode) -----
  gam::SoundFile mWav;       // uses libsndfile under the hood
  std::vector<float> mWavBuf; // interleaved -> mono buffer
  size_t mWavPos = 0;
  bool   mWavLoaded = false;
  float  mWavSR = 48000.0f;   // default; will be updated from file
  float  analysisSR = 48000.0f;

  // Granular/peak driving
  float peakFloor = -80.0f;   // dB threshold
  int   maxPeaks  = 3;        // spawn up to N partials per hop
  double reTrigSec = 0.35;    // min gap per MIDI pitch (debounce)
  std::unordered_map<int,double> lastTrig; // MIDI-> last trigger time (sec)
  std::vector<std::pair<int, double>> grainsToRelease;  // {midi, releaseAtSec}
  float  grainHoldSec     = 0.80f;   // 每个粒子的“按住”时长（决定音长前半段）
  size_t maxActiveGrains  = 48;      // 最多同时活跃的粒子数（安全阈值）


  // Shader and meshes
  ShaderProgram pointShader;


  // 生成一条“截面切片”
  void makeRibbonSlice_(const std::vector<float>& vals,
                        const Mock_EEG::BandArray& B,
                        float theta, float z, RibbonSlice& out)
  {
    out.v.resize(RIBBON_WIDTH);
    out.c.resize(RIBBON_WIDTH);
    out.z0 = z;
  
    Color sliceColor = bandsToColor(B);            // 你的颜色映射函数已经在文件顶部定义
    float baseR = SPIRAL_RADIUS;                   // 初始化先不加“呼吸”，稳一点
    Vec3f radial(std::cos(theta), std::sin(theta), 0.0f);

    for (int j = 0; j < RIBBON_WIDTH; ++j) {
      float t   = (RIBBON_WIDTH<=1) ? 0.0f : (float)j / float(RIBBON_WIDTH-1);
      float off = lerpf(-RIBBON_THICKNESS, +RIBBON_THICKNESS, t);
      float fold = foldFromChannels(vals, j);      // 用当前 raw 值生成“折皱”
      float r    = baseR + off + fold * 1.0f;

      Vec3f p = radial * r;
      out.v[j] = Vec3f(p.x, p.y, z);

      // 丝绸边沿略暗
      Color c = sliceColor;
      float edge = std::abs(2.0f*t - 1.0f);
      float dim  = lerpf(1.0f, 0.88f, edge);
      c.r *= dim; c.g *= dim; c.b *= dim;
      out.c[j] = c;
    }
  }

  // 根据 gRibbon 现状，重建一次三角面片（和你 onAnimate 里的构网格完全一致）
  void rebuildRibbonMesh_()
  {
    gRibbonMesh.reset();
    gRibbonMesh.primitive(Mesh::TRIANGLES);
    if (gRibbon.size() < 2) return;

    for (size_t i = 0; i + 1 < gRibbon.size(); ++i) {
      const RibbonSlice& A = gRibbon[i];
      const RibbonSlice& B = gRibbon[i+1];
      for (int j = 0; j + 1 < RIBBON_WIDTH; ++j) {
        
        // 取四点（注意：rebuild 时不需要再减 gScrollZ）
        Vec3f a0 = A.v[j];
        Vec3f a1 = A.v[j+1];
        Vec3f b0 = B.v[j];
        Vec3f b1 = B.v[j+1];

        Color ca0 = A.c[j];
        Color ca1 = A.c[j+1];
        Color cb0 = B.c[j];
        Color cb1 = B.c[j+1];
        
        // 估法线：用“沿切向”的向量 × “沿宽度”的向量
        Vec3f widthVec = (a1 - a0);           // 当前切片内、沿宽度
        Vec3f tanVec   = (b0 - a0);           // 相邻切片、沿前进方向
        Vec3f n = cross(tanVec, widthVec);
        float nlen = n.mag();
        if (nlen < 1e-6f) n = Vec3f(0,0,1); else n /= nlen;
        Vec3f e = n * (RIBBON_THICKNESS_3D * 0.5f);  // 挤出向量（半厚度）

        // 顶/底四角
        Vec3f a0t=a0+e, a1t=a1+e, b0t=b0+e, b1t=b1+e;
        Vec3f a0b=a0-e, a1b=a1-e, b0b=b0-e, b1b=b1-e;

        // 上表面（朝 n）：a0t-b0t-a1t,  a1t-b0t-b1t
        gRibbonMesh.vertex(a0t); gRibbonMesh.color(ca0);
        gRibbonMesh.vertex(b0t); gRibbonMesh.color(cb0);
        gRibbonMesh.vertex(a1t); gRibbonMesh.color(ca1);

        gRibbonMesh.vertex(a1t); gRibbonMesh.color(ca1);
        gRibbonMesh.vertex(b0t); gRibbonMesh.color(cb0);
        gRibbonMesh.vertex(b1t); gRibbonMesh.color(cb1);

        // 下表面（朝 -n）：a0b-a1b-b0b,  a1b-b1b-b0b（逆绕序）
        gRibbonMesh.vertex(a0b); gRibbonMesh.color(ca0);
        gRibbonMesh.vertex(a1b); gRibbonMesh.color(ca1);
        gRibbonMesh.vertex(b0b); gRibbonMesh.color(cb0);

        gRibbonMesh.vertex(a1b); gRibbonMesh.color(ca1);
        gRibbonMesh.vertex(b1b); gRibbonMesh.color(cb1);
        gRibbonMesh.vertex(b0b); gRibbonMesh.color(cb0);

        // 左侧壁（j==0）
        if (j == 0) {
          gRibbonMesh.vertex(a0b); gRibbonMesh.color(ca0);
          gRibbonMesh.vertex(a0t); gRibbonMesh.color(ca0);
          gRibbonMesh.vertex(b0t); gRibbonMesh.color(cb0);

          gRibbonMesh.vertex(a0b); gRibbonMesh.color(ca0);
          gRibbonMesh.vertex(b0t); gRibbonMesh.color(cb0);
          gRibbonMesh.vertex(b0b); gRibbonMesh.color(cb0);
        }
        // 右侧壁（j == W-2）
        if (j + 2 == RIBBON_WIDTH) {
          gRibbonMesh.vertex(a1b); gRibbonMesh.color(ca1);
          gRibbonMesh.vertex(b1t); gRibbonMesh.color(cb1);
         gRibbonMesh.vertex(a1t); gRibbonMesh.color(ca1);

          gRibbonMesh.vertex(a1b); gRibbonMesh.color(ca1);
          gRibbonMesh.vertex(b1b); gRibbonMesh.color(cb1);
          gRibbonMesh.vertex(b1t); gRibbonMesh.color(cb1);
        }
      }
    }
  }



  // --------------------------------------------------------
  // onCreate
  void onCreate() override {
    bool createPointShaderSuccess = pointShader.compile(slurp("../point_tools/point-vertex.glsl"),
                                                        slurp("../point_tools/point-fragment.glsl"),
                                                        slurp("../point_tools/point-geometry.glsl"));
    if (!createPointShaderSuccess) {
      exit(1);
    }

    // Display the introduction of key control
    printf("------------------------------------------------------------ \n");
    printf("------------------------------------------------------------ \n");
    printf("Press '=' to switch between navigation mode & MIDI Mode \n");
    printf("Press '[' to turn on & off GUI \n");
    printf("Press ']' to turn on & off Spectrum \n");
    printf("Press 'p' to turn on & off EEG data display \n");
    printf("Press 'o' to switch between streaming mode & reading mode \n");
    printf("------------------------------------------------------------ \n");
    printf("------------------------------------------------------------ \n");

    // Set up the parameters for the oval
    frameCount = 0;

    // Initialize parameters for all meshes
    Flowers.primitive(Mesh::LINES);
    gRibbonMesh.primitive(Mesh::TRIANGLES);
    Stars.primitive(Mesh::POINTS);
    StarsQuads.primitive(Mesh::TRIANGLES);

    
    // Initializing the parameters in the common state
    state().baseRadius = BASE_RADIUS;
    state().channelDistance = CHANNEL_DISTANCE;
    state().oscillationAmp = OSC_AMP;

    // 初始化 Ribbon 的 state 字段
    state().ribbonNumSlices = 0;
    state().ribbonWidthUsed = RIBBON_WIDTH;
    // 可选：清空一遍
    for (int i=0;i<RIBBON_SLICES;i++){
      for (int j=0;j<RIBBON_WIDTH;j++){
         state().ribbonPos[i][j] = Vec3f(0,0,0);
      }
    }



    // --- Silky Ribbon: seed a few slices so we can see it immediately ---
    {
      gRibbon.clear();
      gScrollZ    = 0.0f;
      ribbonTheta = 0.0f;
      ribbonZHead = 0.0f;

     // 用“当前可得”的 EEG 状态生成颜色与折皱（回放/实时都能取到值）
      auto vals  = Signal_0.getLatestValues();               // 4 通道 raw

      auto bands = Signal_0.getLatestBandPowers();           // 每通道 band
      Mock_EEG::BandArray Bavg{0,0,0,0,0};
      if (!bands.empty()) {
        for (int k = 0; k < 5; ++k) {
          float s = 0.0f;
          for (auto &b : bands) s += b[k];
          Bavg[k] = s / float(bands.size());
        }
      }


      // —— 把原始 raw μV 映射到一个对位移友好的幅度，并做 EMA ——
      // 你之前就用了 ((v/1000)-0.5)*3 的缩放；我们延续它再做平滑
      for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
        float amp0 = ((vals[ch] / 1000.0f) - 0.5f) * 3.0f; // 原比例
        chAmpEMA[ch] = 0.85f * chAmpEMA[ch] + 0.15f * amp0;
      }

      // 预先铺 16 段（也可以直接铺满 RIBBON_SLICES，但 16 段足够看到形状）
      const int seedSlices = 16;
      float theta = ribbonTheta;
      float z     = ribbonZHead;

      for (int i = 0; i < seedSlices; ++i) {
        RibbonSlice sl;
        makeRibbonSlice_(vals, Bavg, theta, z, sl);

        // 注意：我们初始化采用 “push_back” 使队列从头到尾按时间递增
        gRibbon.push_back(std::move(sl));

        // 让它沿着 -Z 方向延展开（和运行时推进方向一致）
        theta += SPIRAL_SPEED;
        z     -= ADVANCE_PER_FRAME;
      }

      // 和运行时的绘制一致：立刻构建一次三角网格
      rebuildRibbonMesh_();

      printf("[INIT] ribbon slices=%zu, verts=%zu\n",
       gRibbon.size(), gRibbonMesh.vertices().size());
    }


    // Initializing the parameters in the common state
    state().baseRadius = BASE_RADIUS;
    state().channelDistance = CHANNEL_DISTANCE;
    state().oscillationAmp = OSC_AMP;

    // First, handle the backend values by setting them to 0.0
    // ****** FOR EACH PARTICIPANT (FLOWER): ******
    for (int flowerIndex = 0; flowerIndex < NUM_FLOWERS; flowerIndex++) {
      vector<float> oneFlowerLatestValues;
      vector<vector<float>> oneFlowerAllShownValues;

      // ****** FOR EACH EEG CHANNEL IN A FLOWER: ******
      
      for (int channelIndex = 0; channelIndex < NUM_CHANNELS; channelIndex++) {
        // Initialize to red
        HSV initialRed = HSV(0.0f, 1.0f, 0.75f);

        // The standard radius of this channel
        float channelRadius = BASE_RADIUS + channelIndex * CHANNEL_DISTANCE;
      
        // Calculate the amount of samples of this channel
        float channelPerimeter = 2.0 * PI * channelRadius;
        int channelNumSamples = ceil(channelPerimeter / DENSITY);
        state().numSamplesInEachChannel[flowerIndex][channelIndex] = channelNumSamples;
        float channelAngleStep = (2.0 * PI) / float(channelNumSamples);
        vector<float> oneChannelAllShownValues;


        // ****** FOR EACH SAMPLE IN A CHANNEL: ******

        for (int sampleIndex = 0; sampleIndex < channelNumSamples; sampleIndex++) {
          // Initialize the backend "all-shown" values to 0.0
          oneChannelAllShownValues.push_back(0.0);


          // Initialize the points to perfect circles
          float sampleAngle = REFRESH_ANGLE + sampleIndex * channelAngleStep;
          
          float sampleX, sampleY, sampleZ;
          if (flowerIndex == 0) {
            sampleX = channelRadius * cos(sampleAngle);
            sampleY = channelRadius * sin(sampleAngle);
            sampleZ = -1 * FLOWER_DIST * (1.0 - FLOWER_DYNAMIC * channelIndex);
          } else if (flowerIndex == 1) {
            sampleX = channelRadius * cos(sampleAngle) + 30;
            sampleY = channelRadius * sin(sampleAngle);
            sampleZ = -1 * FLOWER_DIST * (1.0 - FLOWER_DYNAMIC * channelIndex);
          } 

          Vec3f samplePos = Vec3f(sampleX, sampleY, sampleZ);
          state().flowersRealTimePositions[flowerIndex][channelIndex][sampleIndex] = samplePos;
          state().flowersRealTimeColors[flowerIndex][channelIndex][sampleIndex] = initialRed;
        }

        // Then, draw the corresponding lines according to the initialized values
        // Their initial positions should place them into circles
        vector<Vec3f> channelAllPositions;
        for (int sampleIndex = 0; sampleIndex < channelNumSamples; sampleIndex++) {
          Vec3f startingPos;
          Vec3f endingPos;

          if (sampleIndex < channelNumSamples - 1) {
            startingPos = state().flowersRealTimePositions[flowerIndex][channelIndex][sampleIndex];
            endingPos = state().flowersRealTimePositions[flowerIndex][channelIndex][sampleIndex + 1];
          } else {
            startingPos = state().flowersRealTimePositions[flowerIndex][channelIndex][sampleIndex];
            endingPos = state().flowersRealTimePositions[flowerIndex][channelIndex][0];
          }

          state().flowersRealTimeStartingPositions[flowerIndex][channelIndex][sampleIndex] = startingPos;
          state().flowersRealTimeEndingPositions[flowerIndex][channelIndex][sampleIndex] = endingPos;

          Flowers.vertex(state().flowersRealTimeStartingPositions[flowerIndex][channelIndex][sampleIndex]);
          Flowers.color(state().flowersRealTimeColors[flowerIndex][channelIndex][sampleIndex]);
          Flowers.vertex(state().flowersRealTimeEndingPositions[flowerIndex][channelIndex][sampleIndex]);
          Flowers.color(state().flowersRealTimeColors[flowerIndex][channelIndex][sampleIndex]);
        }
          
        // Initialize the backend "latest" values to 0.0
        oneFlowerLatestValues.push_back(0.0);
        oneFlowerAllShownValues.push_back(oneChannelAllShownValues);

      }
      
      flowersLatestValues.push_back(oneFlowerLatestValues);
      flowersAllShownValues.push_back(oneFlowerAllShownValues);
    }

    
    starDir.resize(STAR_COUNT);
    starBaseR.resize(STAR_COUNT);
    starPhase.resize(STAR_COUNT);

    // 固定随机分布（不依赖实时数据，primary/secondary 一致）
    for (int i = 0; i < STAR_COUNT; ++i) {
      // 均匀取球面方向（phi∈[0,2π), cosθ∈[-1,1]）
      double u1 = gam::rnd::uni(0.0, 1.0);
      double u2 = gam::rnd::uni(0.0, 1.0);
      double cz = 2.0 * u1 - 1.0;               // cos(theta)
      double sz = std::sqrt(std::max(0.0, 1.0 - cz * cz));
      double phi = 2.0 * M_PI * u2;

      Vec3f dir(sz * std::cos(phi), sz * std::sin(phi), cz);
      starDir[i] = dir.normalize();

      starBaseR[i] = STAR_RMIN + (STAR_RMAX - STAR_RMIN) * gam::rnd::uni(0.0, 1.0);
      starPhase[i] = 2.0 * M_PI * gam::rnd::uni(0.0, 1.0);

      // 初始位置 & 颜色（亮黄）
      Vec3f p = starDir[i] * starBaseR[i];
      Stars.vertex(p);
      Stars.color(Color(1.0, 0.95, 0.2, 0.95f));

      // 同步到 state（让 secondary 端也能重建）
      state().starPos[i] = p;
      state().starCol[i] = Color(1.0, 0.95, 0.2, 0.95f);
    }
    state().numStars = STAR_COUNT;



    navControl().active(false); // Disable navigation via keyboard, since we
                                // will be using keyboard for note triggering
    // Set sampling rate for Gamma objects from app's audio
    gam::sampleRate(audioIO().framesPerSecond());
    analysisSR = audioIO().framesPerSecond();     // 新增
    
    imguiInit();
    // Play example sequence. Comment this line to start from scratch
    synthManager.synthRecorder().verbose(true);

    if (isPrimary()) {

      nav().pos(15.0, 0.0, 20.0);
      nav().faceToward(0.0, 0.0, 0.0);
    }
  }


  // --------------------------------------------------------
  // onInit
  void onInit() override {
    // Try starting the program. If not successful, exit.
    auto cuttleboneDomain =
        CuttleboneStateSimulationDomain<CommonState>::enableCuttlebone(this);
    if (!cuttleboneDomain) {
      std::cerr << "ERROR: Could not start Cuttlebone. Quitting." << std::endl;
      quit();
    } else {
      cout << "Successfully Created Cuttlebone Domain" << endl;
    }

    if (isPrimary()) {
      // Check for connected MIDI devices
      if (midiIn.getPortCount() > 0)
      {
        // Bind ourself to the RtMidiIn object, to have the onMidiMessage()
        // callback called whenever a MIDI message is received
        MIDIMessageHandler::bindTo(midiIn);

        // Set up GUI
        // auto GUIdomain = GUIDomain::enableGUI(defaultWindowDomain());
        // auto& gui = GUIdomain->newGUI();

        // Open the last device found
        unsigned int port = midiIn.getPortCount() - 1;
        midiIn.openPort(port);
        printf("Opened port to %s\n", midiIn.getPortName(port).c_str());
      } else {
        printf("Actually, no MIDI devices found, please use Keyboard.\n");
      }
      // Declare the size of the spectrum 
      spectrum.resize(FFT_SIZE / 2 + 1);
    }
  }


  // --------------------------------------------------------
  // onSound
  // The audio callback function. Called when audio hardware requires data
  void onSound(AudioIOData &io) override
  {
    synthManager.render(io); // Render audio
    io.frame(0);

    // STFT
    while (io()) {
      // 1) 取一个“被分析的”输入样本：优先 WAV 回放，否则用音频输入通道
      float in = 0.0f;
      if (readingMode && mWavLoaded && mWavPos < mWavBuf.size()) {
        in = mWavBuf[mWavPos++];                 // 从 mono 缓冲取样本

        if (readingMode && mWavLoaded && mWavPos >= mWavBuf.size()) {
          mWavLoaded = false;     // WAV “歇着”
        }

      } else if (io.channelsIn() > 0) {
        in = io.in(0);                           // 用麦克风/音频输入第0通道
      }

      // 2) 喂给 STFT；当一帧准备好时，做谱分析
      if (stft(in)) {
        // 更新光谱可视化（可选）
        const int NB = stft.numBins();
        for (int k = 0; k < NB; ++k) {
          auto c = stft.bin(k);                  // 复数
          float re = c.r, im = c.i;
          float mag = std::sqrt(re*re + im*im);  // 幅度
          spectrum[k] = mag;                     // 或者做一点压缩映射
        }

        // 3) 简单峰值拾取：Top-N + dB 门限 + 邻域极大
        struct Peak { int k; float mag; };
        std::vector<Peak> peaks;
        const float floorDB = peakFloor;         // 你已有参数（-80dB 默认）
        for (int k = 1; k < NB-1; ++k) {
          auto c0 = stft.bin(k);
          float m0 = std::hypot(c0.r, c0.i);
          float db = 20.f * std::log10(std::max(1e-12f, m0));
          if (db < floorDB) continue;
          auto cL = stft.bin(k-1);
          auto cR = stft.bin(k+1);
          if (m0 > std::hypot(cL.r, cL.i) && m0 > std::hypot(cR.r, cR.i)) {
            peaks.push_back({k, m0});
          }
        }
        std::sort(peaks.begin(), peaks.end(),
                  [](const Peak& a, const Peak& b){ return a.mag > b.mag; });
        if ((int)peaks.size() > maxPeaks) peaks.resize(maxPeaks);

        // 4) 从峰值触发“慢攻/慢释”的正弦粒子（用你现成的 SineEnv）
        double nowSec = audioIO().time();
        const float binHz = analysisSR / float(FFT_SIZE);

        for (const auto& p : peaks) {
          float freq = p.k * (analysisSR / float(FFT_SIZE));
          if (freq < 40.f || freq > 8000.f) continue;

          double nowSec = audioIO().time();
          int midi = int(std::round(69.f + 12.f * std::log2(freq / 440.f)));
          if (lastTrig.count(midi) && (nowSec - lastTrig[midi]) < reTrigSec) continue;
          lastTrig[midi] = nowSec;

          // 限并发，避免堆积
          if (grainsToRelease.size() >= maxActiveGrains) break;

          float atk = 0.25f;
          float rel = 1.20f;
          float amp = std::min(0.35f, 0.08f * std::sqrt(p.mag));

          auto *v = synthManager.voice();
          v->setInternalParameterValue("frequency", freq);
          v->setInternalParameterValue("amplitude", amp);
          v->setInternalParameterValue("attackTime", atk);
          v->setInternalParameterValue("releaseTime", rel);
          synthManager.triggerOn(midi);                // 触发
          grainsToRelease.push_back({midi, nowSec + grainHoldSec});   // 登记“何时放手”
        }

        double tnow = audioIO().time();
        for (size_t i = 0; i < grainsToRelease.size(); ) {
          int    midi = grainsToRelease[i].first;
          double tRel = grainsToRelease[i].second;
          if (tnow >= tRel) {
            synthManager.triggerOff(midi);     // 放手 => 进入 releaseTime
            grainsToRelease.erase(grainsToRelease.begin() + i);
          } else {
           ++i;
          }
        }
      }

      // （你原先的其他输出混音逻辑可以继续放这里，比如把粒子声混到 io.out(0/1)）
    }
  }


  // --------------------------------------------------------
  // onAnimate
  void onAnimate(double dt) override
  {
    // The GUI is prepared here
    imguiBeginFrame();  // ?? put it in "isPrimary" or right here?
    frameCount += 1;

    ribbonNoisePhase += 0.004f;

    if (isPrimary()) {
      state().pose = nav();


      Flowers.primitive(Mesh::LINES);
      Flowers.vertices().clear();
      Flowers.colors().clear();

      gRibbonMesh.primitive(Mesh::TRIANGLES);
      gRibbonMesh.vertices().clear();
      gRibbonMesh.colors().clear();

      // First, let the classes upgrade the latest value of the EEGs
      // As well as the latest colors decided by the latest frequency values
      
      // vector<float> signal0LatestValues, signal1LatestValues, signal2LatestValues, signal3LatestValues, signal4LatestValues, signal5LatestValues, signal6LatestValues, signal7LatestValues;
      vector<float> signal0LatestValues, signal1LatestValues, signal2LatestValues;
      if (NUM_FLOWERS >= 1) {
        signal0LatestValues = Signal_0.getLatestValues();


        flowersLatestValues[0] = signal0LatestValues;
      } 

      // —— 取值（可能在启动早期还没满 NUM_CHANNELS）——
  
      auto vals       = Signal_0.getLatestValues();
      auto bandPowers = Signal_0.getLatestBandPowers();
      auto domFreqs   = Signal_0.getLatestFrequencies();

      float pulseRaw = Signal_0.getLatestPulse();
      float bpm      = Signal_0.getLatestBPM();

            // --- Heartbeat → star pulsation ---
      // 若采不到 BPM（=0 或异常），用 72 兜底；并做 EMA 平滑，避免频率突变
      float bpmValid = (bpm > 10.0f && bpm < 220.0f) ? bpm : 72.0f;
      bpmEMA = 0.9f * bpmEMA + 0.1f * bpmValid;

      // 心跳相位推进：ω = 2π * BPM / 60
      float omega = 2.0f * M_PI * (bpmEMA / 60.0f);
      pulsePhi += omega * float(dt);

      // 轻度把振幅跟随脉搏强度（未知量纲，做一个温和的压缩/映射）
      float ampScale = 0.6f + 0.4f * std::tanh(std::fabs(pulseRaw) * 0.5f);
      float amp = STAR_PULSE_AMP * ampScale;



      // —— 用世界单位尺寸把每颗星展开成两个三角 ——
      // 先清空
      StarsQuads.vertices().clear();
      StarsQuads.colors().clear();

      for (int i = 0; i < STAR_COUNT; ++i) {
        // 本帧位置（我们上一段已经算好的 p，如果你没保存，就再算一遍 p = starDir[i] * R）
        float R = starBaseR[i] + amp * std::sin(pulsePhi + starPhase[i]);
        Vec3f p = starDir[i] * R;

        // 让大小也随心跳轻微脉动（可选）
        float size = STAR_SIZE_WORLD * (1.0f + 0.25f * std::sin(pulsePhi));

        // 做一个面向“径向外”的 billboard：法线 n 指向外，e1/e2 为切向基
        Vec3f n = p; 
        float nlen = n.mag(); 
        if (nlen < 1e-6f) n = Vec3f(0,0,1); else n /= nlen;

        Vec3f ref(0,1,0);
        if (std::abs(dot(n, ref)) > 0.95f) ref = Vec3f(1,0,0);
        Vec3f e1 = cross(ref, n);  float e1l = e1.mag();  if (e1l<1e-6f) e1=Vec3f(1,0,0); else e1/=e1l;
        Vec3f e2 = cross(n, e1);   float e2l = e2.mag();  if (e2l<1e-6f) e2=Vec3f(0,1,0); else e2/=e2l;

        // 四个角
        Vec3f v0 = p - e1*size - e2*size;
        Vec3f v1 = p + e1*size - e2*size;
        Vec3f v2 = p + e1*size + e2*size;
        Vec3f v3 = p - e1*size + e2*size;

        Color c = Color(1.0, 0.95, 0.2, 0.95f); // 亮黄

        // 两个三角
        StarsQuads.vertex(v0); StarsQuads.color(c);
        StarsQuads.vertex(v1); StarsQuads.color(c);
        StarsQuads.vertex(v2); StarsQuads.color(c);

        StarsQuads.vertex(v0); StarsQuads.color(c);
        StarsQuads.vertex(v2); StarsQuads.color(c);
        StarsQuads.vertex(v3); StarsQuads.color(c);
      }



      // 防止启动时尺寸不足造成越界
      if (vals.size() < NUM_CHANNELS) vals.resize(NUM_CHANNELS, 0.0f);
      if (domFreqs.size() < NUM_CHANNELS) domFreqs.resize(NUM_CHANNELS, 0.0f);
      if (bandPowers.size() < NUM_CHANNELS) {
        bandPowers.resize(NUM_CHANNELS, Mock_EEG::BandArray{0,0,0,0,0});
      }

      // 更新缓存给绘制用
      flowersLatestValues[0] = vals;

      // —— 每帧更新 4 通道 EMA（驱动出平面“被戳”和切向“拧”）——
      for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
        float amp0 = ((vals[ch] / 1000.0f) - 0.5f) * 3.0f; // 和你现有缩放一致
        chAmpEMA[ch] = 0.85f * chAmpEMA[ch] + 0.15f * amp0;
      }


      // —— 有节制地打印 ——（避免刷屏：每 printEveryNFrames 帧打印一次）
      if (debugPrint && (frameCount % printEveryNFrames == 0)) {
        // 1) 原始值（期待 ~[-1,1] 为常态）
        printf("[EEG] raw: ");
        for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
          printf("% .3f  ", vals[ch]);
        }
        printf("\n");

        // 2) 频段功率（线性和）：[Δ,Θ,Α,Β,Γ]，范围受你python端窗长与采样率影响
        printf("[EEG] bands: ");
        for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
          auto b = bandPowers[ch];
          printf("[ch%d Δ%.2e Θ%.2e Α%.2e Β%.2e Γ%.2e]  ",
                 ch, b[0], b[1], b[2], b[3], b[4]);
        }
        printf("\n");

        // 3) 主频（Hz）
        printf("[EEG] domHz: ");
        for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
          printf("%.2f  ", domFreqs[ch]);
        }
        printf("\n");
      }



      vector<HSV> signal0LatestColors;
      signal0LatestColors.reserve(NUM_CHANNELS);

      for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
        float delta = bandPowers[ch][0];
        float theta = bandPowers[ch][1];
        float alpha = bandPowers[ch][2];
        float beta  = bandPowers[ch][3];
        float gamma = bandPowers[ch][4];

        // 亮度由 alpha/theta 主导；你可以调权重
        float score = 0.6f * alpha + 0.4f * theta;
        float eps = 1e-9f;
        float total = delta + theta + alpha + beta + gamma + eps;

        // 归一 & sqrt 压缩，让强信号更亮
        float brightness = std::sqrt(std::max(0.0f, score / total));
        brightness *= 10.33f;

        // 色相：alpha 偏蓝青(≈0.55)，theta 偏蓝紫(≈0.62)
        float a_ratio = alpha / (alpha + theta + eps);
        float hue = a_ratio * 0.55f + (1.0f - a_ratio) * 0.62f;

        // （可选）用主频做一点轻微摆动
        float f = (ch < (int)domFreqs.size()) ? domFreqs[ch] : 0.0f;
        hue += 0.02f * std::sin(f * 0.1f);
        if (hue < 0.0f) hue += 1.0f;
        if (hue > 1.0f) hue -= 1.0f;

        float saturation = 1.0f;
        signal0LatestColors.emplace_back(hue, saturation,
                                         std::max(0.35f, std::min(1.0f, brightness)));
      }




      // Update every color and draw them
      for (int flowerIndex = 0; flowerIndex < NUM_FLOWERS; flowerIndex++) {
        for (int channelIndex = 0; channelIndex < NUM_CHANNELS; channelIndex++) {
          int numSamplesInThisChannel = state().numSamplesInEachChannel[flowerIndex][channelIndex];
          for (int sampleIndex = numSamplesInThisChannel - 1; sampleIndex >= 0; sampleIndex--) {
            if (sampleIndex > 0) {
              state().flowersRealTimeColors[flowerIndex][channelIndex][sampleIndex] = state().flowersRealTimeColors[flowerIndex][channelIndex][sampleIndex - 1];
            } else {
              if (flowerIndex == 0) {
                state().flowersRealTimeColors[flowerIndex][channelIndex][sampleIndex] = signal0LatestColors[channelIndex];
              } 

              // else if (flowerIndex == 1) {
              //   state().flowersRealTimeColors[flowerIndex][channelIndex][sampleIndex] = signal1LatestColors[channelIndex];
              // }
            }
          }
        }
      }


      // Second, loop and push-forward the "all-shown" values for each flower
      // Meanwhile calculate the new position of each sample
      for (int flowerIndex = 0; flowerIndex < NUM_FLOWERS; flowerIndex++) {
        for (int channelIndex = 0; channelIndex < NUM_CHANNELS; channelIndex++) {
          // Update the values in this cahnnel
          vector<float> channelValues = flowersAllShownValues[flowerIndex][channelIndex];
          int channelNumValues = channelValues.size();
          for (int sampleIndex = channelNumValues - 1; sampleIndex >= 0; sampleIndex--) {
            if (sampleIndex > 0) {
              channelValues[sampleIndex] = channelValues[sampleIndex - 1];
            } else {
              if (flowerIndex == 0) {
                channelValues[sampleIndex] = signal0LatestValues[channelIndex];
              } 
              // else if (flowerIndex == 1) {
              //   channelValues[sampleIndex] = signal1LatestValues[channelIndex];
              // }
            }
          }
          flowersAllShownValues[flowerIndex][channelIndex] = channelValues;

          // Update the positions in the channel
          vector<Vec3f> channelPositions;
          float channelRadius = BASE_RADIUS + channelIndex * CHANNEL_DISTANCE;
          float channelAngleStep = (2.0 * PI) / float(channelNumValues);
          
          for (int sampleIndex = 0; sampleIndex < channelNumValues - 1; sampleIndex++) {
            float sampleAngle = REFRESH_ANGLE + sampleIndex * channelAngleStep;
            float sampleX, sampleY, sampleZ;
            float sampleValue = flowersAllShownValues[flowerIndex][channelIndex][sampleIndex];
            sampleValue = ((sampleValue / 1000.0f) - 0.5f) * 3.0;

            // float oscillationAmp = CHANNEL_DISTANCE;
            

            if (flowerIndex == 0) {
              sampleX = (channelRadius + (sampleValue * state().oscillationAmp)) * cos(sampleAngle);
              sampleY = (channelRadius + (sampleValue * state().oscillationAmp)) * sin(sampleAngle);
              sampleZ = -1 * FLOWER_DIST * (1.0 - FLOWER_DYNAMIC * channelIndex);
            } 
            if (flowerIndex == 1) {
              sampleX = (channelRadius + (sampleValue * state().oscillationAmp)) * cos(sampleAngle) + 30;
              sampleY = (channelRadius + (sampleValue * state().oscillationAmp)) * sin(sampleAngle);
              sampleZ = -1 * FLOWER_DIST * (1.0 - FLOWER_DYNAMIC * channelIndex);
            } 

            Vec3f samplePos = Vec3f(sampleX, sampleY, sampleZ);
            state().flowersRealTimePositions[flowerIndex][channelIndex][sampleIndex] = samplePos;
          }

          
        }
      }

      // Third, update each flowers' positions and colors
      for (int flowerIndex = 0; flowerIndex < NUM_FLOWERS; flowerIndex++) {
        for (int channelIndex = 0; channelIndex < NUM_CHANNELS; channelIndex++) {
          vector<float> channelValues = flowersAllShownValues[flowerIndex][channelIndex];
          int channelNumValues = channelValues.size();
          for (int sampleIndex = 0; sampleIndex < channelNumValues; sampleIndex++) {
            Vec3f newStartingPos;
            Vec3f newEndingPos;

            if (sampleIndex < channelNumValues - 1) {
              newStartingPos = state().flowersRealTimePositions[flowerIndex][channelIndex][sampleIndex];
              newEndingPos = state().flowersRealTimePositions[flowerIndex][channelIndex][sampleIndex + 1];
            } else {
              newStartingPos = state().flowersRealTimePositions[flowerIndex][channelIndex][sampleIndex];
              newEndingPos = state().flowersRealTimePositions[flowerIndex][channelIndex][0];
            }

            state().flowersRealTimeStartingPositions[flowerIndex][channelIndex][sampleIndex] = newStartingPos;
            state().flowersRealTimeEndingPositions[flowerIndex][channelIndex][sampleIndex] = newEndingPos;

            Flowers.vertex(state().flowersRealTimeStartingPositions[flowerIndex][channelIndex][sampleIndex]);
            Flowers.color(state().flowersRealTimeColors[flowerIndex][channelIndex][sampleIndex]);
            Flowers.vertex(state().flowersRealTimeEndingPositions[flowerIndex][channelIndex][sampleIndex]);
            Flowers.color(state().flowersRealTimeColors[flowerIndex][channelIndex][sampleIndex]);
          }
        }
      }



      // === 计算用于上色的“综合 band 颜色” ===
      // 用4个通道的 band 求均值做一个“全局色调”（也可改成每列单独算）
      Mock_EEG::BandArray Bavg{0,0,0,0,0};
      for (int ch = 0; ch < NUM_CHANNELS; ++ch){
        auto B = bandPowers[ch];
        for (int k=0; k<5; ++k) Bavg[k] += B[k];
      }
      for (int k=0; k<5; ++k) Bavg[k] /= float(std::max(1, NUM_CHANNELS));
      Color sliceColor = bandsToColor(Bavg);

      // === 生成一个新的“截面切片” ===
      RibbonSlice slice;
      slice.v.resize(RIBBON_WIDTH);
      slice.c.resize(RIBBON_WIDTH);
      slice.z0 = ribbonZHead;

      // 螺旋的参数：theta 控制绕XY转动；半径围绕 SPIRAL_RADIUS 小幅呼吸
      ribbonTheta += SPIRAL_SPEED;
      float theta = ribbonTheta;
      //float baseR = SPIRAL_RADIUS + std::sin(frameCount * 0.03f) * 2.0f; // 呼吸
      // --- organic radius drift (tiny, low-freq, angle- & time-dependent) ---
      const float NOISE_AMP   = 0.70f;   // 半径扰动幅度（单位=世界坐标；保持很小）
      const float ANG_FREQ1   = 0.23f;   // 随角度起伏的低频
      const float ANG_FREQ2   = 0.11f;   // 再叠一层更低频
      const float TIME_FREQ   = 0.015f;  // 时间缓慢起伏

      // theta 是你本来就在用的那一圈极角（makeRibbonSlice_ 里就有）
      float noise =
        NOISE_AMP * ( 0.55f * std::sin(theta * ANG_FREQ1 + ribbonNoisePhase)
                    + 0.35f * std::sin(theta * ANG_FREQ2 - 1.3f * ribbonNoisePhase)
                    + 0.25f * std::sin(frameCount * TIME_FREQ) );

      // 若你还有呼吸项，可略微降低幅度避免叠加过强
      float baseR = SPIRAL_RADIUS
                  + noise
                  + 0.6f * std::sin(frameCount * 0.020f);   // 轻微“呼吸”


      // 径向与切向
      Vec3f radial(std::cos(theta), std::sin(theta), 0.0f); // 指向圆心外


      // —— 建立局部正交基 ——
      // radial 指向圆心外（你已有）；切向沿旋转方向；bHat 出平面（Z 向）
      Vec3f tHat(-std::sin(theta), std::cos(theta), 0.0f);
      Vec3f bHat(0.0f, 0.0f, 1.0f); // XY 平面法线

      // —— slice 级“有机倾斜” ——（让丝带平面轻微晃动，而不是永远同向）
      const float ORIENT_TILT_AMP  = 0.30f;   // 倾斜角幅度（弧度）
      const float ORIENT_T_TIME    = 0.08f;   // 随时间变化的低频
      const float ORIENT_T_THETA   = 0.31f;   // 随角度变化的低频
      float yawN   = ORIENT_TILT_AMP * 0.7f * std::sin(frameCount * ORIENT_T_TIME  + theta * ORIENT_T_THETA);
      float pitchN = ORIENT_TILT_AMP * 0.5f * std::sin(frameCount * (ORIENT_T_TIME*1.3f) - theta * 0.22f + 1.1f);

      // 基法线以 z 轴为基，叠一点 x/y 抖动，然后重新正交化出“宽度方向”
      Vec3f bBase(0,0,1);
      Vec3f bTilt = (bBase + Vec3f(yawN, pitchN, 0)).normalize();          // 倾斜后的“法线”
      Vec3f widthDir = cross(bTilt, tHat).normalize();                      // 丝带“横向”
      Vec3f radialUsed = cross(tHat, bTilt).normalize();                    // “圆心外”方向（受倾斜影响一点点）


      for (int j = 0; j < RIBBON_WIDTH; ++j) {
        float t   = (RIBBON_WIDTH<=1) ? 0.0f : (float)j / float(RIBBON_WIDTH-1); // 0..1
        float off = lerpf(-RIBBON_THICKNESS, +RIBBON_THICKNESS, t);

        // —— 径向起伏（保留你原来的“拉抻毛巾”逻辑）——
        float foldR = foldFromChannels(vals, j);         // 径向
        // float r     = baseR + off + foldR * 1.0f;

        // —— 出平面“被戳”的起伏（4 通道高斯加权，连续影响）——
        // 通道中心（内→外）：0.125, 0.375, 0.625, 0.875
        const float centers[NUM_CHANNELS] = {
          (0 + 0.5f)/NUM_CHANNELS, (1 + 0.5f)/NUM_CHANNELS,
          (2 + 0.5f)/NUM_CHANNELS, (3 + 0.5f)/NUM_CHANNELS
        };
        
        // 先算全局均值（四通道平均），用于“去公共位移”
        float avgAmp = 0.0f;
        for (int k = 0; k < NUM_CHANNELS; ++k) avgAmp += chAmpEMA[k];
        avgAmp *= 0.25f;

        // 高斯核权重（中心 0.125/0.375/0.625/0.875；sigma 用上面的 LIFT_SIGMA）
        float wsum = 0.0f, liftMix = 0.0f;
        for (int k = 0; k < NUM_CHANNELS; ++k) {
          float u = (t - centers[k]) / LIFT_SIGMA;
          float w = std::exp(-0.5f * u * u);
          wsum    += w;
          // ★ 用“通道偏离全局均值”的量，强调左右/中部差异，抑制整条一起动
          liftMix += w * (chAmpEMA[k] - avgAmp);
        }
        if (wsum > 1e-6f) liftMix /= wsum;

        // 温和非线性放大，让差异更显眼但不炸
        liftMix = std::tanh(1.3f * liftMix);

        // —— 切向“拧”一点（让丝带不只在一个平面摇）——
        float twist = TWIST_GAIN * liftMix * std::sin((t - 0.5f) * PI); // 中央≈0，边缘相反

        // —— 合成 3D 位置：径向 + 切向 + 出平面 —— 
        // r0 是沿圆周的中心半径 + EEG 折皱的径向扰动（不会把整条带掀飞）
        float r0 = baseR + foldR * 1.0f;

        // 圆周上的“中心点”
        Vec3f center = radialUsed * r0;

        // 横向展开、切向微拧 + 你已有的“被戳出平面”的 lift（保持你现有的 liftMix 逻辑）
        Vec3f xy = center + widthDir * off + tHat * twist;

        // 如果你还在用“以世界 Up 为主”的 liftDir 写法，可以把下面两行改成：
        // Vec3f p3 = Vec3f(xy.x, xy.y, slice.z0) + liftDir * (LIFT_GAIN * liftMix);
        // slice.v[j] = p3;
        float z = slice.z0 + LIFT_GAIN * liftMix;
        slice.v[j] = Vec3f(xy.x, xy.y, z);


        // —— 你已有的“边缘略暗”处理，保留 —— 
        Color c = sliceColor;
        float edge = std::abs(2.0f*t - 1.0f);
        float dim  = lerpf(1.0f, 0.88f, edge);
        c.r *= dim; c.g *= dim; c.b *= dim;
        slice.c[j] = c;
      }


      // 推入队列 & 维护长度
      gRibbon.emplace_front(std::move(slice));
      if ((int)gRibbon.size() > RIBBON_SLICES) gRibbon.pop_back();

      // 整体前推滚动（视觉上“河”向前缓慢推进）
      ribbonZHead += ADVANCE_PER_FRAME;
      gScrollZ    += ADVANCE_PER_FRAME;

      // === 以相邻两截面构网格（三角带→三角形） ===
      // gRibbonMesh.colors().clear();
      // gRibbonMesh.vertices().clear();
      // gRibbonMesh.primitive(Mesh::TRIANGLES);
      if (gRibbon.size() >= 2){
        for (size_t i=0; i+1<gRibbon.size(); ++i){
          const RibbonSlice& A = gRibbon[i];
          const RibbonSlice& B = gRibbon[i+1];
          for (int j=0; j+1<RIBBON_WIDTH; ++j){
            
            // 取四点（主分支：要减 gScrollZ）
            Vec3f a0 = A.v[j];     a0.z -= gScrollZ;
            Vec3f a1 = A.v[j+1];   a1.z -= gScrollZ;
            Vec3f b0 = B.v[j];     b0.z -= gScrollZ;
            Vec3f b1 = B.v[j+1];   b1.z -= gScrollZ;

            Color ca0 = A.c[j];
            Color ca1 = A.c[j+1];
            Color cb0 = B.c[j];
            Color cb1 = B.c[j+1];

            // 法线与挤出
            Vec3f widthVec = (a1 - a0);
            Vec3f tanVec   = (b0 - a0);
            Vec3f n = cross(tanVec, widthVec);
            float nlen = n.mag();
            if (nlen < 1e-6f) n = Vec3f(0,0,1); else n /= nlen;
            Vec3f e = n * (RIBBON_THICKNESS_3D * 0.5f);

            // 顶/底四角
            Vec3f a0t=a0+e, a1t=a1+e, b0t=b0+e, b1t=b1+e;
            Vec3f a0b=a0-e, a1b=a1-e, b0b=b0-e, b1b=b1-e;

            // 上表面
            gRibbonMesh.vertex(a0t); gRibbonMesh.color(ca0);
            gRibbonMesh.vertex(b0t); gRibbonMesh.color(cb0);
            gRibbonMesh.vertex(a1t); gRibbonMesh.color(ca1);

            gRibbonMesh.vertex(a1t); gRibbonMesh.color(ca1);
            gRibbonMesh.vertex(b0t); gRibbonMesh.color(cb0);
            gRibbonMesh.vertex(b1t); gRibbonMesh.color(cb1);

            // 下表面（逆绕序）
            gRibbonMesh.vertex(a0b); gRibbonMesh.color(ca0);
            gRibbonMesh.vertex(a1b); gRibbonMesh.color(ca1);
            gRibbonMesh.vertex(b0b); gRibbonMesh.color(cb0);

            gRibbonMesh.vertex(a1b); gRibbonMesh.color(ca1);
            gRibbonMesh.vertex(b1b); gRibbonMesh.color(cb1);
            gRibbonMesh.vertex(b0b); gRibbonMesh.color(cb0);

            // 侧壁
            if (j == 0) {
              gRibbonMesh.vertex(a0b); gRibbonMesh.color(ca0);
              gRibbonMesh.vertex(a0t); gRibbonMesh.color(ca0);
              gRibbonMesh.vertex(b0t); gRibbonMesh.color(cb0);

              gRibbonMesh.vertex(a0b); gRibbonMesh.color(ca0);
              gRibbonMesh.vertex(b0t); gRibbonMesh.color(cb0);
              gRibbonMesh.vertex(b0b); gRibbonMesh.color(cb0);
            }
            if (j + 2 == RIBBON_WIDTH) {
              gRibbonMesh.vertex(a1b); gRibbonMesh.color(ca1);
              gRibbonMesh.vertex(b1t); gRibbonMesh.color(cb1);
              gRibbonMesh.vertex(a1t); gRibbonMesh.color(ca1);
            
              gRibbonMesh.vertex(a1b); gRibbonMesh.color(ca1);
              gRibbonMesh.vertex(b1b); gRibbonMesh.color(cb1);
              gRibbonMesh.vertex(b1t); gRibbonMesh.color(cb1);
            }
          }
        }
      }

      

      // === 把 Ribbon 的最终绘制数据同步到 state() ===
      // 注意：我们把 z 都写成 “已经 -gScrollZ 之后”的最终绘制值
      int S = std::min((int)gRibbon.size(), RIBBON_SLICES);
      state().ribbonNumSlices = S;
      state().ribbonWidthUsed = RIBBON_WIDTH;
      for (int i = 0; i < S; ++i) {
        const auto& sl = gRibbon[i];
        for (int j = 0; j < RIBBON_WIDTH; ++j) {
          Vec3f p = sl.v[j]; p.z -= gScrollZ;         // 最终绘制坐标
          state().ribbonPos[i][j] = p;
          state().ribbonCol[i][j] = sl.c[j];          // 直接存 RGBA
        }
      }



      // Draw a window that contains the synth control panel
      // Refresh & is primary
      synthManager.drawSynthControlPanel();
      imguiEndFrame();
      navControl().active(navi);

    } else {
      nav().set(state().pose);
      Flowers.vertices().clear();
      Flowers.colors().clear();
      for (int flowerIndex = 0; flowerIndex < NUM_FLOWERS; flowerIndex++) {
        for (int channelIndex = 0; channelIndex < NUM_CHANNELS; channelIndex++) {
          int numSamplesInThisChannel = state().numSamplesInEachChannel[flowerIndex][channelIndex];
          for (int sampleIndex = 0; sampleIndex < numSamplesInThisChannel - 1; sampleIndex++) {
            Flowers.vertex(state().flowersRealTimeStartingPositions[flowerIndex][channelIndex][sampleIndex]);
            Flowers.color(state().flowersRealTimeColors[flowerIndex][channelIndex][sampleIndex]);
            Flowers.vertex(state().flowersRealTimeEndingPositions[flowerIndex][channelIndex][sampleIndex]);
            Flowers.color(state().flowersRealTimeColors[flowerIndex][channelIndex][sampleIndex]);
          }
        }
      }

      // --- 从 state() 重建 ribbon 的本地网格（非 primary）---
      gRibbonMesh.reset();
      gRibbonMesh.primitive(Mesh::TRIANGLES);

      int S = std::max(0, state().ribbonNumSlices);
      int W = std::max(2, state().ribbonWidthUsed);
      if (S >= 2) {
        for (int i = 0; i + 1 < S; ++i) {
          for (int j = 0; j + 1 < W; ++j) {
            Vec3f a0 = state().ribbonPos[i][j];
            Vec3f a1 = state().ribbonPos[i][j+1];
            Vec3f b0 = state().ribbonPos[i+1][j];
            Vec3f b1 = state().ribbonPos[i+1][j+1];

            Color ca0 = state().ribbonCol[i][j];
            Color ca1 = state().ribbonCol[i][j+1];
            Color cb0 = state().ribbonCol[i+1][j];
            Color cb1 = state().ribbonCol[i+1][j+1];

            // 法线与挤出
            Vec3f widthVec = (a1 - a0);
            Vec3f tanVec   = (b0 - a0);
            Vec3f n = cross(tanVec, widthVec);
            float nlen = n.mag();
            if (nlen < 1e-6f) n = Vec3f(0,0,1); else n /= nlen;
            Vec3f e = n * (RIBBON_THICKNESS_3D * 0.5f);

            // 顶/底四角
            Vec3f a0t=a0+e, a1t=a1+e, b0t=b0+e, b1t=b1+e;
            Vec3f a0b=a0-e, a1b=a1-e, b0b=b0-e, b1b=b1-e;

            // 上表面
            gRibbonMesh.vertex(a0t); gRibbonMesh.color(ca0);
            gRibbonMesh.vertex(b0t); gRibbonMesh.color(cb0);
            gRibbonMesh.vertex(a1t); gRibbonMesh.color(ca1);

            gRibbonMesh.vertex(a1t); gRibbonMesh.color(ca1);
            gRibbonMesh.vertex(b0t); gRibbonMesh.color(cb0);
            gRibbonMesh.vertex(b1t); gRibbonMesh.color(cb1);

            // 下表面（逆绕序）
            gRibbonMesh.vertex(a0b); gRibbonMesh.color(ca0);
            gRibbonMesh.vertex(a1b); gRibbonMesh.color(ca1);
            gRibbonMesh.vertex(b0b); gRibbonMesh.color(cb0);

            gRibbonMesh.vertex(a1b); gRibbonMesh.color(ca1);
            gRibbonMesh.vertex(b1b); gRibbonMesh.color(cb1);
            gRibbonMesh.vertex(b0b); gRibbonMesh.color(cb0);

            // 侧壁
            if (j == 0) {
              gRibbonMesh.vertex(a0b); gRibbonMesh.color(ca0);
              gRibbonMesh.vertex(a0t); gRibbonMesh.color(ca0);
              gRibbonMesh.vertex(b0t); gRibbonMesh.color(cb0);

              gRibbonMesh.vertex(a0b); gRibbonMesh.color(ca0);
              gRibbonMesh.vertex(b0t); gRibbonMesh.color(cb0);
              gRibbonMesh.vertex(b0b); gRibbonMesh.color(cb0);
            }
            if (j + 2 == RIBBON_WIDTH) {
              gRibbonMesh.vertex(a1b); gRibbonMesh.color(ca1);
              gRibbonMesh.vertex(b1t); gRibbonMesh.color(cb1);
              gRibbonMesh.vertex(a1t); gRibbonMesh.color(ca1);

              gRibbonMesh.vertex(a1b); gRibbonMesh.color(ca1);
              gRibbonMesh.vertex(b1b); gRibbonMesh.color(cb1);
              gRibbonMesh.vertex(b1t); gRibbonMesh.color(cb1);
            }
          }
        }
      }

      StarsQuads.vertices().clear();
      StarsQuads.colors().clear();
      int Ns = std::max(0, state().numStars);
      if (Ns > STAR_MAX) Ns = STAR_MAX;

      for (int i = 0; i < Ns; ++i) {
        Vec3f p = state().starPos[i];

        float size = STAR_SIZE_WORLD; // 也可以用和主端一致的脉动：1+0.25*sin(pulsePhi)，
                                      // 如果 pulsePhi 你也同步在 state 里

        Vec3f n = p; 
        float nlen = n.mag(); 
        if (nlen < 1e-6f) n = Vec3f(0,0,1); else n /= nlen;

        Vec3f ref(0,1,0);
        if (std::abs(dot(n, ref)) > 0.95f) ref = Vec3f(1,0,0);
       Vec3f e1 = cross(ref, n);  float e1l = e1.mag();  if (e1l<1e-6f) e1=Vec3f(1,0,0); else e1/=e1l;
        Vec3f e2 = cross(n, e1);   float e2l = e2.mag();  if (e2l<1e-6f) e2=Vec3f(0,1,0); else e2/=e2l;

        Vec3f v0 = p - e1*size - e2*size;
        Vec3f v1 = p + e1*size - e2*size;
        Vec3f v2 = p + e1*size + e2*size;
        Vec3f v3 = p - e1*size + e2*size;

        Color c = state().starCol[i]; // 亮黄

        StarsQuads.vertex(v0); StarsQuads.color(c);
        StarsQuads.vertex(v1); StarsQuads.color(c);
        StarsQuads.vertex(v2); StarsQuads.color(c);
        StarsQuads.vertex(v0); StarsQuads.color(c);
        StarsQuads.vertex(v2); StarsQuads.color(c);
        StarsQuads.vertex(v3); StarsQuads.color(c);
      }
    } 
  }


  // --------------------------------------------------------
  // onDraw
  // The graphics callback function.
  void onDraw(Graphics &g) override
  {
    g.clear(0.0);
    // synthManager.render(g); <- This is commented out because we show ANN but not the notes

    // --- Flower ---
    g.meshColor();
    g.draw(Flowers);

    // --- Ribbon ---
    g.shader().use();
    g.blending(true);
    g.blendTrans();
    g.depthTesting(true);
    g.meshColor();
    g.draw(gRibbonMesh);

    // --- Stars ---
    g.shader().use();       // 退回默认 shader（确保没沿用 pointShader）
    g.blending(true);
    g.blendAdd();           // 星星更“发光”，也可以用 g.blendTrans()
    g.depthTesting(true);
    g.meshColor();
    g.draw(StarsQuads);




    // Draw Spectrum
    // Commented out for testing drawing the meshes of ANN only
    ///*
    mSpectrogram.reset();
    mSpectrogram.primitive(Mesh::LINE_STRIP);
    if (showSpectro)
    {
      for (int i = 0; i < FFT_SIZE / 2; i++)
      {
        mSpectrogram.color(HSV(0.5 - spectrum[i] * 100));
        mSpectrogram.vertex(i, spectrum[i], 0.0);
      }
      g.meshColor(); // Use the color in the mesh
      g.pushMatrix();
      g.translate(-5.0, -3, 0);
      g.scale(100.0 / FFT_SIZE, 100, 1.0);
      g.draw(mSpectrogram);
      g.popMatrix();
    }
    //*/
    // GUI is drawn here
    if (showGUI)
    {
      imguiDraw();
      // ? how to show the "gui"
      // defined on line (): auto& gui = GUIdomain->newGUI();
      // with more adjustable parameters?
    }
    
  }


  // This gets called whenever a MIDI message is received on the port
  void onMIDIMessage(const MIDIMessage &m)
  {
    switch (m.type())
    {
    case MIDIByte::NOTE_ON:
    {
      int midiNote = m.noteNumber();
      if (midiNote > 0 && m.velocity() > 0.001)
      {
        synthManager.voice()->setInternalParameterValue(
            "frequency", ::pow(2.f, (midiNote - 69.f) / 12.f) * 432.f);
        synthManager.voice()->setInternalParameterValue(
            "attackTime", 0.1/m.velocity());
        synthManager.triggerOn(midiNote);
        printf("On Note %u, Vel %f \n", m.noteNumber(), m.velocity());
      }
      else
      {
        synthManager.triggerOff(midiNote);
        printf("Off Note %u, Vel %f \n", m.noteNumber(), m.velocity());
      }
      break;
    }
    case MIDIByte::NOTE_OFF:
    {
      int midiNote = m.noteNumber();
      printf("Note OFF %u, Vel %f", m.noteNumber(), m.velocity());
      synthManager.triggerOff(midiNote);
      break;
    }
    default:;
    }
  }

  // Whenever a key is pressed, this function is called
  bool onKeyDown(Keyboard const &k) override
  {
    if (ParameterGUI::usingKeyboard())
    { // Ignore keys if GUI is using
      // keyboard
      return true;
    }
    if (!navi)
    {
      if (k.shift())
      {
        // If shift pressed then keyboard sets preset
        int presetNumber = asciiToIndex(k.key());
        synthManager.recallPreset(presetNumber);
      }
      else
      {
        // Otherwise trigger note for polyphonic synth
        int midiNote = asciiToMIDI(k.key());
        if (midiNote > 0)
        {
          synthManager.voice()->setInternalParameterValue(
              "frequency", ::pow(2.f, (midiNote - 69.f) / 12.f) * 432.f);
          synthManager.triggerOn(midiNote);
        }
      }
    }
    switch (k.key())
    {
    case ']':
      showGUI = !showGUI;
      break;
    case '[':
      showSpectro = !showSpectro;
      break;
    case '=':
      navi = !navi;
      if (navi) {
        printf("Nagivation Mode, MIDI Disabled");
      } else {
        printf("MIDI Mode, Navigation Disabled");
      }
      break;
    case 'p':
      debugPrint = !debugPrint;
      if (debugPrint) {
        printf("Data Showing Mode On");
      } else {
        printf("Data Showing Mode Off");
      }
      break;
    case 'o':
      // if (audioIO().time() - modeSwitchAt < 0.15) continue; // 150ms 宽限
      readingMode = !readingMode;
      if (readingMode) {
        bool ok = Signal_0.enablePlayback(playbackStamp);
        if (ok) {
          printf("[READING MODE] Playing CSV %s\n", playbackStamp.c_str());
          if (!openWavForStamp(playbackStamp)) {
            printf("[AUDIO] WAV not found for %s (visuals continue)\n", playbackStamp.c_str());
          }
        } else {
          readingMode = false;
          printf("[READING MODE] Failed to load CSV. Back to LIVE.\n");
        }
      } else {
        Signal_0.disablePlayback();
        mWavLoaded = false; mWavBuf.clear(); mWavPos = 0;
        analysisSR = audioIO().framesPerSecond();
        printf("[STREAMING MODE] Back to live OSC.\n");
      }
      break;
    case 'i':
      zoomOut = !zoomOut;
      if (zoomOut) {
        nav().pos(30.0, 0.0, 70.0);
        nav().faceToward(0.0, 0.0, 0.0);
      } else {
        nav().pos(0.0, 0.0, 0.0);
        nav().faceToward(0.0, 0.0, 0.0);
      }
      break;
    }
    
    return true;
  }

  // Whenever a key is released this function is called
  bool onKeyUp(Keyboard const &k) override
  {
    int midiNote = asciiToMIDI(k.key());
    if (midiNote > 0)
    {
      synthManager.triggerOff(midiNote);
    }
    return true;
  }

  void onExit() override { 
    imguiShutdown(); 

    Flowers.colors().clear();
    Flowers.vertices().clear();

    gRibbonMesh.colors().clear();
    gRibbonMesh.vertices().clear();
  }
};


// slurp
// To slurp from a file
//
string slurp(string fileName) {
  fstream file(fileName);
  string returnValue = "";
  while (file.good()) {
    string line;
    getline(file, line);
    returnValue += line + "\n";
  }
  return returnValue;
}

int main()
{
  MyApp app;

  if (al::Socket::hostName() == "ar01.1g") {
    AudioDevice device = AudioDevice("ECHO X5");
    app.configureAudio(device, 44100, 128, device.channelsOutMax(), 2);
  } else {
    app.configureAudio(48000., 512, 2, 2);
  }
  app.start();
  return 0;
}




