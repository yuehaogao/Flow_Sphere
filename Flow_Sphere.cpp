// MAT594P, Spring 2024
// Yuehao Gao
// Designed based on Myungin Lee(2022) Sine Envelope with Visuals
// Inspired by Neo (Mostafa) Mohsenvand's Project: "Flower"
// https://www.media.mit.edu/projects/flower-eeg-visualization-with-the-aid-of-machine-learning/overview/


// 3:55PM
// LAST VERSION BEFORE THE MAJOR CHANGE TO MULTI-FLOWER IN ONE MESH

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

#include "Mock_EEG.cpp"       // The simulator of multi-channel EEG signal

//#include "al/app/al_App.hpp"
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

// using namespace gam;
using namespace al;
using namespace std;

#define FFT_SIZE 4048
#define PI 3.1415926535

const int NUM_FLOWERS = 2;                   // How many flowers are there, meaning how many participants are observed
const int NUM_CHANNELS = 24;                 // How many EEG channels are there for one participant
const int NUM_SHADOWS = 8;                   // How many lines (hot and shadows) are there for each channel
const int WAVE_BUFFER_LENGTH = 600;          // The length of the buffer storing the wave values
const float DENSITY = 0.35;                  // How dense the samples are on each channel
const float MIN_FREQUENCY = 4.0;             // Lower limit of mock EEG frequency range
const float MAX_FREQUENCY = 30.0;            // Upper limit of mock EEG frequency range
const float REFRESH_ANGLE = PI * 0.5;        // The refreshing point in the circle
                                             // ** CHANGE TO VARIABLE LATER
const float CENTRAL_RADIUS = 6.0;            // The radius of the middle channel of each EEG Flower Mesh
const float CHANNEL_DISTANCE = 10.0 / NUM_CHANNELS;          
                                             // The distance between each channel
                                             // ** CHANGE TO VARIABLE LATER
const float OSC_AMP = CHANNEL_DISTANCE;      // The oscilation amplitude of each channel
                                             // ** CHANGE TO VARIABLE LATER
const float OSC_FACTOR = 0.05;                // A constant deciding the oscillation amplitude
const float BASE_RADIUS = CENTRAL_RADIUS - (0.5 * CHANNEL_DISTANCE * NUM_FLOWERS);               
                                             // The radius of the inner-most channel
const float FLOWER_DIST = 50.0;              // The distance of each flower mesh to the origin
                                             // ** CHANGE TO VARIABLE LATER

// const float INTERVAL = 1.0 / 60.0; // 1/60 second
// const int NUM_SAMPLES_INNER = 300;
// const int DURATION = 6; 


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
  double spin = al::rnd::uniformS();
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

    createInternalTriggerParameter("amplitude", 0.03, 0.0, 0.1);
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
    a = al::rnd::uniform();
    b = al::rnd::uniform();
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

  // Flowers' Colors and Line Locations
  HSV flowersRealTimeColors[NUM_FLOWERS][NUM_CHANNELS];
  Vec3f flowersRealTimePositions[NUM_FLOWERS][NUM_CHANNELS][WAVE_BUFFER_LENGTH];
};


// To slurp a file
string slurp(string fileName);


// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// The main "app" structure
struct MyApp : public DistributedAppWithState<CommonState>, public MIDIMessageHandler
{
public:
  
  SynthGUIManager<SineEnv> synthManager{"SineEnv"};                                      // GUI manager for SineEnv voices
  RtMidiIn midiIn;            // MIDI input carrier
  Mesh mSpectrogram;
  vector<float> spectrum;
  bool showGUI = true;
  bool showSpectro = false;   // SPECTRO is not used
  bool navi = true;
  int frameCount;

  // --------- EEG VALUES ARCHIVE ----------
  Mock_EEG Mock_Signal_1 = Mock_EEG(NUM_CHANNELS, MIN_FREQUENCY, MAX_FREQUENCY);
  Mock_EEG Mock_Signal_2 = Mock_EEG(NUM_CHANNELS, MIN_FREQUENCY, MAX_FREQUENCY);

  vector<vector<float>> flowersLatestValues;
  vector<vector<vector<float>>> flowersAllShownValues;
  
  // --------- FLOWER TOPOLOGICAL ----------
  Mesh Flowers;
  
  // ** SAVED FOR LATER
  // List of triggered MIDI Notes
  // vector<int> MIDINoteTriggeredLastTime;

  // STFT variables
  gam::STFT stft = gam::STFT(FFT_SIZE, FFT_SIZE / 4, 0, gam::HANN, gam::MAG_FREQ);

  // Shader and meshes
  ShaderProgram pointShader;


  // --------------------------------------------------------
  // onCreate
  void onCreate() override {
    bool createPointShaderSuccess = pointShader.compile(slurp("../point_tools/point-vertex.glsl"),
                                                        slurp("../point_tools/point-fragment.glsl"),
                                                        slurp("../point_tools/point-geometry.glsl"));
    if (!createPointShaderSuccess) {
      exit(1);
    }

    // Set up the parameters for the oval
    frameCount = 0;

    // Initialize parameters for all meshes
    Flowers.primitive(Mesh::LINES);

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
        HSV initialRed = HSV(0.0f, 1.0f, 0.7f);
        state().flowersRealTimeColors[flowerIndex][channelIndex] = initialRed;

        // The standard radius of this channel
        float channelRadius = BASE_RADIUS + channelIndex * CHANNEL_DISTANCE;
      
        // Calculate the amount of samples of this channel
        float channelPerimeter = 2.0 * PI * channelRadius;
        int channelNumSamples = ceil(channelPerimeter / DENSITY);
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
            sampleZ = -1 * FLOWER_DIST;
          } else if (flowerIndex == 1) {
            sampleX = channelRadius * cos(sampleAngle) + 20;
            sampleY = channelRadius * sin(sampleAngle);
            sampleZ = -1 * FLOWER_DIST;
          } 

          Vec3f samplePos = Vec3f(sampleX, sampleY, sampleZ);
          state().flowersRealTimePositions[flowerIndex][channelIndex][sampleIndex] = samplePos;
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

          Flowers.vertex(startingPos);
          Flowers.color(initialRed);
          Flowers.vertex(endingPos);
          Flowers.color(initialRed);
        }

        // Initialize the backend "latest" values to 0.0
        oneFlowerLatestValues.push_back(0.0);
        oneFlowerAllShownValues.push_back(oneChannelAllShownValues);
      }

      flowersLatestValues.push_back(oneFlowerLatestValues);
      flowersAllShownValues.push_back(oneFlowerAllShownValues);
    }

    navControl().active(false); // Disable navigation via keyboard, since we
                                // will be using keyboard for note triggering
    // Set sampling rate for Gamma objects from app's audio
    gam::sampleRate(audioIO().framesPerSecond());
    imguiInit();
    // Play example sequence. Comment this line to start from scratch
    synthManager.synthRecorder().verbose(true);

    if (isPrimary()) {
      nav().pos(0.0, 0.0, 0.0);
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
    // STFT
    while (io())
    {
      if (stft(io.out(0)))
      { // Loop through all the frequency bins
        for (unsigned k = 0; k < stft.numBins(); ++k)
        {
          // Here we simply scale the complex sample
          spectrum[k] = 10.0 * tanh(pow(stft.bin(k).real(), 1.5) );
          //spectrum[k] = stft.bin(k).real();
        }
      }
    }
  }


  // --------------------------------------------------------
  // onAnimate
  void onAnimate(double dt) override
  {
    // The GUI is prepared here
    imguiBeginFrame();  // ?? put it in "isPrimary" or right here?
    frameCount += 1;

    if (isPrimary()) {
      state().pose = nav();
      Flowers.primitive(Mesh::LINES);

      // Clear the positions and colors from the previous frame
      Flowers.vertices().clear();
      HSV initialRed = HSV(0.0f, 1.0f, 0.7f);
      // Flowers.colors().clear();

      // First, let the classes upgrade the latest value of the mock EEGs
      vector<float> signal0LatestValues = Mock_Signal_1.getLatestValues();
      flowersLatestValues[0] = signal0LatestValues;
      vector<float> signal1LatestValues = Mock_Signal_2.getLatestValues();
      flowersLatestValues[1] = signal1LatestValues;
      
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
              } else if (flowerIndex == 1) {
                channelValues[sampleIndex] = signal1LatestValues[channelIndex];
              }
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
            float oscillationAmp = CHANNEL_DISTANCE;
            

            if (flowerIndex == 0) {
              sampleX = channelRadius * cos(sampleAngle) * (1.0 + sampleValue * OSC_AMP * OSC_FACTOR);
              sampleY = channelRadius * sin(sampleAngle) * (1.0 + sampleValue * OSC_AMP * OSC_FACTOR);
              sampleZ = -1 * FLOWER_DIST;
            } else if (flowerIndex == 1) {
              sampleX = channelRadius * cos(sampleAngle) * (1.0 + sampleValue * OSC_AMP * OSC_FACTOR) + 20;
              sampleY = channelRadius * sin(sampleAngle) * (1.0 + sampleValue * OSC_AMP * OSC_FACTOR);
              sampleZ = -1 * FLOWER_DIST;
            } 

            if (frameCount <= 5) {
              cout << sampleValue << endl;
            }

            Vec3f samplePos = Vec3f(sampleX, sampleY, sampleZ);
            state().flowersRealTimePositions[flowerIndex][channelIndex][sampleIndex] = samplePos;
          }

          // Third, update each flowers' positions and colors
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

            Flowers.vertex(newStartingPos);
            Flowers.color(initialRed);
            Flowers.vertex(newEndingPos);
            Flowers.color(initialRed);
          }
        }
      }

      // Third, update each flowers' positions and colors
      // Flowers.vertices().clear();
      // // Flowers.colors().clear();
      // for (int flowerIndex = 0; flowerIndex < NUM_FLOWERS; flowerIndex++) {
      //   for (int channelIndex = 0; channelIndex < NUM_CHANNELS; channelIndex++) {
      //     for (int sampleIndex = 0; sampleIndex < channelNumValues; sampleIndex++) {
      //       Vec3f startingPos;
      //       Vec3f endingPos;

      //       if (sampleIndex < channelNumValues - 1) {
      //         startingPos = state().flowersRealTimePositions[flowerIndex][channelIndex][sampleIndex];
      //         endingPos = state().flowersRealTimePositions[flowerIndex][channelIndex][sampleIndex + 1];
      //       } else {
      //         startingPos = state().flowersRealTimePositions[flowerIndex][channelIndex][sampleIndex];
      //         endingPos = state().flowersRealTimePositions[flowerIndex][channelIndex][0];
      //      }

      //       Flowers.vertex(startingPos);
      //       Flowers.color(initialRed);
      //       Flowers.vertex(endingPos);
      //       Flowers.color(initialRed);
      //     }
      //   }
      // }
      
      // ------- EXECUTE THE REFRESHING WORKS HERE -------


        // for (int startLayer = 0; startLayer < currentFiredNeuronPos.size() - 1; startLayer++) {
        //   vector<Vec3f> startLayerPositions = currentFiredNeuronPos[startLayer];
        //   vector<Vec3f> endLayerPositions = currentFiredNeuronPos[startLayer + 1];

        //   for (Vec3f oneStartPosition : startLayerPositions) {
        //    for (Vec3f oneEndPosition : endLayerPositions) {
        //       ConnectionLines.vertex(oneStartPosition);
        //       ConnectionLines.color(HSV(0.17f, 1.0f, 1.0f));
        //       ConnectionLines.vertex(oneEndPosition);
        //       ConnectionLines.color(HSV(0.17f, 1.0f, 1.0f));

        //       state().linesStartingFixedPosition[lineIteration] = oneStartPosition;
        //       state().linesEndingFixedPosition[lineIteration] = oneEndPosition;

        //       lineIteration += 1;
        //     }
        //   }
        // }
        

        // Draw a window that contains the synth control panel
        // Refresh & is primary
        synthManager.drawSynthControlPanel();
        imguiEndFrame();
        navControl().active(navi);

    } else {
        nav().set(state().pose);

        // Flowers.vertices().clear();
        // Flowers.colors().clear();
        // for (int line = 0; line < linePosArchiveSize; line++) {
        //   ConnectionLines.vertex(state().linesStartingFixedPosition[line]);
        //   ConnectionLines.color(state().linesRealTimeColor);
        //   ConnectionLines.vertex(state().linesEndingFixedPosition[line]);
        //   ConnectionLines.color(state().linesRealTimeColor);
        // }

    }

    
  }


  // --------------------------------------------------------
  // onDraw
  // The graphics callback function.
  void onDraw(Graphics &g) override
  {
    g.clear(0.0);
    // synthManager.render(g); <- This is commented out because we show ANN but not the notes

    g.meshColor();
    g.draw(Flowers);

    g.shader(pointShader);
    g.blending(true);
    g.blendTrans();
    g.depthTesting(true);


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
