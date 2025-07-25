

/*
 * -----------------------------------------------------------------------------
 * THESE CODE ARE FOR RECEIVING REAL SIGNALS FROM "read_mindmonitor.py"
 * THIS FILE IS CALLED "MOCK_EEG" BECAUSE IT WAS ORIGINALLY GENERATING MOCK SIGNALS,
 * WITHOUT LISTENING TO ANY INPUT
 * "Flow_Sphere" READS DATA FROM THIS FILE FOR VISUALIZATION
 * 
 * BUT NOW, THIS FILE IS DESIGNED TO LISTEN TO OSC / SOCKET SIGNALS FROM "read_mindmonitor.py"
 * 
 * THE CODE BELOW (UNCOMMENTED) ARE THOSE FOR LISTENING THE REAL SIGNALS
 * THEY ARE NOT WORKING YET
 * -----------------------------------------------------------------------------
 */


 

#include "Mock_EEG.hpp"
#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include <cstring>

#include "osc/OscReceivedElements.h"
#include "osc/OscPacketListener.h"
#include "ip/UdpSocket.h"

#define EEG_OSC_PORT 9000

// #define NUM_CHANNELS 4



class EEGOscListener : public osc::OscPacketListener {
    public:
        static const int NUM_CHANNELS = 4;
        float receivedValues[NUM_CHANNELS] = {0.0f};
        std::mutex mtx;
    
    protected:
    void ProcessMessage(const osc::ReceivedMessage& m, const IpEndpointName& remoteEndpoint) override {
        try {
            if (std::strcmp(m.AddressPattern(), "/eeg/raw") == 0) {
                float temp[NUM_CHANNELS] = {0.0f};
                osc::ReceivedMessage::const_iterator arg = m.ArgumentsBegin();
                for (int i = 0; i < NUM_CHANNELS && arg != m.ArgumentsEnd(); ++i, ++arg) {
                    temp[i] = (arg)->AsFloatUnchecked();
                }
    
                std::lock_guard<std::mutex> lock(mtx);
                const float scaleFactor = 1000.0f;  // 缩放到[-1,1]范围
                for (int i = 0; i < NUM_CHANNELS; ++i) {
                    float originalValue = temp[i];
                    originalValue = ((originalValue / scaleFactor) - 0.75f) * 10.0f;

                    receivedValues[i] = originalValue;
                }
            }
        } catch (osc::Exception& e) {
            std::cerr << "Error while parsing OSC message: " << e.what() << std::endl;
        }
    }
};




static EEGOscListener listener;
static UdpListeningReceiveSocket oscSocket(
    IpEndpointName(IpEndpointName::ANY_ADDRESS, EEG_OSC_PORT),
    &listener
);

static void startOSCThread() {
    std::thread([] { oscSocket.Run(); }).detach();
}



// Mock_EEG implementation
Mock_EEG::Mock_EEG(int numChannels, float minFreq, float maxFreq)
    : numChannels(numChannels), minFreq(minFreq), maxFreq(maxFreq), useRealData(true) {
    values.resize(numChannels, 0.0f);
    freqs.resize(numChannels, minFreq);

    startOSCThread();

    updateThread = std::thread([this] {
        while (true) {
            {
                std::lock_guard<std::mutex> lock(listener.mtx);
                for (int i = 0; i < this->numChannels; ++i) {
                    this->values[i] = listener.receivedValues[i];
                    this->freqs[i] = this->minFreq + static_cast<float>(rand()) /
                        (static_cast<float>(RAND_MAX / (this->maxFreq - this->minFreq)));
                }

                std::cout << "[EEG OSC] Values: ";
                for (int i = 0; i < this->numChannels; ++i) {
                    std::cout << this->values[i] << " ";
                }
                std::cout << std::endl;


            }
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
    });
    updateThread.detach();
}


std::vector<float> Mock_EEG::getLatestValues() {
    return values;
}

std::vector<float> Mock_EEG::getLatestFrequencies() {
    return freqs;
}






/*
 * -----------------------------------------------------------------------------
 * THE CODE BELOW (COMMENTED) ARE THOSE FOR GENERATING MOCK SIGNALS
 * THEY USED TO BE WORKING FINE
 * PLEASE UNCOMMENT THEM FOR TESTING
 * 
 * PLEASE DO NOT UNCOMMENT THE "TESTS" PART
 * -----------------------------------------------------------------------------
 */

 /*
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <optional>
#include "al/math/al_Random.hpp"


using namespace std;

class Mock_EEG {
private:
    int numChannels;
    float minFreq;
    float maxFreq;
    vector<float> frequencies;              // Current frequencies for each channel
    vector<float> frequencyDeltas;          // Rate of frequency change for each channel
    vector<float> phases;                   // Current phase for each channel
    const float refreshTime = 1.0f / 60.0f; // Time step for external refresh rate (1/60 seconds)
    const float deltaPhase = 2 * M_PI;      // Phase increment per second

public:
    // Constructor
    Mock_EEG(int channels, float minFrequency, float maxFrequency) 
        : numChannels(channels), minFreq(minFrequency), maxFreq(maxFrequency) {
        // Seed random generator
        srand(time(0));

        // Initialize frequencies, deltas, and phases
        for (int i = 0; i < numChannels; ++i) {
            frequencies.push_back(randomFloat(minFreq, maxFreq));
            frequencyDeltas.push_back(randomFloat(0.01f, 0.05f)); // Small deltas for gradual change
            phases.push_back(randomFloat(0, 2 * M_PI));           // Random initial phase
        }
    }

    // Function to generate a random float between a range
    float randomFloat(float min, float max) {
        return min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
    }

    // Function to update frequencies and phases, and return the latest signal values
    vector<float> getLatestValues() {
        vector<float> latestValues;
        for (int i = 0; i < numChannels; ++i) {
            // Update frequency within bounds
            frequencies[i] += frequencyDeltas[i];
            if (frequencies[i] > maxFreq || frequencies[i] < minFreq) {
                frequencyDeltas[i] *= -1; // Reverse direction when out of bounds
            }

            
            // Calculate the latest value based on sine wave
            float value = sin(phases[i]); // Generate sine value
            latestValues.push_back(value);

            // Update the phase
            phases[i] += deltaPhase * frequencies[i] * refreshTime;
            if (phases[i] > 2 * M_PI) {
                phases[i] -= 2 * M_PI; // Keep phase within 0 to 2π
            }
        }
        return latestValues;
    }

    // New function to return the current frequencies of all channels
    vector<float> getLatestFrequencies() {
        return frequencies;
    }
};

*/



// ---------------------------------------------------------
// ---------- PLEASE DO NOT UNCOMMENT THIS PART ------------
// ---------------------------------------------------------

// Tests
// int main() {
    // Mock_EEG eeg(8, 4.0f, 30.0f);  // 8 channels, frequency range [4 Hz, 30 Hz]
    //                                // Same parameters for p5.js
    // // Simulation of something happening in 1 second
    // for (int i = 0; i < 60; ++i) {
    //     vector<float> values = eeg.getLatestValues();
    //     vector<float> freqs = eeg.getLatestFrequencies();

    //     cout << "EEG Values: ";
    //     for (float val : values) {
    //         cout << val << " ";
    //     }
    //     cout << endl;

    //     cout << "Frequencies: ";
    //     for (float freq : freqs) {
    //         cout << freq << " ";
    //     }
    //     cout << endl;
    // }
    // return 0;
// }


// ---------------------------------------------------------
