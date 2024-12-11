// Yuehao Gao | MAT594P
// 2024-11-1  | Mock EEG

/*
 * Mock_EEG.cpp
 * 
 * This is a C++ implementation of a Mock EEG generator that simulates multi-channel EEG signals.
 * Each channel produces a sine wave oscillating between -1.0 and 1.0, with frequencies that 
 * gradually vary within a specified range. The class mimics real-time EEG behavior, allowing 
 * dynamic and realistic testing scenarios.
 * 
 * Features:
 * - Configurable number of channels and frequency range.
 * - Gradually changing frequencies for dynamic signal simulation.
 * - Outputs the latest values of all channels as a vector of floats.
 * 
 * Usage:
 * 1. Instantiate the Mock_EEG class with the desired number of channels and frequency range.
 * 2. Call `getLatestValues()` to retrieve the latest simulated signal values for all channels.
 * 
 * Example:
 * Mock_EEG eeg(8, 4.0f, 30.0f);
 * std::vector<float> values = eeg.getLatestValues();

 */

// ---------------------------------------------------------------------------------

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

// Tests
// int main() {
//     Mock_EEG eeg(8, 4.0f, 30.0f);  // 8 channels, frequency range [4 Hz, 30 Hz]
//                                    // Same parameters for p5.js
//     // Simulation of something happening in 1 second
//     for (int i = 0; i < 60; ++i) {
//         vector<float> values = eeg.getLatestValues();
//         vector<float> freqs = eeg.getLatestFrequencies();

//         cout << "EEG Values: ";
//         for (float val : values) {
//             cout << val << " ";
//         }
//         cout << endl;

//         cout << "Frequencies: ";
//         for (float freq : freqs) {
//             cout << freq << " ";
//         }
//         cout << endl;
//     }
//     return 0;
// }
