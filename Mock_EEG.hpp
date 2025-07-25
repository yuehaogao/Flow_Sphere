#pragma once
#ifndef MOCK_EEG_HPP
#define MOCK_EEG_HPP

#include <vector>
#include <thread>

class Mock_EEG {
    public:
        Mock_EEG(int numChannels, float minFreq, float maxFreq);
        std::vector<float> getLatestValues();
        std::vector<float> getLatestFrequencies();
    
    private:
        int numChannels;
        float minFreq;
        float maxFreq;
        bool useRealData;
    
        std::vector<float> values;
        std::vector<float> freqs;
        std::thread updateThread;
    };

#endif // MOCK_EEG_HPP
