#pragma once

/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include <string>
#include <map>
using std::string;

#define MAX_MESSAGE_LENGTH 256

class WordSegmenter
{
private:
    std::map<string, double> words;
    int max_length = 0;
    size_t  total_count = 0;
    string input_text;
    int best_lengths[MAX_MESSAGE_LENGTH + 1];
    double length_probs[MAX_MESSAGE_LENGTH + 1];

    double GetWordScore(const string & word);
    string UpCaseKnownWords(const string & text, bool with_spaces = true);
    bool IsWord(const string & word);
public:
    void LoadWordsFromFile(const string & file_name = "");
    bool IsLoaded() { return total_count > 0; }
    int FindSegmentation(const string & text);
    string GetSegmentedText(bool up_case = false, bool with_spaces = true);
};

