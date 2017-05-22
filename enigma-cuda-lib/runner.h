#pragma once

/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include <iostream>
#include <fstream>
#include "runner.h"
#include "settings.h"
#include "iterator.h"
#include "segmenter.h"
#include "cuda_code.h" 

class Runner
{
private:
    std::ofstream out_file_stream;
    std::ostream * out_stream;

    string ciphertext;
    int length;
    KeyIterator iterator;
    Plugboard plugboard;
    WordSegmenter segmenter;
    clock_t start_time, last_save_time;
    int current_pass;
    string progress_string;
public:
    Settings settings;
    bool silent = false;

    bool Initialize(int max_length = MAX_MESSAGE_LENGTH);
    bool Run();
    void ProcessResult(const Result & result);
    void ShowProgressString(bool show = true);
};
