#pragma once

/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include <string>
#include <stdint.h>
#include "const.h"
using std::string;


//TODO: try uint8_t instead of int
#define NGRAM_DATA_TYPE int

class Unigrams
{
public:
    NGRAM_DATA_TYPE data[ALPSIZE] = { 0 };
    void LoadFromFile(const string& file_name);
};

class Bigrams
{
public:
    NGRAM_DATA_TYPE data[ALPSIZE][ALPSIZE] = { 0 };
    int pitch = ALPSIZE * sizeof(NGRAM_DATA_TYPE);
    void LoadFromFile(const string& file_name);
    int ScoreText(const string & text);
};

class Trigrams
{
public:
    NGRAM_DATA_TYPE data[ALPSIZE][ALPSIZE][ALPSIZE] = { 0 };
    int pitch = ALPSIZE * sizeof(NGRAM_DATA_TYPE);
    void LoadFromFile(const string& file_name);
    int ScoreText(const string & text);
};
