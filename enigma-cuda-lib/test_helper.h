#pragma once

/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include <inttypes.h>
#include "cuda_code.h"

extern int RunTrySwap(const string & cipher_string, const string & key_string,
    const string & plugboard_string, const int8_t * order,
    int letter1, int letter2, ScoreKind score_kind,
    const string & unigram_file_name, const string & trigram_file_name);
extern int RunMaximizeScore(const string & cipher_string, const string & key_string,
        const string & plugboard_string, const int8_t * order, ScoreKind score_kind,
        const string & unigram_file_name, const string & trigram_file_name);
extern void MockResults(int count);
extern void SetUpClimb(int digits, const string cipher_string, const string key_string,
        const int8_t * order, int turnover_modes, const string & unigram_file_name,
        const string & trigram_file_name);
extern Result RunClimb(int digits, const string cipher_string, const string key_string);
extern void RunFullBreak(const string cipher_string, const int8_t * order,
        const string & unigram_file_name, const string & trigram_file_name);
extern int ComputeIcScore(const string & cipher_string,
        const string & key_string, const string & plugboard_string);
extern int ComputeUniScore(const string & cipher_string,
        const string & key_string, const string & plugboard_string,
        const string & unigrams_file_name);
extern int ComputeBiScore(const string & cipher_string, const string & key_string,
    const string & plugboard_string, const string & bigram_file_name);
extern int ComputeTriScore(const string & cipher_string,
        const string & key_string, const string & plugboard_string,
        const string & trigrams_file_name);
extern void PrintTrigrams(const string & file_name);
