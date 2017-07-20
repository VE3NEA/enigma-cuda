#pragma once

/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include "cuda_err_check.h"
#include "cuda_code.h"
#include "ngrams.h"

struct Task
{
    int count;
    PitchedArray scrambler;
    PitchedArray trigrams;
    int8_t turnover_modes;
    int score_kinds; 
    Result * results;
};

struct Block
{
    int count;
    int * trigrams;
    int8_t plugs[ALPSIZE];
    int unigrams[ALPSIZE];
    ScoreKind score_kind;
    int volatile score_buf[MAX_MESSAGE_LENGTH];
    int8_t plain_text[MAX_MESSAGE_LENGTH];
    int score;
};
 
extern Task h_task;

__constant__ extern int8_t d_ciphertext[MAX_MESSAGE_LENGTH];
__constant__ extern Wiring d_wiring;
__constant__ extern Key d_key;
__constant__ extern NGRAM_DATA_TYPE d_unigrams[ALPSIZE];
__constant__ extern int8_t d_order[ALPSIZE];
__constant__ extern int8_t d_plugs[ALPSIZE];
extern Result * d_temp;


extern "C"
{
    __device__
    void Sum(int count, volatile int * data, int * score);
    __device__
    void IcScore(Block & block, const int8_t * scrambling_table);
    __device__
    void UniScore(Block & block, const int8_t * scrambling_table);
    __device__
    void BiScore(Block & block, const int8_t * scrambling_table);
    __device__
    void TriScore(Block & block, const int8_t * scrambling_table);
    __device__ 
    void TrySwap(int8_t i, int8_t k,
      const int8_t * scrambling_table, Block & block);
    __device__
    void MaximizeScore(Block & block, const int8_t * scrambling_table);
    __device__
    void CalculateScore(Block & block, const int8_t * scrambling_table);
    __device__
    const int8_t * ScramblerToShared(const int8_t * global_scrambling_table);

    __global__
    void GenerateScramblerKernel(const Task task);
    __global__ 
    void FindBestResultKernel(Result *g_idata, Result *g_odata,
        unsigned int count);
    __global__ 
    void ClimbKernel(const Task task);
}

