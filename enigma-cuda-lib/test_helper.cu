/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include <iostream>
#include <iomanip>
#include <time.h>
#include <math.h> 
#include <ctime>
#include <algorithm>

#include "test_helper.h"
#include "cuda_err_check.h"
#include "cuda_code.cuh"
#include "cuda_code.h"
#include "plugboard.h"
#include "ngrams.h"
#include "iterator.h"


void CopyScramblerToHost(PitchedArray & dst)
{
    dst.pitch = ALPSIZE;
    CUDA_CHECK(cudaMemcpy2D(dst.data, ALPSIZE, h_task.scrambler.data,
        h_task.scrambler.pitch, ALPSIZE, ALPSIZE_TO3, cudaMemcpyDeviceToHost));
}

__global__
void TestRunIcScoreKernel(PitchedArray scrambler, int * score)
{
    const int8_t * scrambling_table = scrambler.data + ComputeScramblerIndex(threadIdx.x, d_key.stru, d_key.sett, d_wiring) * scrambler.pitch;
    scrambling_table = ScramblerToShared(scrambling_table);
   
    __shared__ Block block;
    if (threadIdx.x < ALPSIZE) block.plugs[threadIdx.x] = d_plugs[threadIdx.x];
    block.count = blockDim.x;

    IcScore(block, scrambling_table);

    if (threadIdx.x == 0) *score = block.score;
}

__global__
void TestRunUniScoreKernel(PitchedArray scrambler, int * score)
{
    const int8_t * scrambling_table = scrambler.data + ComputeScramblerIndex(threadIdx.x, d_key.stru, d_key.sett, d_wiring) * scrambler.pitch;
    scrambling_table = ScramblerToShared(scrambling_table);

    __shared__ Block block;
    if (threadIdx.x < ALPSIZE)
    {
      block.plugs[threadIdx.x] = d_plugs[threadIdx.x];
      block.unigrams[threadIdx.x] = d_unigrams[threadIdx.x];
    }
    block.count = blockDim.x;        
    UniScore(block, scrambling_table);

    if (threadIdx.x == 0) *score = block.score;
}

__global__
void TestRunBiScoreKernel(PitchedArray scrambler, int count, int * score)
{
    const int8_t * scrambling_table = scrambler.data + ComputeScramblerIndex(threadIdx.x, d_key.stru, d_key.sett, d_wiring) * scrambler.pitch;
    scrambling_table = ScramblerToShared(scrambling_table);

    __syncthreads();

    __shared__ Block block;
    if (threadIdx.x < ALPSIZE) block.plugs[threadIdx.x] = d_plugs[threadIdx.x];
    block.count = count;

    BiScore(block, scrambling_table);

    if (threadIdx.x == 0) *score = block.score;
}

__global__
void TestRunTriScoreKernel(
    PitchedArray scrambler, PitchedArray trigrams, int count, int * score)
{
    int i = threadIdx.x;

    const int8_t * scrambling_table = scrambler.data + ComputeScramblerIndex(i, d_key.stru, d_key.sett, d_wiring) * scrambler.pitch;
    scrambling_table = ScramblerToShared(scrambling_table);

    __syncthreads();

    __shared__ Block block;
    block.trigrams = reinterpret_cast<int*>(trigrams.data);
    if (threadIdx.x < ALPSIZE) block.plugs[threadIdx.x] = d_plugs[threadIdx.x];
    block.count = count;

    TriScore(block, scrambling_table);

    if (threadIdx.x == 0) *score = block.score;
}

__global__
void TestRunTrySwapKernel(PitchedArray scrambler,
    int letter1, int letter2, ScoreKind score_kind, PitchedArray trigrams, int * score)
{
    const int8_t * scrambling_table = scrambler.data + ComputeScramblerIndex(threadIdx.x, d_key.stru, d_key.sett, d_wiring) * scrambler.pitch;
    scrambling_table = ScramblerToShared(scrambling_table);

    __shared__ Block block;
    block.trigrams = reinterpret_cast<int*>(trigrams.data);
    block.score_kind = score_kind;
    block.count = blockDim.x;
    if (threadIdx.x < ALPSIZE)
    {
      block.plugs[threadIdx.x] = d_plugs[threadIdx.x];
      block.unigrams[threadIdx.x] = d_unigrams[threadIdx.x];
    }

    CalculateScore(block, scrambling_table);

    TrySwap(letter1, letter2, scrambling_table, block);
    __syncthreads();

    if (threadIdx.x == 0) *score = block.score;
    __syncthreads();

}

__global__
void TestRunMaximizeScoreKernel(PitchedArray scrambler,
    ScoreKind score_kind, PitchedArray trigrams, int * score)
{
    const int8_t * scrambling_table = scrambler.data + ComputeScramblerIndex(threadIdx.x, d_key.stru, d_key.sett, d_wiring) * scrambler.pitch;
    scrambling_table = ScramblerToShared(scrambling_table);

    __shared__ Block block;
    block.trigrams = reinterpret_cast<int*>(trigrams.data);
    block.score_kind = score_kind;
    block.count = blockDim.x;
    if (threadIdx.x < ALPSIZE)
    {
      block.plugs[threadIdx.x] = d_plugs[threadIdx.x];
      block.unigrams[threadIdx.x] = d_unigrams[threadIdx.x];
    }
    __syncthreads();

    MaximizeScore(block, scrambling_table);

    if (threadIdx.x == 0) *score = block.score;
    __syncthreads();
}



int * test_d_score;



void SetUpScoreTest(const string cipher_string, const string key_string,
    const string & plugboard_string)
{
    //cipher
    CipherTextToDevice(cipher_string);

    //plugs
    PlugboardStringToDevice(plugboard_string);

    //key
    Key key;
    key.FromString(key_string);

    //scrambler
    SetUpScramblerMemory();
    GenerateScrambler(key);

    CUDA_CHECK(cudaMalloc((void**)&test_d_score, sizeof(int)));
}

int CleanUpScoreTest()
{
    //check for errors launching the kernel
    CUDA_CHECK(cudaGetLastError());
    //check for errors executing the kernel
    CUDA_CHECK(cudaDeviceSynchronize());

    int score;
    CUDA_CHECK(cudaMemcpy(&score, test_d_score, sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(test_d_score);

    cudaDeviceReset();

    return score;
}

int ComputeIcScore(const string & cipher_string, const string & key_string,
    const string & plugboard_string)
{
    SetUpScoreTest(cipher_string, key_string, plugboard_string);

    Plugboard plugboard;
    plugboard.FromString(plugboard_string);
    CUDA_CHECK(cudaMemcpyToSymbol(d_plugs, plugboard.plugs, ALPSIZE));

    int block_size = std::max(32, (int)cipher_string.length());
    int shared_scrambler_size = ((cipher_string.length() + 31) & ~31) * 28;
    TestRunIcScoreKernel << < 1, block_size, shared_scrambler_size >> >
        (h_task.scrambler, test_d_score);

    return CleanUpScoreTest();
}

int ComputeUniScore(const string & cipher_string, const string & key_string,
    const string & plugboard_string, const string & unigram_file_name)
{
    SetUpScoreTest(cipher_string, key_string, plugboard_string);
    NgramsToDevice(unigram_file_name, "", "");

    int block_size = std::max(32, (int)cipher_string.length());
    int shared_scrambler_size = ((cipher_string.length() + 31) & ~31) * 28;
    TestRunUniScoreKernel << < 1, block_size, shared_scrambler_size >> >
        (h_task.scrambler, test_d_score);

    return CleanUpScoreTest();
}

int ComputeBiScore(const string & cipher_string, const string & key_string,
    const string & plugboard_string, const string & bigram_file_name)
{
    SetUpScoreTest(cipher_string, key_string, plugboard_string);
    NgramsToDevice("", bigram_file_name, "");

    int count = (int)cipher_string.length();
    int block_size = std::max(32, count);
    int shared_scrambler_size = ((cipher_string.length() + 31) & ~31) * 28;
    TestRunBiScoreKernel << < 1, block_size, shared_scrambler_size >> >
        (h_task.scrambler, count, test_d_score);

    return CleanUpScoreTest();
}

int ComputeTriScore(const string & cipher_string, const string & key_string,
    const string & plugboard_string, const string & trigram_file_name)
{
    SetUpScoreTest(cipher_string, key_string, plugboard_string);
    NgramsToDevice("", "", trigram_file_name);

    int count = (int)cipher_string.length();
    int block_size = std::max(32, count);
    int shared_scrambler_size = ((cipher_string.length() + 31) & ~31) * 28;
    TestRunTriScoreKernel << < 1, block_size, shared_scrambler_size >> >
        (h_task.scrambler, h_task.trigrams, count, test_d_score);

    return CleanUpScoreTest();
}

int RunTrySwap(const string & cipher_string, const string & key_string,
    const string & plugboard_string, const int8_t * order, int letter1,
    int letter2, ScoreKind score_kind,
    const string & unigram_file_name, const string & trigram_file_name)
{
    SetUpScoreTest(cipher_string, key_string, plugboard_string);
    NgramsToDevice(unigram_file_name, "", trigram_file_name);
    OrderToDevice(order);

    int shared_scrambler_size = ((cipher_string.length() + 31) & ~31) * 28;
    TestRunTrySwapKernel << < 1, (int)cipher_string.length(), shared_scrambler_size >> >
        (h_task.scrambler, letter1, letter2,
            score_kind, h_task.trigrams, test_d_score);

    return CleanUpScoreTest();
}

int RunMaximizeScore(const string & cipher_string, const string & key_string,
    const string & plugboard_string, const int8_t * order, ScoreKind score_kind,
    const string & unigram_file_name, const string & trigram_file_name)

{
    SetUpScoreTest(cipher_string, key_string, plugboard_string);
    NgramsToDevice(unigram_file_name, "", trigram_file_name);
    OrderToDevice(order);

    int block_size = std::max(32, (int)cipher_string.length());
    int shared_scrambler_size = ((cipher_string.length() + 31) & ~31) * 28;
    TestRunMaximizeScoreKernel << < 1, block_size, shared_scrambler_size >> >
        (h_task.scrambler, score_kind, h_task.trigrams, test_d_score);

    return CleanUpScoreTest();
}

void MockResults(int count)
{
    SetUpResultsMemory(count);

    Result * mock_data = new Result[count];
    for (int i = 0; i < count; i++)
    {
        mock_data[i].index = i;
        mock_data[i].score = i + 1;
    }

    CUDA_CHECK(cudaMemcpy((void *)h_task.results, (void *)mock_data, count * sizeof(Result), cudaMemcpyHostToDevice));

    delete[] mock_data;

    d_temp = NULL;
}

void SetUpClimb(int digits, const string cipher_string, const string key_string,
    const int8_t * order, int turnover_modes, const string & unigram_file_name,
    const string & trigram_file_name)
{
    //d_ciphertext
    CipherTextToDevice(cipher_string);
    //d_wiring
    SetUpScramblerMemory();
    //d_unigrams, trigrams
    NgramsToDevice(unigram_file_name, "", trigram_file_name);
    //d_order
    OrderToDevice(order);
    //d_plugs
    Plugboard plugboard;
    plugboard.Reset();
    CUDA_CHECK(cudaMemcpyToSymbol(d_plugs, plugboard.plugs, ALPSIZE));
    //allow_turnover
    h_task.turnover_modes = turnover_modes;
    //d_results
    int count = (int)pow(ALPSIZE, digits);
    SetUpResultsMemory(count);
    //d_key, scrambler
    Key key;
    key.FromString(key_string);
    GenerateScrambler(key);
    h_task.score_kinds = skIC | skUnigram | skTrigram;
}


Result RunClimb(int digits, const string cipher_string, const string key_string)
{

    const dim3 gridDim[] = {
        { 1, 1 },
        { ALPSIZE, 1 },
        { ALPSIZE_TO2, 1 },
        { ALPSIZE_TO3, 1 },
        { ALPSIZE_TO3, ALPSIZE },
        { ALPSIZE_TO3, ALPSIZE_TO2 } };


    try
    {
        Key key;
        key.FromString(key_string);
        CUDA_CHECK(cudaMemcpyToSymbol(d_key, &key, sizeof(Key)));

        int block_size = std::max(32, h_task.count);
        int shared_scrambler_size = ((cipher_string.length() + 31) & ~31) * 28;
        ClimbKernel << <gridDim[digits], block_size, shared_scrambler_size >> > (h_task);

        //check for errors launching the kernel
        CUDA_CHECK(cudaGetLastError());
        //check for errors executing the kernel
        CUDA_CHECK(cudaDeviceSynchronize());

        int count = (int)pow(ALPSIZE, digits);
        Result result = GetBestResult(count);

        cudaFree(d_temp);
        d_temp = NULL;
        cudaFree(h_task.results);

        return result;
    }
    catch (const std::runtime_error & e)
    {
        std::cout << e.what() << std::endl;
        cudaGetLastError();
        cudaDeviceReset();
        cudaSetDevice(0);
        throw e;
    }
}

void PrintResult(const Result & result, Key key, clock_t start_time)
{
    key.sett.r_mesg = result.index % ALPSIZE;
    key.sett.m_mesg = (result.index / ALPSIZE) % ALPSIZE;
    key.sett.l_mesg = result.index / ALPSIZE_TO2;

    Plugboard plugs;
    plugs.FromData(result.plugs);

    std::cout << "Time:  " << TimeDiffString(clock() - start_time) << std::endl;
    std::cout << "Score: " << result.score << std::endl;
    std::cout << "Key:   " << key.ToString() << std::endl;
    std::cout << "Plugs: " << plugs.ToString() << std::endl;
    std::cout << std::endl;
}

void RunFullBreak(const string cipher_string, const int8_t * order,
    const string & unigram_file_name, const string & trigram_file_name)
{
    bool small_blocks = true;


    KeyIterator iter;
    //iter.SetRange("B:123:XX:XXX", "B:543:XX:XXX"); //all rotor orders and ring pos, 10 min
    iter.SetRange("B:524:AA:XXX", "B:524:ZZ:XXX");   //all ring pos, 8 sec

    SetUpClimb(small_blocks ? 3 : 4, cipher_string, iter.key.ToString(), order, toBeforeMessage, unigram_file_name, trigram_file_name);

    Result best_result = { 0 };
    clock_t start_time = clock();

    do
    {
//        GenerateScrambler(iter.key);
        do
        {
            GenerateScrambler(iter.key);
            Result result = RunClimb(small_blocks ? 3 : 4, cipher_string, iter.key.ToString());

            if (result.score > best_result.score)
            {
                best_result = result;
                PrintResult(result, iter.key, start_time);

            }
        } while (iter.NextRingPosition(small_blocks));
    } while (iter.NextRotorOrder());

    std::cout << "DONE in " << TimeDiffString(clock() - start_time) << std::endl;

    cudaFree(d_temp);
    d_temp = NULL;
    cudaFree(h_task.results);
}


//------------------------------------------------------------------------------
//                        trigram masking experiments
//------------------------------------------------------------------------------
void PrintTrigramsOld(const string & file_name)
{
  const string GermanFreqLetters = "ENXRSIATUOLFDGMBZQKHWPVYCJ";

  Trigrams trigrams;
  trigrams.LoadFromFile(file_name);

  std::cout << "--+--------------------------+" << std::endl;
  for (int i = 0; i < ALPSIZE; i++)
  {
    for (int j = 0; j < ALPSIZE; j++)
    {
      std::cout << char('a' + i) << char('a' + j) << "|";
      for (int k = 0; k < ALPSIZE; k++)
      {
        char c = k;// GermanFreqLetters[k] - 'A';
        std::cout << ((trigrams.data[i][j][c] > 0) ? char('A' + c) : ' ');
      }
      std::cout << "|" << std::endl;
    }
    std::cout << "--+--------------------------+" << std::endl;
  }
}

void PrintTrigrams(const string & file_name)
{
  Trigrams trigrams;
  trigrams.LoadFromFile(file_name);
  bool mask[ALPSIZE][ALPSIZE][ALPSIZE] = { false };
  bool m;

  for (int i = 0; i < ALPSIZE; i++)
    for (int j = 0; j < ALPSIZE; j++)
    {
      m = false;
      for (int k = 0; k < ALPSIZE; k++) m = m || trigrams.data[i][j][k] > 0;
      for (int k = 0; k < ALPSIZE; k++) mask[i][j][k] = m;
    }

  for (int i = 0; i < ALPSIZE; i++)
    for (int j = 0; j < ALPSIZE; j++)
    {
      m = false;
      for (int k = 0; k < ALPSIZE; k++) m = m || trigrams.data[k][j][i] > 0;
      for (int k = 0; k < ALPSIZE; k++) mask[k][j][i] = mask[k][j][i] && m;
    }

  int count = 0;
  for (int i = 0; i < ALPSIZE; i++)
    for (int j = 0; j < ALPSIZE; j++)
      for (int k = 0; k < ALPSIZE; k++)
        if (mask[i][j][k]) count++;

  float rate = (float)count / ALPSIZE_TO3;
  std::cout << rate;
}