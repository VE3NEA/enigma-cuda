#pragma once

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
#include "cuda_code.h"
#include "cuda_code.cuh"
#include "cuda_err_check.h"
#include "plugboard.h"
#include "ngrams.h"
#include "iterator.h"


__constant__ int8_t d_ciphertext[MAX_MESSAGE_LENGTH];
__constant__ Wiring d_wiring;
__constant__ Key d_key;
__constant__ NGRAM_DATA_TYPE d_unigrams[ALPSIZE];
__constant__ NGRAM_DATA_TYPE d_bigrams[ALPSIZE][ALPSIZE];
__constant__ int8_t d_order[ALPSIZE];
__constant__ int8_t d_plugs[ALPSIZE];
__constant__ bool d_fixed[ALPSIZE];
Result * d_temp;

Task h_task;
 
 


//------------------------------------------------------------------------------
//                            scrambler
//------------------------------------------------------------------------------
__host__ __device__
int8_t mod26(const int16_t x)
{
  return (ALPSIZE * 2 + x) % ALPSIZE;
}

void SetUpScramblerMemory()
{
  //wiring to gpu
  CUDA_CHECK(cudaMemcpyToSymbol(d_wiring, &wiring, sizeof(Wiring)));

  //memory for scrambler
  CUDA_CHECK(cudaMallocPitch(&h_task.scrambler.data, &h_task.scrambler.pitch,
        28, ALPSIZE_TO3));
}

__global__
void GenerateScramblerKernel(const Task task)
{
  __shared__ const int8_t * reflector;

  __shared__ const int8_t * g_rotor;
  __shared__ const int8_t * l_rotor;
  __shared__ const int8_t * m_rotor;
  __shared__ const int8_t * r_rotor;

  __shared__ const int8_t * g_rev_rotor;
  __shared__ const int8_t * l_rev_rotor;
  __shared__ const int8_t * m_rev_rotor;
  __shared__ const int8_t * r_rev_rotor;

  __shared__ int8_t r_core_position;
  __shared__ int8_t m_core_position;
  __shared__ int8_t l_core_position;
  __shared__ int8_t g_core_position;

  __shared__ int8_t * entry;

  if (threadIdx.x == 0)
  {
    //wirings
    reflector = d_wiring.reflectors[d_key.stru.ukwnum];

    g_rotor = d_wiring.rotors[d_key.stru.g_slot];
    l_rotor = d_wiring.rotors[d_key.stru.l_slot];
    m_rotor = d_wiring.rotors[d_key.stru.m_slot];
    r_rotor = d_wiring.rotors[d_key.stru.r_slot];

    g_rev_rotor = d_wiring.reverse_rotors[d_key.stru.g_slot];
    l_rev_rotor = d_wiring.reverse_rotors[d_key.stru.l_slot];
    m_rev_rotor = d_wiring.reverse_rotors[d_key.stru.m_slot];
    r_rev_rotor = d_wiring.reverse_rotors[d_key.stru.r_slot];

    //core positions
    r_core_position = blockIdx.x;
    m_core_position = blockIdx.y;
    l_core_position = blockIdx.z;
    g_core_position = mod26(d_key.sett.g_mesg - d_key.sett.g_ring);

    //address of scrambler entry
    entry = task.scrambler.data + task.scrambler.pitch * (
      l_core_position *  ALPSIZE * ALPSIZE +
      m_core_position *  ALPSIZE +
      r_core_position);
  }
  __syncthreads();

  //scramble one char
  int8_t ch_in = threadIdx.x;
  int8_t ch_out = ch_in;

  ch_out = r_rotor[mod26(ch_out + r_core_position)] - r_core_position;
  ch_out = m_rotor[mod26(ch_out + m_core_position)] - m_core_position;
  ch_out = l_rotor[mod26(ch_out + l_core_position)] - l_core_position;

  if (d_key.stru.model == enigmaM4)
  {
    ch_out = g_rotor[mod26(ch_out + g_core_position)] - g_core_position;
    ch_out = reflector[mod26(ch_out)];
    ch_out = g_rev_rotor[mod26(ch_out + g_core_position)] - g_core_position;
  }
  else
  {
    ch_out = reflector[mod26(ch_out)];
  }

  ch_out = l_rev_rotor[mod26(ch_out + l_core_position)] - l_core_position;
  ch_out = m_rev_rotor[mod26(ch_out + m_core_position)] - m_core_position;
  ch_out = r_rev_rotor[mod26(ch_out + r_core_position)] - r_core_position;

  //char to scrambler
  entry[ch_in] = mod26(ch_out);
}

void GenerateScrambler(const Key & key)
{
  //key to gpu
  CUDA_CHECK(cudaMemcpyToSymbol(d_key, &key, sizeof(Key)));

  //block and grid dimensions
  dim3 dimBlock(ALPSIZE);
  dim3 dimGrid(ALPSIZE, ALPSIZE, ALPSIZE);

  //run kernel
  GenerateScramblerKernel << < dimGrid, dimBlock >> > (h_task);
  CUDA_CHECK(cudaDeviceSynchronize());
}

__host__ __device__
int ComputeScramblerIndex(int char_pos, 
  const ScramblerStructure & stru,
  const RotorSettings & sett, const Wiring & wiring)
{
  //retrieve notch info
  const int8_t * r_notch = wiring.notch_positions[stru.r_slot];
  const int8_t * m_notch = wiring.notch_positions[stru.m_slot];

  //period of the rotor turnovers
  int m_period = (r_notch[1] == NONE) ? ALPSIZE : HALF_ALPSIZE;
  int l_period = (m_notch[1] == NONE) ? ALPSIZE : HALF_ALPSIZE;
  l_period = --l_period * m_period;

  //current wheel position relative to the last notch
  int r_after_notch = sett.r_mesg - r_notch[0];
  if (r_after_notch < 0) r_after_notch += ALPSIZE;
  if (r_notch[1] != NONE && r_after_notch >= (r_notch[1] - r_notch[0]))
    r_after_notch -= r_notch[1] - r_notch[0];

  int m_after_notch = sett.m_mesg - m_notch[0];
  if (m_after_notch < 0) m_after_notch += ALPSIZE;
  if (m_notch[1] != NONE && m_after_notch >= (m_notch[1] - m_notch[0]))
    m_after_notch -= m_notch[1] - m_notch[0];

  //middle wheel turnover phase
  int m_phase = r_after_notch - 1;
  if (m_phase < 0) m_phase += m_period;

  //left wheel turnover phase
  int l_phase = m_phase - 1 + (m_after_notch - 1) * m_period;
  if (l_phase < 0) l_phase += l_period;

  //hacks
  if (m_after_notch == 0) l_phase += m_period;
  if (m_after_notch == 1 && r_after_notch == 1)
    l_phase -= l_period; //effectively sets l_phase to -1
  if (m_after_notch == 0 && r_after_notch == 0)
  {
    m_phase -= m_period;
    l_phase -= m_period;
    if (char_pos == 0) l_phase++;
  }

  //save debug info
  //	r_after_notch_display = r_after_notch;
  //	m_after_notch_display = m_after_notch;
  //	l_phase_display = l_phase;

  //number of turnovers
  int m_steps = (m_phase + char_pos + 1) / m_period;
  int l_steps = (l_phase + char_pos + 1) / l_period;

  //double step of the middle wheel
  m_steps += l_steps;

  //rotor core poistions to scrambling table index
  return mod26(sett.l_mesg - sett.l_ring + l_steps) * ALPSIZE_TO2 +
    mod26(sett.m_mesg - sett.m_ring + m_steps) * ALPSIZE +
    mod26(sett.r_mesg - sett.r_ring + char_pos + 1);
}

__host__ __device__
TurnoverLocation GetTurnoverLocation(const ScramblerStructure & stru,
  const RotorSettings sett, int ciphertext_length, const Wiring & wiring)
{
  //rotors with two notches
    if (stru.r_slot > rotV && sett.r_ring >= HALF_ALPSIZE) 
        return toAfterMessage;
    if (stru.m_slot > rotV && sett.m_ring >= HALF_ALPSIZE) 
        return toAfterMessage;

  //does the left hand rotor turn right before the message?
  int8_t l_core_before = mod26(sett.l_mesg - sett.l_ring);
    int8_t l_core_first = ComputeScramblerIndex(0, stru, sett, wiring)
        / ALPSIZE_TO2;
  if (l_core_first != l_core_before) return toBeforeMessage;

  //does it turn during the message?
    int8_t l_core_last = 
        ComputeScramblerIndex(ciphertext_length-1, stru, sett, wiring) 
        / ALPSIZE_TO2;
  if (l_core_last != l_core_first) return toDuringMessage;

  return toAfterMessage;
}


//move the relevant part of the scrambler from global to shared memory
//and shuffle it to avoid bank conflicts
__device__
const int8_t * ScramblerToShared(const int8_t * global_scrambling_table)
{
  //global: ALPSIZE bytes at sequential addresses
  const int32_t * src = 
    reinterpret_cast<const int32_t *>(global_scrambling_table);

  //shared: same bytes in groups of 4 at a stride of 128
  extern __shared__ int8_t shared_scrambling_table[];
  int32_t * dst = reinterpret_cast<int32_t *>(shared_scrambling_table);

  //copy ALPSIZE bytes as 7 x 32-bit words
  int idx = (threadIdx.x & ~31) * 7 + (threadIdx.x & 31);
  for (int i = 0; i < 7; ++i) dst[idx + 32 * i] = src[i];
  return &shared_scrambling_table[idx * 4];
}






//------------------------------------------------------------------------------
//                         constants to device
//------------------------------------------------------------------------------
void CipherTextToDevice(string ciphertext_string)
{
  std::vector<int8_t> cipher = TextToNumbers(ciphertext_string);
  int8_t * cipher_data = cipher.data();
  CUDA_CHECK(cudaMemcpyToSymbol(d_ciphertext, cipher_data, cipher.size()));
    h_task.count = (int)cipher.size();
}

void PlugboardStringToDevice(string plugboard_string)
{
  Plugboard plugboard;
  plugboard.FromString(plugboard_string);
  PlugboardToDevice(plugboard);
}
    
void PlugboardToDevice(const Plugboard & plugboard)
{        
  CUDA_CHECK(cudaMemcpyToSymbol(d_plugs, plugboard.plugs, ALPSIZE));
  CUDA_CHECK(cudaMemcpyToSymbol(d_fixed, plugboard.fixed, 
    sizeof(bool) * ALPSIZE));
}

void OrderToDevice(const int8_t * order)
{
  CUDA_CHECK(cudaMemcpyToSymbol(d_order, order, ALPSIZE));
}

void InitializeArrays(const string cipher_string, int turnover_modes,
  int score_kinds, int digits)
{
  //d_ciphertext
  CipherTextToDevice(cipher_string);

  //d_wiring
  SetUpScramblerMemory();

  //allow_turnover
  h_task.turnover_modes = turnover_modes;
  //use unigrams
  h_task.score_kinds = score_kinds;

  //d_results
  int count = (int)pow(ALPSIZE, digits);
  SetUpResultsMemory(count);
}







//------------------------------------------------------------------------------
//                               score
//------------------------------------------------------------------------------
void NgramsToDevice(const string & uni_filename,
  const string & bi_filename, const string & tri_filename)
{
  if (uni_filename != "")
  {
    Unigrams unigrams;
    unigrams.LoadFromFile(uni_filename);
    CUDA_CHECK(cudaMemcpyToSymbol(d_unigrams, unigrams.data, sizeof(d_unigrams)));
  }

  if (bi_filename != "")
  {
    Bigrams bigrams;
    bigrams.LoadFromFile(bi_filename);
    CUDA_CHECK(cudaMemcpyToSymbol(d_bigrams, bigrams.data, sizeof(d_bigrams)));
  }

  if (tri_filename != "")
  {
    //trigram data
    Trigrams trigrams_obj;
    trigrams_obj.LoadFromFile(tri_filename);

    //non-pitched array in device memory. slightly faster than pitched
    CUDA_CHECK(cudaMalloc(&h_task.trigrams.data, 
      sizeof(NGRAM_DATA_TYPE) * ALPSIZE_TO3));
    h_task.trigrams.pitch = sizeof(NGRAM_DATA_TYPE) * ALPSIZE;

    //data to device
    CUDA_CHECK(cudaMemcpy(h_task.trigrams.data, trigrams_obj.data,
      sizeof(NGRAM_DATA_TYPE) * ALPSIZE_TO3, cudaMemcpyHostToDevice));
  }
}

__device__
int8_t Decode(const int8_t * plugboard, const int8_t * scrambling_table)
{
  int8_t c = d_ciphertext[threadIdx.x];
  c = plugboard[c];
  c = scrambling_table[(c & ~3) * 32 + (c & 3)];
  c = plugboard[c];  
  return c;
}

//MIN_MESSAGE_LENGTH <= count <= MAX_MESSAGE_LENGTH
__device__
void Sum(int count, volatile int * data, int * sum)
{
  if ((threadIdx.x + 128) < count) data[threadIdx.x] += data[128 + threadIdx.x];
  __syncthreads();

  if (threadIdx.x < 64 && (threadIdx.x + 64) < count)
    data[threadIdx.x] += data[64 + threadIdx.x];
  __syncthreads();

  if (threadIdx.x < 32)
  {
    if ((threadIdx.x + 32) < count) data[threadIdx.x] += data[32 + threadIdx.x];
    if ((threadIdx.x + 16) < count) data[threadIdx.x] += data[16 + threadIdx.x];
    data[threadIdx.x] += data[8 + threadIdx.x];
    data[threadIdx.x] += data[4 + threadIdx.x];
    data[threadIdx.x] += data[2 + threadIdx.x];
    if (threadIdx.x == 0) *sum = data[0] + data[1];
  }
  __syncthreads();
}

#define HISTO_SIZE 32

__device__
void IcScore(Block & block, const int8_t * scrambling_table)
{
  //init histogram
  if (threadIdx.x < HISTO_SIZE) block.score_buf[threadIdx.x] = 0;
  __syncthreads();

  //compute histogram
  if (threadIdx.x < block.count)
  {
    int8_t c = Decode(block.plugs, scrambling_table);
    atomicAdd((int *)&block.score_buf[c], 1);
  }
  __syncthreads();

  //TODO: try lookup table here, ic[MAX_MESSAGE_LENGTH]
  if (threadIdx.x < HISTO_SIZE)
    block.score_buf[threadIdx.x] *= block.score_buf[threadIdx.x] - 1;

  //sum up
  if (threadIdx.x < HISTO_SIZE / 2)
  {
    block.score_buf[threadIdx.x] += block.score_buf[threadIdx.x + 16];
    block.score_buf[threadIdx.x] += block.score_buf[threadIdx.x + 8];
    block.score_buf[threadIdx.x] += block.score_buf[threadIdx.x + 4];
    block.score_buf[threadIdx.x] += block.score_buf[threadIdx.x + 2];
    if (threadIdx.x == 0) block.score = block.score_buf[0] + block.score_buf[1];
  }

  __syncthreads();
}


//TODO: put unigram table to shared memory
__device__
void UniScore(Block & block, const int8_t * scrambling_table)
{
  if (threadIdx.x < block.count)
  {
    int8_t c = Decode(block.plugs, scrambling_table);
    block.score_buf[threadIdx.x] = block.unigrams[c];
  }
  __syncthreads();

  Sum(block.count, block.score_buf, &block.score);
}

__device__
void BiScore(Block & block, const int8_t * scrambling_table)
{
  if (threadIdx.x < block.count)
    block.plain_text[threadIdx.x] = Decode(block.plugs, scrambling_table);
  __syncthreads();

  //TODO: trigrams are faster than bigrams. 
  //is it because trigrams are not declared as constants?
  //or because their index is computed explicitly?
  if (threadIdx.x < (block.count - 1))
    block.score_buf[threadIdx.x] = 
      d_bigrams[block.plain_text[threadIdx.x]]
               [block.plain_text[threadIdx.x + 1]];
  __syncthreads();

  Sum(block.count - 1, block.score_buf, &block.score);
}

//TODO: use bit mask in shared memory for non-zero elements 
//676 bit flags for first 2 letters in tirgram
//save ~ half global memory reads
__device__
void TriScore(Block & block, const int8_t * scrambling_table)
{
  //decode char
  if (threadIdx.x < block.count) 
    block.plain_text[threadIdx.x] = Decode(block.plugs, scrambling_table);
  __syncthreads();

  //look up scores
  if (threadIdx.x < (block.count - 2))
    block.score_buf[threadIdx.x] = block.trigrams[
      block.plain_text[threadIdx.x] * ALPSIZE_TO2 +
      block.plain_text[threadIdx.x + 1] * ALPSIZE +
      block.plain_text[threadIdx.x+2]];
  __syncthreads();

  Sum(block.count - 2, block.score_buf, &block.score);
}

__device__
void CalculateScore(Block & block, const int8_t * scrambling_table)
{
  switch (block.score_kind)
  {
  case skTrigram: TriScore(block, scrambling_table); break;
  case skBigram:  BiScore(block, scrambling_table); break;
  case skUnigram: UniScore(block, scrambling_table); break;
  case skIC:      IcScore(block, scrambling_table); break;
  }
}






//------------------------------------------------------------------------------
//                               climber
//------------------------------------------------------------------------------
__device__ void TrySwap(int8_t i, int8_t k,
    const int8_t * scrambling_table, Block & block)
{
  __shared__ int old_score;
    int8_t x, z;
  old_score = block.score;

  if (d_fixed[i] || d_fixed[k]) return;

  if (threadIdx.x == 0)
  {              
    x = block.plugs[i];
    z = block.plugs[k];

    if (x == k)
    {
      block.plugs[i] = i;
      block.plugs[k] = k;
    }
    else
    {
      if (x != i)
      {
        block.plugs[i] = i;
        block.plugs[x] = x;
      };
      if (z != k)
      {
        block.plugs[k] = k;
        block.plugs[z] = z;
      };
      block.plugs[i] = k;
      block.plugs[k] = i;
    }
  }
  __syncthreads();

  CalculateScore(block, scrambling_table);

  if (threadIdx.x == 0 && block.score <= old_score)
  {
    block.score = old_score;

    block.plugs[z] = k;
    block.plugs[x] = i;
    block.plugs[k] = z;
    block.plugs[i] = x;
  }
  __syncthreads();
}

__device__ void MaximizeScore(Block & block, const int8_t * scrambling_table)
{
  CalculateScore(block, scrambling_table);

  for (int p = 0; p < ALPSIZE - 1; p++)
    for (int q = p + 1; q < ALPSIZE; q++)
      TrySwap(d_order[p], d_order[q], scrambling_table, block);
}

__global__ void ClimbKernel(const Task task)
{
  __shared__ Block block;
  __shared__ RotorSettings sett;
  __shared__ bool skip_this_key;
  __shared__ Result * result;

  if (threadIdx.x < ALPSIZE)
  {
    block.plugs[threadIdx.x] = d_plugs[threadIdx.x];
    block.unigrams[threadIdx.x] = d_unigrams[threadIdx.x];
  }

  if (threadIdx.x == 0)
  {
    block.trigrams = reinterpret_cast<int*>(task.trigrams.data);
    block.count = task.count;

    //ring and rotor settings to be tried
    sett.g_ring = 0;
    sett.l_ring = 0;

    //depending on the grid size, ring positions 
    //either from grid index or fixed (from d_key)
    sett.m_ring = (gridDim.y > ALPSIZE) ? blockIdx.y / ALPSIZE : d_key.sett.m_ring;
    sett.r_ring = (gridDim.y > 1) ? blockIdx.y % ALPSIZE : d_key.sett.r_ring;

    sett.g_mesg = d_key.sett.g_mesg;
    sett.l_mesg = (gridDim.x > ALPSIZE_TO2) ? blockIdx.x / ALPSIZE_TO2 : d_key.sett.l_mesg;
    sett.m_mesg = (gridDim.x > ALPSIZE) ? (blockIdx.x / ALPSIZE) % ALPSIZE : d_key.sett.m_mesg;
    sett.r_mesg = (gridDim.x > 1) ? blockIdx.x % ALPSIZE : d_key.sett.r_mesg;

    //element of results[] to store the output 
    int linear_idx = blockIdx.z * ALPSIZE_TO2 + blockIdx.y * ALPSIZE + blockIdx.x;
    result = &task.results[linear_idx];
    result->index = linear_idx;

    skip_this_key = ((gridDim.x > 1) &&
      (GetTurnoverLocation(d_key.stru, sett, block.count, d_wiring)
        & task.turnover_modes) == 0);
  }
  __syncthreads();

  if (skip_this_key) return;

  const int8_t * scrambling_table;
  if (threadIdx.x < block.count)
    {
      scrambling_table = task.scrambler.data + 
      ComputeScramblerIndex(threadIdx.x, d_key.stru, sett, d_wiring) * 
        task.scrambler.pitch;
      scrambling_table = ScramblerToShared(scrambling_table);
    }

  //IC once
  if (task.score_kinds & skIC)
  {
    block.score_kind = skIC;
    MaximizeScore(block, scrambling_table);
  }

  //unigrams once
  if (task.score_kinds & skUnigram)
  {
    block.score_kind = skUnigram;
    MaximizeScore(block, scrambling_table);
  }

  //bigrams once
  if (task.score_kinds & skBigram)
  {
    block.score_kind = skBigram;
    MaximizeScore(block, scrambling_table);
  }

  //trigrams until convergence
  if (task.score_kinds & skTrigram)
  {
    block.score_kind = skTrigram;
    block.score = 0;
    int old_score;
    do
    {
      old_score = block.score;
      MaximizeScore(block, scrambling_table);
    } 
    while (block.score > old_score);
  }

  //copy plugboard solution to global results array;
  if (threadIdx.x < ALPSIZE) result->plugs[threadIdx.x] = block.plugs[threadIdx.x];
  if (threadIdx.x == 0) result->score = block.score;
}

Result Climb(int cipher_length, const Key & key, bool single_key)
{
    try
    {
        CUDA_CHECK(cudaMemcpyToSymbol(d_key, &key, sizeof(Key)));

        int grid_size = single_key ? 1 : ALPSIZE_TO3;
        int block_size = std::max(32, cipher_length);
        int shared_scrambler_size = ((cipher_length + 31) & ~31) * 28;

        ClimbKernel << <grid_size, block_size, shared_scrambler_size >> > (h_task);
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        return GetBestResult(ALPSIZE_TO3);
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







//------------------------------------------------------------------------------
//                             results
//------------------------------------------------------------------------------
#define REDUCE_MAX_THREADS 256

void SetUpResultsMemory(int count)
{
  CUDA_CHECK(cudaMalloc((void**)&h_task.results, count * sizeof(Result)));
}

__device__ void SelectHigherScore(Result & a, const Result & b)
{
  if (b.score > a.score)
  {
    a.index = b.index;
    a.score = b.score;
  }
}

__global__ void FindBestResultKernel(Result *g_idata, Result *g_odata, 
    unsigned int count)
{
  __shared__ Result sdata[REDUCE_MAX_THREADS];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;

  Result best_pair;
  if (i < count)
  {
    best_pair.index = g_idata[i].index;
    best_pair.score = g_idata[i].score;
  }
  else best_pair.score = 0;

  if (i + blockDim.x < count) SelectHigherScore(best_pair, g_idata[i + blockDim.x]);

  sdata[tid] = best_pair;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (tid < s)
    {
      SelectHigherScore(best_pair, sdata[tid + s]);
      sdata[tid] = best_pair;
    }
    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = g_idata[best_pair.index];
}

unsigned int nextPow2(unsigned int x)
{
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

void ComputeDimensions(int count, int & grid_size, int & block_size)
{
  block_size = (count < REDUCE_MAX_THREADS * 2) ? nextPow2((count + 1) / 2) : REDUCE_MAX_THREADS;
  grid_size = (count + (block_size * 2 - 1)) / (block_size * 2);
}

Result GetBestResult(int count)
{
  int grid_size, block_size;
  ComputeDimensions(count, grid_size, block_size);

    if (d_temp == NULL)    
        CUDA_CHECK(cudaMalloc((void **)&d_temp, grid_size * sizeof(Result)));

    FindBestResultKernel << < grid_size, block_size >> > 
        (h_task.results, d_temp, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


  int s = grid_size;
  while (s > 1)
  {
        CUDA_CHECK(cudaMemcpy(h_task.results, d_temp, s * sizeof(Result), 
            cudaMemcpyDeviceToDevice));
    ComputeDimensions(s, grid_size, block_size);
        FindBestResultKernel << < grid_size, block_size >> > 
            (h_task.results, d_temp, s);
        CUDA_CHECK(cudaGetLastError());
    s = (s + (block_size * 2 - 1)) / (block_size * 2);
  }

  Result result;
  CUDA_CHECK(cudaMemcpy(&result, d_temp, sizeof(Result), cudaMemcpyDeviceToHost));

  return result;
}






//------------------------------------------------------------------------------
//                                  util
//------------------------------------------------------------------------------
bool SelectGpuDevice(int req_major, int req_minor, bool silent)
{
  int best_device = 0;
  int num_devices;
  cudaDeviceProp prop;

  CUDA_CHECK(cudaGetDeviceCount(&num_devices));

  switch (num_devices)
  {
  case 0:
    std::cerr << "GPU not found. Terminating.";
    return false;

  case 1:
    break;

  default:
    int max_sm = 0;
    for (int device = 0; device < num_devices; device++)
    {
      CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
      if (prop.multiProcessorCount > max_sm)
      {
        max_sm = prop.multiProcessorCount;
        best_device = device;
      }
    }
  }

  CUDA_CHECK(cudaGetDeviceProperties(&prop, best_device));
  if (!silent)
  {
    std::cout << "Found GPU '" << prop.name << "' with compute capability ";
    std::cout << prop.major << "." << prop.minor << "." << std::endl;
  }

  if (prop.major < req_major || (prop.major == req_major && prop.minor < req_minor))
  {
    std::cerr << "Program requires GPU with compute capability ";
    std::cerr << req_major << "." << req_minor;
    std::cerr << " or higher." << std::endl << "Terminating.";
    return false;
  }

  CUDA_CHECK(cudaSetDevice(best_device));
  return true;
}

int8_t DecodeLetter(int8_t c, const Key & key, const int8_t * plugs)
{
  int8_t r = mod26(key.sett.r_mesg - key.sett.r_ring);
  int8_t m = mod26(key.sett.m_mesg - key.sett.m_ring);
  int8_t l = mod26(key.sett.l_mesg - key.sett.l_ring);
  int8_t g = mod26(key.sett.g_mesg - key.sett.g_ring);

  c = plugs[c];

  c = wiring.rotors[key.stru.r_slot][mod26(c + r)] - r;
  c = wiring.rotors[key.stru.m_slot][mod26(c + m)] - m;
  c = wiring.rotors[key.stru.l_slot][mod26(c + l)] - l;
  c = wiring.rotors[key.stru.g_slot][mod26(c + g)] - g;

  c = wiring.reflectors[key.stru.ukwnum][mod26(c)];

  c = wiring.reverse_rotors[key.stru.g_slot][mod26(c + g)] - g;
  c = wiring.reverse_rotors[key.stru.l_slot][mod26(c + l)] - l;
  c = wiring.reverse_rotors[key.stru.m_slot][mod26(c + m)] - m;
  c = wiring.reverse_rotors[key.stru.r_slot][mod26(c + r)] - r;

  return plugs[mod26(c)];
}

string DecodeMessage(const string & ciphertext, const string & key_string,
  const int8_t * plugs)
{
  Key key;
  key.FromString(key_string);

  string result = ciphertext;

  for (int i = 0; i < result.length(); i++)
  {
    key.Step();
    result[i] = ToChar(DecodeLetter(ToNum(result[i]), key, plugs));
  }

  return LowerCase(result);
}

string DecodeMessage(const string & ciphertext, const string & key_string,
  const string & plugboard_string)
{
  Plugboard plugboard;
  plugboard.FromString(plugboard_string);

  return DecodeMessage(ciphertext, key_string, plugboard.plugs);
}
