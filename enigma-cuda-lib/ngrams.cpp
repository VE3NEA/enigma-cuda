/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include <fstream>
#include "ngrams.h"
#include "util.h"
#include "key.h"


void FileErr(const string & msg, const string & file_name)
{
  throw std::runtime_error(msg + GetAbsolutePath(file_name));
}

void Unigrams::LoadFromFile(const string & file_name)
{
	string ngram;
	int score;

  memset(data, 0, sizeof(data));

	std::ifstream infile(file_name);
  if (!infile) FileErr("Cannot open file: ", file_name);

  while (infile >> ngram >> score)
    if (infile.fail() || ngram.length() != 1) FileErr("Error in file: ", file_name);
    else data[ToNum(toupper(ngram[0]))] = score;

    if (!infile.eof()) FileErr("Error in file: ", file_name);
}

void Bigrams::LoadFromFile(const string & file_name)
{
  string ngram;
  int score;

  memset(data, 0, sizeof(data));

  std::ifstream infile(file_name);
  if (!infile) FileErr("Cannot open file: ", file_name);

  while (infile >> ngram >> score)
    if (infile.fail() || ngram.length() != 2) FileErr("Error in file: ", file_name);
    else data[ToNum(toupper(ngram[0]))][ToNum(toupper(ngram[1]))] = score;

    if (!infile.eof()) FileErr("Error in file: ", file_name);
}

int Bigrams::ScoreText(const string & text)
{
    int8_t nums[MAX_MESSAGE_LENGTH];
    int score = 0;

    for (int i = 0; i < text.length(); i++)
    {
        nums[i] = ToNum(toupper(text[i]));
        if (i > 0) score += data[nums[i - 1]][nums[i]];
    }
    return score;
}

void Trigrams::LoadFromFile(const string & file_name)
{
  string ngram;
  int score;

  memset(data, 0, sizeof(data));

  std::ifstream infile(file_name);
  if (!infile) FileErr("Cannot open file: ", file_name);

  while (infile >> ngram >> score)
    if (infile.fail() || ngram.length() != 3) FileErr("Error in file: ", file_name);
    else data[ToNum(toupper(ngram[0]))]
             [ToNum(toupper(ngram[1]))]
             [ToNum(toupper(ngram[2]))] = score;

    if (!infile.eof()) FileErr("Error in file: ", file_name);
}

int Trigrams::ScoreText(const string & text)
{
    int8_t nums[MAX_MESSAGE_LENGTH];
    int score = 0;

    for (int i = 0; i < text.length(); i++)
    {
        nums[i] = ToNum(toupper(text[i]));
        if (i > 1) score += data[nums[i - 2]][nums[i - 1]][nums[i]];
    }
    return score;
}