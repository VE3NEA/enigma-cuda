/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include <fstream>
#include "ngrams.h"
#include "util.h"
#include "key.h"


//TODO: add parameter bool scale = true
void Unigrams::LoadFromFile(const string & file_name)
{
	char c;
	int count;

	std::ifstream infile(file_name);
    if (!infile) 
        throw std::runtime_error(string("Cannot open file: ") + GetAbsolutePath(file_name));

    int temp_data[ALPSIZE] = { 0 };

	while (infile >> c >> count) temp_data[ToNum(toupper(c))] = count;

    //scale n-gram scores for easier debugging
	int mx = 0;
	for (int i = 0; i < ALPSIZE; ++i)				
        if (temp_data[i] > mx) mx = temp_data[i];

	for (int i = 0; i < ALPSIZE; ++i)				
        //data[i] = round((temp_data[i] * 255) / (float)mx);    
        data[i] = temp_data[i];
}

void Bigrams::LoadFromFile(const string & file_name)
{
    string s;
    int count;
    std::ifstream infile(file_name);
    if (!infile) throw std::runtime_error(string("File not found: ") + GetAbsolutePath(file_name));

    int temp_data[ALPSIZE][ALPSIZE] = { 0 };

    while (infile >> s >> count)
        temp_data[ToNum(toupper(s[0]))][ToNum(toupper(s[1]))] = count;

    int mx = 0;
    for (int i = 0; i < ALPSIZE; ++i)
        for (int j = 0; j < ALPSIZE; ++j)
                if (temp_data[i][j] > mx) mx = temp_data[i][j];

    for (int i = 0; i < ALPSIZE; ++i)
        for (int j = 0; j < ALPSIZE; ++j)
                //data[i][j] = round((temp_data[i][j] * 255) / (float)mx);    
                data[i][j] = temp_data[i][j];
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
	string s;
	int count;
	std::ifstream infile(file_name);
	if (!infile) throw std::runtime_error(string("File not found: ") + GetAbsolutePath(file_name));

    //temp_data is int, data might be uint8_t if it is faster
    int temp_data[ALPSIZE][ALPSIZE][ALPSIZE] = { 0 };
    
    while (infile >> s >> count)
        temp_data[ToNum(toupper(s[0]))]
		    [ToNum(toupper(s[1]))]
	        [ToNum(toupper(s[2]))] = count;

	int mx = 0;
	for (int i = 0; i < ALPSIZE; ++i)
		for (int j = 0; j < ALPSIZE; ++j)
			for (int k = 0; k < ALPSIZE; ++k)
				if (temp_data[i][j][k] > mx) mx = temp_data[i][j][k];

    for (int i = 0; i < ALPSIZE; ++i)
		for (int j = 0; j < ALPSIZE; ++j)
			for (int k = 0; k < ALPSIZE; ++k)
                //data[i][j][k] = round((temp_data[i][j][k] * 255) / (float)mx);    
                data[i][j][k] = temp_data[i][j][k];
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