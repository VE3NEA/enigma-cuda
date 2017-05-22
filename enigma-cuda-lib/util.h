#pragma once

/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include <string>
#include <vector>
#include <time.h>
using std::string;

inline int8_t ToNum(char x)  { return x - 'A'; }
inline char ToChar(int8_t x) { return x + 'A'; }

extern std::vector<int8_t> TextToNumbers(const string& text);
extern string NumbersToText(const std::vector<int8_t> numbers);
extern string LoadTextFromConsole();
extern string LoadTextFromFile(const string& file_name);
extern void SaveTextToFile(const string & text, const string & file_name);
extern string LettersFromText(const string & text);
extern string LettersAndSpacesFromText(const string & text);
extern string GetAbsolutePath(const string & file_name);
extern string GetExeDir();
extern string TimeDiffString(clock_t clock);
extern string TimeString();
extern string LowerCase(const string & text);
extern string UpperCase(const string & text);
extern std::vector<string> ListFilesInDirectory(const string & directory);
extern void TextToClipboard(const std::string & text);