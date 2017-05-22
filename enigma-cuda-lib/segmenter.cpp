/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include <fstream>
#include <iterator>
#include <algorithm>
#include <sstream>
#include "segmenter.h"

//words: http://corpus.leeds.ac.uk/frqc/internet-de-forms.num

string StringToLowerCase(const string & text)
{
  string result = text;
  for (int i = 0; i < result.length(); ++i) result[i] = tolower(result[i]);
  return result;
}

string StringToUpperCase(const string & text)
{
  string result = text;
  for (int i = 0; i < result.length(); ++i) result[i] = toupper(result[i]);
  return result;
}

void WordSegmenter::LoadWordsFromFile(const string & file_name)
{
    words.clear();
    total_count = 0;
    max_length = 0;

    string word;
    int count;
    std::ifstream infile;

    //default file name
    infile.open((file_name == "") ? "words.txt" : file_name);
    //do not complain if file not found
    if (!infile.is_open()) return;

    //load
    while (infile >> word >> count)
    {
        word = StringToLowerCase(word);
        words[word] = count;
        max_length = std::max(max_length, (int)word.length());
        total_count += count;
    }
    
    //take ln
    for (auto iter = words.begin(); iter != words.end(); ++iter)
        iter->second = log(iter->second / total_count);
}

double WordSegmenter::GetWordScore(const string & word)
{
    //known word
    auto iter = words.find(word);
    if (iter != words.end()) return iter->second;
 
    //unknown word
    return -log(total_count) * pow(word.length(), 0.99);
}


#define IMPROBABLE -9999

int WordSegmenter::FindSegmentation(const string & text)
{
  input_text = text;

  if (!IsLoaded()) return 0;

  length_probs[0] = 0;

  for (int pos = 0; pos < text.length(); ++pos)
  {
    for (int len = max_length; len > 0; --len) 
      length_probs[len] = length_probs[len - 1];
    length_probs[0] = IMPROBABLE;

    for (int len = 1; len <= (pos + 1) && len <= max_length; ++len)
    {
      double prob = length_probs[len] +
        GetWordScore(text.substr(pos - len + 1, len));

      if (prob >= length_probs[0])
      {
        best_lengths[pos] = len;
        length_probs[0] = prob;
      }
    }
  }

  return length_probs[0];
}

//with_spaces is honoured only if up_case is true
string WordSegmenter::GetSegmentedText(bool up_case, bool with_spaces)
{
  if (!IsLoaded()) return input_text;
  if (input_text == "") return "";

  string result = input_text;
  int pos = (int)input_text.length();

  while (true)
  {
    int len = best_lengths[pos-1];
    pos -= len;
    if (pos <= 0) break;
    result.insert(pos, 1, ' ');
  }

  if (up_case) result = UpCaseKnownWords(result, with_spaces);

  return result;
}

string WordSegmenter::UpCaseKnownWords(const string & text, bool with_spaces)
{
  std::stringstream is(text);
  std::stringstream os;
  string s;
  
  while (is >> s)
  {
    os << (IsWord(s) ? StringToUpperCase(s) : s);
    if (with_spaces) os << " ";
  }
  s = os.str();

  if (with_spaces) s = s.substr(0, s.length()-1);
  return s;
}


bool WordSegmenter::IsWord(const string & word)
{
  return words.find(word) != words.end();
}