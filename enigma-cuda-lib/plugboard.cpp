/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include "plugboard.h"
#include "util.h"

void Plugboard::Reset()
{
  for (int c = 0; c < ALPSIZE; ++c)
  {
    plugs[c] = c;
    fixed[c] = false;
  }
}

void Plugboard::FromString(const string plugboard_string)
{
	Reset();
	for (int i = 0; i < plugboard_string.length(); i += 2)
	{
		plugs[ToNum(plugboard_string[i])] = ToNum(plugboard_string[i + 1]);
		plugs[ToNum(plugboard_string[i + 1])] = ToNum(plugboard_string[i]);
	}
}

void Plugboard::FromData(const int8_t * plugboard_data)
{
    Reset();
    if (plugboard_data == NULL) return;
    for (int i = 0; i < ALPSIZE; i++) plugs[i] = plugboard_data[i];
}

string Plugboard::ToString()
{
	string result;
	for (int8_t c = 0; c < ALPSIZE; ++c)
		if (plugs[c] > c)
			result = result + ToChar(c) + ToChar(plugs[c]);

		return result;
}


void Plugboard::Randomize()
{
  Plugboard pb;
  pb.RandomizeSwapOrder();
  string s(26, ' ');
  for (int i = 0; i < s.length(); ++i) s[i] = ToChar(pb.order[i]);
  FromString(s);
}


void Plugboard::InitializeSwapOrder(const string & ciphertext)
{
    int counts[ALPSIZE] = { 0 };

    for (int i = 0; i < ciphertext.size(); ++i)
        counts[ToNum(ciphertext[i])]++;

    for (int i = 0; i < ALPSIZE; ++i)
    {
        int m = 0;
        for (int j = 1; j < ALPSIZE; ++j)
            if (counts[j] > counts[m]) m = j;
        order[i] = m;
        counts[m] = -1;
    }
}

void Plugboard::RandomizeSwapOrder()
{
    string letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    for (int i = 0; i < ALPSIZE; i++)
    {
        int idx = rand() % letters.length();
        order[i] = ToNum(letters[idx]);
        letters.erase(idx, 1);
    }
}

void Plugboard::SetSwapOrder(const string & order_string)
{
  for (int i = 0; i < ALPSIZE; i++)
    order[i] = ToNum(order_string[i]);
}

string Plugboard::GetSwapOrder()
{
  string result;
  result.resize(ALPSIZE);
  for (int i = 0; i < ALPSIZE; i++) result[i] = ToChar(order[i]);
  return result;
}

void Plugboard::SetFixedPlugs(const string & plugs_string)
{
  FromString(plugs_string);
  for (int i = 0; i < ALPSIZE; i++)
    fixed[i] = plugs_string.find(ToChar(i)) != std::string::npos;
}

void Plugboard::ClearFixedPlugs()
{
  for (int i = 0; i < ALPSIZE; i++) fixed[i] = false;
}

string two_chars(char c1, char c2)
{
  string blank = "";
  return (c1 < c2) ? (blank + c1) + c2 : (blank + c2) + c1;
}

void Plugboard::StartExhaustive(const string & letters, bool single_plug)
{
  Reset();
  list.clear();
  if (letters == "") return;

  if (single_plug)
  {
    for (int i = 0; i < letters.length(); ++i)
      for (char c = 'A'; c <= 'Z'; ++c)
        list.push_back(two_chars(letters[i], c));
  }
  else
    //do not fix self-steckered letters and a1-a2 pair, just like in EPhi
    for (int i1 = 0; i1 < letters.length() - 1; ++i1)
      for (int i2 = i1 + 1; i2 < letters.length(); ++i2)
      {
        char a1 = letters[i1];
        char a2 = letters[i2];

        for (char b1 = 'A'; b1 <= 'Z'; ++b1)
          if (b1 != a1 && b1 != a2)
            for (char b2 = 'A'; b2 <= 'Z'; ++b2)
              if (b2 != a1 && b2 != a2 && b2 != b1)
              {
                string pair1 = two_chars(a1, b1);
                string pair2 = two_chars(a2, b2);
                list.push_back(pair1 < pair2 ? pair1 + pair2 : pair2 + pair1);
              }
      }

//    for (char c0 = 'A'; c0 <= 'Z'; ++c0)
//      for (char c1 = 'A'; c1 <= 'Z'; ++c1)
//      {
//        if (c1 == letters[0] || c1 == c0) continue;
//        if (c0 == letters[1]) 
//          list.push_back(two_chars(letters[0], letters[1]));
//        else
//        {
//          string pair0 = two_chars(letters[0], c0);
//          string pair1 = two_chars(letters[1], c1);
//          list.push_back(pair0 < pair1 ? pair0 + pair1 : pair1 + pair0);
//        }
//      }
  

  list.sort();
  list.unique();

  list.push_front("");
  NextExahustive();
}

bool Plugboard::NextExahustive()
{
  if (list.empty()) return false;
  list.pop_front();
  if (list.empty()) return false;

  SetFixedPlugs(list.front());
  return true;
}

string Plugboard::ExahustivePlugsToString()
{
  return list.empty() ? "" : list.front();
}

int Plugboard::Compare(const string plugboard_string)
{
  Plugboard pb;
  pb.FromString(plugboard_string);
  int result = 0;
  for (int i = 0; i < ALPSIZE; ++i) 
    if (pb.plugs[i] != plugs[i])    
      result++;
  return result;
}

