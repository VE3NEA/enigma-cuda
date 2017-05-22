#pragma once

/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include <stdint.h>
#include <string>
#include <list>
#include "const.h"
using std::string;

class Plugboard
{
private:
  string letters_to_fix;
  int fixed_letter_no;
  char fixed_partner;
  std::list<string> list;
public:
  void Reset();
  void FromString(const string plugboard_string);
  void FromData(const int8_t * plugboard_data);
  string ToString();
  void Randomize();

  void InitializeSwapOrder(const string & ciphertext);
  void RandomizeSwapOrder();
  void SetSwapOrder(const string & order_string);

  void SetFixedPlugs(const string & plugs_string);
  void ClearFixedPlugs();
  void StartExhaustive(const string & letters, bool single_plug);
  bool NextExahustive();

  string FixedPlugsToString();

  int Compare(const string plugboard_string);

  int8_t plugs[ALPSIZE];
  int8_t order[ALPSIZE];
  bool   fixed[ALPSIZE];
};
