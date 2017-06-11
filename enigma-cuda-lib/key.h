#pragma once

/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include <string>
#include "const.h"
#include "util.h"

enum EnigmaModel { enigmaInvalid, enigmaHeeres, enigmaM3, enigmaM4, ENIGMA_MODEL_CNT };
enum RotorType { rotNone, rotI, rotII, rotIII, rotIV, rotV, rotVI, rotVII, rotVIII, rotBeta, rotGamma, ROTOR_TYPE_CNT };
enum ReflectorType { refA, refB, refC, refB_thin, refC_thin, REFLECTOR_TYPE_CNT };

enum TurnoverLocation { toBeforeMessage = 1, toDuringMessage = 2, toAfterMessage = 4 };

struct ScramblerStructure
{
    EnigmaModel model;
    ReflectorType ukwnum;

    RotorType g_slot;
    RotorType l_slot;
    RotorType m_slot;
    RotorType r_slot;
};

struct RotorSettings
{
    int g_ring;
    int l_ring;
    int m_ring;
    int r_ring;

    int g_mesg;
    int l_mesg;
    int m_mesg;
    int r_mesg;

    bool inc();
};

struct Key 
{
    ScramblerStructure stru;
    RotorSettings sett;

	bool FromString(const string & key_str);		
	bool FromString(const string & key_str, const EnigmaModel model);
	string ToString();
	bool IsValid();
	void Step();
  void StepBack();
  int GetScramblerIndex();
};

//just for fun
extern int compare_keys(const Key & k1, const Key & k2);
inline bool operator==(const Key& k1, const Key& k2) 
{ return compare_keys(k1, k2) == 0; }
inline bool operator!=(const Key& k1, const Key& k2) 
{ return compare_keys(k1, k2) != 0; }
inline bool operator> (const Key& k1, const Key& k2) 
{ return compare_keys(k1, k2) > 0; }
inline bool operator< (const Key& k1, const Key& k2) 
{ return compare_keys(k1, k2) < 0; }
inline bool operator>=(const Key& k1, const Key& k2) 
{ return compare_keys(k1, k2) >= 0; }
inline bool operator<=(const Key& k1, const Key& k2) 
{ return compare_keys(k1, k2) <= 0; }

extern int compare_key_stru(const ScramblerStructure & s1, 
    const ScramblerStructure & s2);
inline bool operator==(const ScramblerStructure & s1, 
    const ScramblerStructure & s2)
{ return compare_key_stru(s1, s2) == 0; }

extern EnigmaModel StringToEnigmaModel(const string & model_str);
extern string EnigmaModelToString(const EnigmaModel & model);