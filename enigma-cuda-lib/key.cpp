/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include <sstream>
#include "key.h"
#include "wiring.h"
#include "cuda_code.h"


//enigma model names used on the command line
const string enigma_names[ENIGMA_MODEL_CNT]{ "<invalid>", "H", "M3", "M4" };

//letters used in a string representation of an enigma key
const string reflector_letters = "ABCbc";
const string rotor_letters = "x12345678bg";

//min/max reflector and rotor kind for the model
const string first_rotors[ENIGMA_MODEL_CNT] { "", "B:x111", "B:x111", "b:b111" };
const string last_rotors[ENIGMA_MODEL_CNT]  { "", "C:x555", "C:x888", "c:g888" };

extern int compare_key_stru(const ScramblerStructure & s1,
    const ScramblerStructure & s2)
{
    int result;

    if ((result = s1.model - s2.model) ||
        (result = s1.ukwnum - s2.ukwnum) ||
                              
        (result = s1.g_slot - s2.g_slot) ||
        (result = s1.l_slot - s2.l_slot) ||
        (result = s1.m_slot - s2.m_slot) ||
        (result = s1.r_slot - s2.r_slot)) 
        return result;

    return 0;
}


int compare_keys(const Key & k1, const Key & k2)
{
    int result = compare_key_stru(k1.stru, k2.stru);
    if (result != 0) return result;

	if ((result = k1.sett.g_ring - k2.sett.g_ring) ||
		(result = k1.sett.l_ring - k2.sett.l_ring) ||
		(result = k1.sett.m_ring - k2.sett.m_ring) ||
		(result = k1.sett.r_ring - k2.sett.r_ring) ||
                     
		(result = k1.sett.g_mesg - k2.sett.g_mesg) ||
		(result = k1.sett.l_mesg - k2.sett.l_mesg) ||
		(result = k1.sett.m_mesg - k2.sett.m_mesg) ||
		(result = k1.sett.r_mesg - k2.sett.r_mesg)) 
        return result;

	return 0;
}

EnigmaModel StringToEnigmaModel(const string & model_str)
{
	for (int idx = enigmaHeeres; idx <= enigmaM4; ++idx)
		if (enigma_names[idx] == model_str)	return EnigmaModel(idx);
	return enigmaInvalid;
}

string EnigmaModelToString(const EnigmaModel & model)
{
	return enigma_names[model];
}

bool Key::FromString(const string & key_str)
{
	EnigmaModel model = 
		((key_str[0] == 'b') || (key_str[0] == 'c')) ? enigmaM4 : enigmaM3;
	if (!FromString(key_str, model)) return false;

	if ((stru.model == enigmaM3) && (stru.l_slot <= rotV) &&
		(stru.m_slot <= rotV) && (stru.r_slot <= rotV))
		stru.model = enigmaHeeres;
	return true;
}

bool Key::FromString(const string & key_str, const EnigmaModel model)
{
	//key length
	stru.model = enigmaInvalid;
	if ((key_str.length() < (model == enigmaM4 ? 14 : 12) ||
		(key_str.length() > (model == enigmaM4 ? 16 : 13)))) return false;

	//enigma model
	stru.model = model;

	//rotor order
    stru.ukwnum = ReflectorType(reflector_letters.find(key_str[0]));
	int p = 2;
	stru.g_slot = RotorType((model == enigmaM4) ? rotor_letters.find(key_str[p++]) : 0);
	stru.l_slot = RotorType(rotor_letters.find(key_str[p++]));
	stru.m_slot = RotorType(rotor_letters.find(key_str[p++]));
	stru.r_slot = RotorType(rotor_letters.find(key_str[p++]));

	//ring positions: AA, AAA or AAAA
	p++;
	if (key_str[p + 4] == ':') { sett.g_ring = ToNum(key_str[p++]); sett.l_ring = ToNum(key_str[p++]); }
	else if (key_str[p + 3] == ':') { sett.g_ring = 0; sett.l_ring = ToNum(key_str[p++]); }
	else sett.g_ring = sett.l_ring = 0;
    sett.m_ring = ToNum(key_str[p++]);
    sett.r_ring = ToNum(key_str[p++]);

	//rotor positions
	p++;
	sett.g_mesg = (model == enigmaM4) ? ToNum(key_str[p++]) : 0;
	sett.l_mesg = ToNum(key_str[p++]);
	sett.m_mesg = ToNum(key_str[p++]);
	sett.r_mesg = ToNum(key_str[p++]);

	if (p != key_str.length()) stru.model = enigmaInvalid;

	return true;
}

string Key::ToString()
{
	std::stringstream result;

	//rotor order
	result << reflector_letters[stru.ukwnum] << ':';
	if (stru.model == enigmaM4) result << rotor_letters[stru.g_slot];
	result << rotor_letters[stru.l_slot];
	result << rotor_letters[stru.m_slot];
	result << rotor_letters[stru.r_slot] << ':';

	//ring positons
	if (sett.g_ring > 0) result << ToChar(sett.g_ring) << ToChar(sett.l_ring);
	else if (sett.l_ring > 0) result << ToChar(sett.l_ring);
	result << ToChar(sett.m_ring) << ToChar(sett.r_ring) << ':';

	//rotor positions
	if (stru.model == enigmaM4) result << ToChar(sett.g_mesg);
	result << ToChar(sett.l_mesg) << ToChar(sett.m_mesg) << ToChar(sett.r_mesg);

	return result.str();
}

bool Key::IsValid()
{
	//enigma model
	if (stru.model == enigmaInvalid) return false;

	//reflector kind
	if (stru.ukwnum < reflector_letters.find(first_rotors[stru.model][0]) ||
        stru.ukwnum > reflector_letters.find(last_rotors[stru.model][0])) return false;

	//rotor kinds
	if (stru.g_slot < rotor_letters.find(first_rotors[stru.model][2]) ||
        stru.g_slot > rotor_letters.find(last_rotors[stru.model][2])  ||

        stru.l_slot < rotor_letters.find(first_rotors[stru.model][3]) ||
        stru.l_slot > rotor_letters.find(last_rotors[stru.model][3])  ||

        stru.m_slot < rotor_letters.find(first_rotors[stru.model][4]) ||
        stru.m_slot > rotor_letters.find(last_rotors[stru.model][4])  ||

        stru.r_slot < rotor_letters.find(first_rotors[stru.model][5]) ||
        stru.r_slot > rotor_letters.find(last_rotors[stru.model][5]) ) return false;

	//all rotors different
	if (stru.l_slot == stru.m_slot ||
        stru.l_slot == stru.r_slot ||
        stru.m_slot == stru.r_slot) return false;

	//ring positions
	if (sett.g_ring < 0 || sett.g_ring >= ALPSIZE ||
		sett.l_ring < 0 || sett.l_ring >= ALPSIZE ||
		sett.m_ring < 0 || sett.m_ring >= ALPSIZE ||
		sett.r_ring < 0 || sett.r_ring >= ALPSIZE) return false;

	//rotor positions
	if (sett.g_mesg < 0 || sett.g_mesg >= ALPSIZE ||
		sett.l_mesg < 0 || sett.l_mesg >= ALPSIZE ||
		sett.m_mesg < 0 || sett.m_mesg >= ALPSIZE ||
		sett.r_mesg < 0 || sett.r_mesg >= ALPSIZE) return false;

	return true;
}

void Key::Step()
{
  //left wheel turnover
  if (sett.m_mesg == wiring.notch_positions[stru.m_slot][0] ||
    sett.m_mesg == wiring.notch_positions[stru.m_slot][1])
  {
    sett.l_mesg = mod26(++sett.l_mesg);
    sett.m_mesg = mod26(++sett.m_mesg);
  }

  //middle wheel turnover
  else if (sett.r_mesg == wiring.notch_positions[stru.r_slot][0] ||
    sett.r_mesg == wiring.notch_positions[stru.r_slot][1])
  {
    sett.m_mesg = mod26(++sett.m_mesg);
  }

  //right wheel turnover
  sett.r_mesg = mod26(++sett.r_mesg);
}

void Key::StepBack()
{
  int8_t p = mod26(sett.r_mesg - 1);
  bool m_turn = 
    p == wiring.notch_positions[stru.r_slot][0] ||
    p == wiring.notch_positions[stru.r_slot][1];

  p = mod26(sett.m_mesg - 1);
  bool l_turn = 
    p == wiring.notch_positions[stru.m_slot][0] ||          
    p == wiring.notch_positions[stru.m_slot][1];

  if (l_turn) sett.l_mesg = mod26(--sett.l_mesg);
  if (m_turn || l_turn) sett.m_mesg = mod26(--sett.m_mesg);

  sett.r_mesg = mod26(--sett.r_mesg);
}

int Key::GetScramblerIndex()
{
    int8_t r = mod26(sett.r_mesg - sett.r_ring);
    int8_t m = mod26(sett.m_mesg - sett.m_ring);
    int8_t l = mod26(sett.l_mesg - sett.l_ring);
    return l * ALPSIZE_TO2 + m * ALPSIZE + r;
}

bool RotorSettings::inc()
{
  if (++r_mesg < ALPSIZE) return true;
  r_mesg = 0;

  if (++m_mesg < ALPSIZE) return true;
  m_mesg = 0;

  if (++l_mesg < ALPSIZE) return true;
  l_mesg = 0;

  return false;
}

void RotorSettings::FromInt(int value, int digits)
{
  if (digits > 4) m_ring = (value / ALPSIZE_TO4) % ALPSIZE;
  if (digits > 3) r_ring = (value / ALPSIZE_TO3) % ALPSIZE;

  l_mesg = (value / ALPSIZE_TO2) % ALPSIZE;
  m_mesg = (value / ALPSIZE) % ALPSIZE;
  r_mesg = value % ALPSIZE;
}
