#pragma once

/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include "key.h"

#define NONE -1

struct Wiring {
	int8_t reflectors[REFLECTOR_TYPE_CNT][ALPSIZE];
	int8_t rotors[ROTOR_TYPE_CNT][ALPSIZE];
	int8_t reverse_rotors[ROTOR_TYPE_CNT][ALPSIZE];
	int8_t notch_positions[ROTOR_TYPE_CNT][2];
};

extern const Wiring wiring;