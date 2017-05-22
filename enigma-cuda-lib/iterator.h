#pragma once

/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include "key.h"

extern const string key_range_starts[ENIGMA_MODEL_CNT];
extern const string key_range_ends[ENIGMA_MODEL_CNT];

class KeyIterator
{
private:
    Key last_key;

    void ClearPositions();
    bool Inc(RotorType & rotor);
    void Inc(ReflectorType & reflector);
public:
    Key key;

    //g_ring, l_ring, l_mesg, m_mesg and r_mesg are ignored
    void SetRange(const string & first_key_string, 
        const string & last_key_string);
    void SetFullRange(EnigmaModel model);

    //iterate over a range of reflector/rotor selections 
    //and orders, and over g_mesg positions if M4
    bool NextRotorOrder();

    //iterate over all valid m_ring and r_ring positions.
    //if both_wheels = false, iterate only over m_ring positions
    bool NextRingPosition(bool both_wheels = true);

    bool Next();
};
