/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include "iterator.h"
#include "wiring.h"


const string key_range_starts[ENIGMA_MODEL_CNT]
    { "", "B:123:AA:XXX", "B:123:AA:XXX", "b:b123:AA:AXXX" };

const string key_range_ends[ENIGMA_MODEL_CNT]
    { "", "C:543:ZZ:XXX", "C:876:MM:XXX", "c:g876:MM:ZXXX" };


void KeyIterator::SetFullRange(EnigmaModel model)
{
    SetRange(key_range_starts[model], key_range_ends[model]);
}

//ring and rotor positions, except g_mesg, are ignored
void KeyIterator::SetRange(const string & first_key_string,
    const string & last_key_string)
{
    key.FromString(first_key_string);
    if (!key.IsValid())
        throw std::runtime_error("Invalid key: " + first_key_string);

    last_key.FromString(last_key_string);
    if (!last_key.IsValid())
        throw std::runtime_error("Invalid key: " + last_key_string);

    //promote Heer to M3: same machine, just wider wheel choice
    if (key.stru.model == enigmaHeeres && last_key.stru.model == enigmaM3)
        key.stru.model = enigmaM3;

    if (last_key.stru.model != key.stru.model)
        throw std::runtime_error("Incompatible keys: " + first_key_string +
            " and " + last_key_string);

    //enforce valid ring positions for rotors VI to VIII
    if (last_key.stru.r_slot > rotV && last_key.sett.r_ring >= HALF_ALPSIZE)
        last_key.sett.r_ring = HALF_ALPSIZE - 1;
    if (last_key.stru.m_slot > rotV && last_key.sett.m_ring >= HALF_ALPSIZE)
        last_key.sett.m_ring = HALF_ALPSIZE - 1;

    if (last_key_string != first_key_string) ClearPositions();
}

void KeyIterator::ClearPositions()
{
    key.sett.g_ring = last_key.sett.g_ring = 0;
    key.sett.l_ring = last_key.sett.l_ring = 0;

    key.sett.l_mesg = last_key.sett.l_mesg = 0;
    key.sett.m_mesg = last_key.sett.m_mesg = 0;
    key.sett.r_mesg = last_key.sett.r_mesg = 0;
}

bool KeyIterator::NextRingPosition(bool both_wheels)
{
    if (key == last_key) return false;

    int period;

    if (both_wheels)
    {
        period = (key.stru.r_slot > rotV) ? HALF_ALPSIZE : ALPSIZE;
        if (++key.sett.r_ring < period) return true;
        key.sett.r_ring = 0;
    }

    period = (key.stru.m_slot > rotV) ? HALF_ALPSIZE : ALPSIZE;
    if (++key.sett.m_ring < period) return true;
    key.sett.m_ring = 0;
    return false;
}

void KeyIterator::Inc(ReflectorType & reflector)
{
    if (key.stru.model == enigmaM4)
    {
        //in M4, the Greek wheel never rotates and may be viewed as part 
        //of the reflector, then M4 is just an M3 with 2*2*26 different reflectors
        if (++key.sett.g_mesg < ALPSIZE) return;
        key.sett.g_mesg = 0;
        key.stru.l_slot = rotI;
        key.stru.m_slot = rotII;
        key.stru.r_slot = rotIII;
    }

    key.stru.ukwnum = (ReflectorType)(key.stru.ukwnum + 1);
}

bool KeyIterator::Inc(RotorType & rotor)
{
    RotorType max_rotor = (key.stru.model == enigmaHeeres) ? rotV : rotVIII;

    rotor = (RotorType)(rotor + 1);
    if (rotor <= max_rotor) return true;

    rotor = rotI;
    return false;
}

//in M4, the Greek wheel never rotates and may be viewed as part 
//of the reflector, then M4 is just an M3 with 2*2*26 different reflectors
bool KeyIterator::NextRotorOrder()
{
    //rotor and ring positions, except g_mesg, 
    //are cleared on rotor order change
    ClearPositions();

    if (key.stru == last_key.stru && key.sett.g_mesg == last_key.sett.g_mesg) 
        return false;

    do
    {
        bool done = false;
        if (key.stru.model == enigmaM4)
        {
            done = ++key.sett.g_mesg < ALPSIZE;
            if (!done) key.sett.g_mesg = 0;
        }

        done = done || Inc(key.stru.r_slot);
        done = done || Inc(key.stru.m_slot);
        done = done || Inc(key.stru.l_slot);

        if (!done && key.stru.model == enigmaM4)
        {
            done = key.stru.g_slot == rotBeta;
            key.stru.g_slot = done ? rotGamma : rotBeta;
        }
        if (!done)     
            key.stru.ukwnum = (ReflectorType)(key.stru.ukwnum + 1);

    }        
    while (!key.IsValid());

    return true;
}

bool KeyIterator::Next()
{
    if (key == last_key) return false;
    return NextRingPosition() || NextRotorOrder();
}