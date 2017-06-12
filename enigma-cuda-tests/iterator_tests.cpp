/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include "CppUnitTest.h"
#include "test_data.h"
#include "cuda_code.h"
#include "key.h"
#include "iterator.h"
#include "plugboard.h"


using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace enigmacudatests
{
    const string TurnOverLocationNames[] = { "BAD", "before", "in", "BAD", "after" };

    TEST_CLASS(IteratorTest)
    {
    private:
      TurnoverLocation GoldenGetTurnoverLocation(const ScramblerStructure & stru,
            const RotorSettings sett, int ciphertext_length)
        {
            Key key;
            key.stru = stru;
            key.sett = sett;

            key.Step();
            if (key.sett.l_mesg != sett.l_mesg) return toBeforeMessage;

            for (int i = 1; i < ciphertext_length; i++) key.Step();
            if (key.sett.l_mesg != sett.l_mesg) return toDuringMessage;

            return toAfterMessage;
        }

        void TestTurnovers(Key key)
        {
            for (int8_t m_ring = 0; m_ring < HALF_ALPSIZE; m_ring++)
                for (int8_t r_ring = 0; r_ring < HALF_ALPSIZE; r_ring++)
                    for (int8_t m_mesg = 0; m_mesg < ALPSIZE; m_mesg++)
                        for (int8_t r_mesg = 0; r_mesg < ALPSIZE; r_mesg++)
                        {
                            key.sett.m_ring = m_ring;
                            key.sett.r_ring = r_ring;
                            key.sett.m_mesg = m_mesg;
                            key.sett.r_mesg = r_mesg;

                            TurnoverLocation expected =
                                GoldenGetTurnoverLocation(
                                    key.stru, key.sett, 100);

                            TurnoverLocation computed =
                                GetTurnoverLocation(
                                    key.stru, key.sett, 100, wiring);

                            Assert::AreEqual(
                                key.ToString() + "=" + TurnOverLocationNames[expected],
                                key.ToString() + "=" + TurnOverLocationNames[computed]);
                        }
        }

    public:
        TEST_METHOD(TurnOverTest)
        {
            Key key;
            key.FromString("B:123:AAA:AAA");
            TestTurnovers(key);
            key.FromString("b:g876:AAAA:AAAA");
            TestTurnovers(key);
        }

        TEST_METHOD(RotorOrderTest)
        {
            KeyIterator iter;
            int count;

            //full range
            iter.SetFullRange(enigmaHeeres);
            count = 0;
            do count++; while (iter.NextRotorOrder());
            Assert::AreEqual(2 * 5 * 4 * 3, count);

            iter.SetFullRange(enigmaM3);
            count = 0;
            do count++; while (iter.NextRotorOrder());
            Assert::AreEqual(2 * 8 * 7 * 6, count);

            iter.SetFullRange(enigmaM4);
            count = 0;
            do count++; while (iter.NextRotorOrder());
            Assert::AreEqual(2 * 2 * 26 * 8 * 7 * 6, count);

            //subrange
            iter.SetRange("B:123:AA:AAA", "B:123:AA:AAA");
            count = 0;
            do count++; while (iter.NextRotorOrder());
            Assert::AreEqual(1, count);

            iter.SetRange("B:123:AA:AAA", "B:124:AA:AAA");
            count = 0;
            do count++; while (iter.NextRotorOrder());
            Assert::AreEqual(2, count);

            iter.SetRange("B:123:AA:AAA", "B:543:AA:AAA");
            count = 0;
            do count++; while (iter.NextRotorOrder());
            Assert::AreEqual(60, count);

            iter.SetRange("b:b123:AA:AAAA", "b:b123:AA:ZAAA");
            count = 0;
            do count++; while (iter.NextRotorOrder());
            Assert::AreEqual(26, count);

            //nested iteration
            iter.SetFullRange(enigmaHeeres);
            count = 0;
            do
            {
                do
                {
                    count++;
                } 
                while (iter.NextRingPosition());
            }
            while (iter.NextRotorOrder());
                    
            Assert::AreEqual(2 * 5 * 4 * 3 * 26 * 26, count);
        }

        TEST_METHOD(RingPositionTest)
        {
            KeyIterator iter;
            int count;

            //both wheels
            iter.SetFullRange(enigmaHeeres);
            count = 1;
            while (iter.NextRingPosition()) count++;
            Assert::AreEqual(26 * 26, count);

            //right wheel only
            iter.SetFullRange(enigmaHeeres);
            count = 1;
            while (iter.NextRingPosition(false)) count++;
            Assert::AreEqual(26, count);

            //AA..AM + BA, BB
            iter.SetRange("b:b678:AA:XXXX", "b:b678:BB:XXXX");
            count = 1;
            while (iter.NextRingPosition()) count++;
            Assert::AreEqual(13+2, count);
        }

        TEST_METHOD(NextKeyTest)
        {
            KeyIterator iter;
            //note r_ring Y auto replaced with M for rotor rotVII
            iter.SetRange("b:b167:AA:XXXX", "b:b167:BY:XXXX");
            int count = 0;
            do count++; while (iter.Next());
            Assert::AreEqual(13 * 2, count);
        }
        
        TEST_METHOD(FixedPlugsTest)
        {
          Plugboard plugboard;

          plugboard.StartExhaustive("ENRXSI", true);
          int count = 0;
          do count++; while (plugboard.NextExahustive());
          Assert::AreEqual(26 + 25 + 24 + 23 + 22 + 21, count);

//          plugboard.StartExhaustive("EN", false);
//          count = 0;
//          do count++; while (plugboard.NextExahustive());
//          Assert::AreEqual(
//            24 * 23 + //both letters steckered
//            24 * 2 +  //one self-steckered
//            1 +       //both self-steckered
//            1,        //steckered to each other
//            count);

          plugboard.StartExhaustive("ENR", false);
          count = 0;
          do count++; while (plugboard.NextExahustive());
          Assert::AreEqual(1587, count);
        }
    };
}