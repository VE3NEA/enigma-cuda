/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include<iostream>
#include "CppUnitTest.h"
#include "key.h"
#include "util.h"
#include "wiring.h"
#include "cuda_code.h"


using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace enigmacudatests
{
	TEST_CLASS(ScramblerTest)
	{
    private:
        //returns a 26-element mapping table from scrambler as string
        string ScramblerEntryString(PitchedArray& scrambler, int l, int m, int r)
        {
            int8_t* start = scrambler.data + scrambler.pitch *
                (l * ALPSIZE_TO2 + m * ALPSIZE + r);
            std::vector<int8_t> entry(start, start + ALPSIZE);
            return NumbersToText(entry);
        }


	public:
		TEST_METHOD(Mod26Test)
		{
			const int8_t mod = 11;
			Assert::AreEqual(int8_t(0), mod26(0));
			Assert::AreEqual(int8_t(0), mod26(ALPSIZE));
			Assert::AreEqual(int8_t(0), mod26(-ALPSIZE));
			Assert::AreEqual(mod, mod26(2 * ALPSIZE + mod));
			Assert::AreEqual(mod, mod26(-2 * ALPSIZE + mod));
		}

		TEST_METHOD(TableTest)
		{
			Key key;
            PitchedArray scrambler;
			scrambler.data = new int8_t[ALPSIZE_TO4];
            scrambler.pitch = ALPSIZE;
            SetUpScramblerMemory();
            
			key.FromString("B:123:AA:AAA");
            GenerateScrambler(key);
            CopyScramblerToHost(scrambler);
			Assert::AreEqual((string)"UEJOBTPZWCNSRKDGVMLFAQIYXH", ScramblerEntryString(scrambler, 0,0,0));
			Assert::AreEqual((string)"CJARYKLZQBFGXUSWIDOVNTPMEH", ScramblerEntryString(scrambler, 2, 3, 4));
			Assert::AreEqual((string)"XZNTGHEFYOLKSCJQPUMDRWVAIB", ScramblerEntryString(scrambler, 25,25,25));

			key.FromString("b:g568:AA:AAAA");
			GenerateScrambler(key);
			CopyScramblerToHost(scrambler); 
			Assert::AreEqual((string)"XPVLMKTIHNFDEJSBZWOGYCRAUQ", ScramblerEntryString(scrambler, 0, 0, 0));
			Assert::AreEqual((string)"XKDCUVNYPZBSOGMIWTLREFQAHJ", ScramblerEntryString(scrambler, 15, 20, 25));

			key.FromString("b:g568:AA:CAAA");
			GenerateScrambler(key);
			CopyScramblerToHost(scrambler);
			Assert::AreEqual((string)"ZUJOMKQPTCFYERDHGNVIBSXWLA", ScramblerEntryString(scrambler, 0, 0, 0));

			key.FromString("b:g568:CAAA:AAAA");
			GenerateScrambler(key);
			CopyScramblerToHost(scrambler);
			Assert::AreEqual((string)"CSAVKQZNMTEUIHRXFOBJLDYPWG", ScramblerEntryString(scrambler, 0, 0, 0));
		}


		TEST_METHOD(IndexTest) 
		{
			string err_key_str;
			Key start_key;
			start_key.FromString("B:678:AAA:AAA");
			int n = 0;

			for (int m_slot = rotI; m_slot <= rotVIII; m_slot++)
				for (int r_slot = rotI; r_slot <= rotVIII; r_slot++)
					if (r_slot != m_slot)
					{
						int l_slot = rotI;
						if (l_slot == m_slot || l_slot == r_slot) l_slot++;
						if (l_slot == m_slot || l_slot == r_slot) l_slot++;

						start_key.stru.l_slot = (RotorType)l_slot;
						start_key.stru.m_slot = (RotorType)m_slot;
						start_key.stru.r_slot = (RotorType)r_slot;

                        //try zero and non-zero ring positions
                        for (int8_t r_ring = 0; r_ring < 2; r_ring++)
                            //try all rotor positions
                            for (int m_mesg = 0; m_mesg < ALPSIZE; m_mesg++)
							    for (int r_mesg = 0; r_mesg < ALPSIZE; r_mesg++)
							    {
                                    start_key.sett.r_ring = r_ring;
                                    start_key.sett.r_mesg = r_mesg;
                                    start_key.sett.m_mesg = m_mesg;
                                    Key key = start_key;

								    for (int i = 0; i < 26 * 26 + 1; i++)
								    {
									    key.Step();
									    int expected_idx = key.GetScramblerIndex();
									    int actual_idx = ComputeScramblerIndex(i, start_key.stru, start_key.sett, wiring);

									    if (actual_idx != expected_idx)
									    {
										    n++;
										    break;
									    }
								    }
							    }
					}

			Assert::AreEqual(0, n);
		}
	};
}