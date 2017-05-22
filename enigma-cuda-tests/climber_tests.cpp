/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include "CppUnitTest.h"
#include "test_data.h"
#include "test_helper.h"
#include "plugboard.h"


using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace enigmacudatests
{
    TEST_CLASS(ClimberTest)
    {
    public:
        TEST_METHOD(SwapTest)
        {
            int score = RunTrySwap(cipher_string, solution_key_string, "", test_order, 2, 18, skIC, UNI_FILE, TRI_FILE);
            Assert::AreEqual(1382, score);

            score = RunTrySwap(cipher_string, solution_key_string, uni_plug_string, test_order, 2, 6, skUnigram, UNI_FILE, TRI_FILE);
            Assert::AreEqual(5617548, score);

            score = RunTrySwap(cipher_string, solution_key_string, tri_plug_string, test_order, 2, 18, skTrigram, UNI_FILE, TRI_FILE);
            Assert::AreEqual(1130973, score);
        }

        TEST_METHOD(MaxScoreTest)
        {
            int score;
            score = RunMaximizeScore(cipher_string, solution_key_string, "", test_order, skIC, UNI_FILE, TRI_FILE);
            Assert::AreEqual(2122, score);

            score = RunMaximizeScore(cipher_string, solution_key_string, uni_plug_string, test_order, skUnigram, UNI_FILE, TRI_FILE);
            Assert::AreEqual(6260064, score);

            score = RunMaximizeScore(cipher_string, solution_key_string, tri_plug_string, test_order, skTrigram, UNI_FILE, TRI_FILE);
            Assert::AreEqual(1513204, score);
        }

        TEST_METHOD(ClimbTest)
        {
            SetUpClimb(0, cipher_string, solution_key_string, test_order, 
              toBeforeMessage | toDuringMessage | toAfterMessage, UNI_FILE, TRI_FILE);
            Result result = RunClimb(0, cipher_string, "B:524:AA:AAA");
            Assert::AreEqual(1299190, result.score);

            Plugboard plugboard;
            plugboard.FromData(result.plugs);
            Assert::AreEqual((string)"ASBHCXDOEVGIKTLMPQUY", plugboard.ToString());


            //short message 
            SetUpClimb(0, cipher_EJRSB, key_EJRSB, test_order_EJRSB,
              toBeforeMessage | toDuringMessage | toAfterMessage, UNI_FILE, TRI_FILE);
            result = RunClimb(0, cipher_EJRSB, key_EJRSB);
            Assert::AreEqual(440861, result.score);
    
            plugboard.FromData(result.plugs);
            Assert::AreEqual((string)"BWDTEUFJHXILKSMOPRYZ", plugboard.ToString());
        }

        //break a real cipher, iterate over the right hand ring and all rotor positions
        TEST_METHOD(SolveTest)
        {
            SetUpClimb(4, cipher_string, solution_key_string, test_order, toBeforeMessage, UNI_FILE, TRI_FILE);
            Result result = RunClimb(4, cipher_string, "B:524:KX:XXX");
            Assert::AreEqual(solution_score, result.score);

            Plugboard plugboard;
            plugboard.FromData(result.plugs);
            Assert::AreEqual(solution_plug_string, plugboard.ToString());
        }
    };
}