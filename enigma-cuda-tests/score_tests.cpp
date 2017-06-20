/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include<iostream>
#include "CppUnitTest.h"
#include "test_helper.h"
#include "test_data.h"
#include "ngrams.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace enigmacudatests
{
	TEST_CLASS(ScoreTest)
	{
    public:
        TEST_METHOD(IcScoreTest)
        {
            int score = ComputeIcScore(cipher_string, solution_key_string, "");
            Assert::AreEqual(1328, score);
        }

        TEST_METHOD(UniScoreTest)
        {
          int score = ComputeUniScore(cipher_string, solution_key_string,
            uni_plug_string, UNI_FILE);
          Assert::AreEqual(5603014, score);

          try
          {
            NgramsToDevice(BI_FILE, "", "");
            Assert::Fail();
          }
          catch (std::runtime_error e) {}
        }

        TEST_METHOD(BiScoreTest)
        {
            Bigrams bigrams;
            bigrams.LoadFromFile(BI_FILE_1943);

            int expected_score = bigrams.ScoreText(text_AMERI);
            int score = ComputeBiScore(cipher_AMERI, key_AMERI, plugs_AMERI, BI_FILE_1943);
            Assert::AreEqual(expected_score, score);

            expected_score = bigrams.ScoreText(text_EJRSB);
            score = ComputeBiScore(cipher_EJRSB, key_EJRSB, plugs_EJRSB, BI_FILE_1943);
            Assert::AreEqual(expected_score, score);
            
            try
            {
              NgramsToDevice("", UNI_FILE, "");
              Assert::Fail();
            }
            catch (std::runtime_error e) {}

            try
            {
              NgramsToDevice("", TRI_FILE, "");
              Assert::Fail();
            }
            catch (std::runtime_error e) {}
        }

        TEST_METHOD(TriScoreTest)
        {
            //compare to original program
            int score = ComputeTriScore(cipher_string, solution_key_string,
                tri_plug_string, TRI_FILE);
            Assert::AreEqual(1115880, score);

            //compare to golden standard
            Trigrams trigrams;
            trigrams.LoadFromFile(TRI_FILE_1943);

            int expected_score = trigrams.ScoreText(text_AMERI);
            score = ComputeTriScore(cipher_AMERI, key_AMERI, plugs_AMERI, TRI_FILE_1943);
            Assert::AreEqual(expected_score, score);

            expected_score = trigrams.ScoreText(text_EJRSB);
            score = ComputeTriScore(cipher_EJRSB, key_EJRSB, plugs_EJRSB, TRI_FILE_1943);
            Assert::AreEqual(expected_score, score);

            try
            {
              NgramsToDevice("", "", BI_FILE);
              Assert::Fail();
            }
            catch (std::runtime_error e) {}
        }
    };
}