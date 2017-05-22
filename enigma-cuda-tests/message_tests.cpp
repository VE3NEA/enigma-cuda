/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include<iostream>
#include "CppUnitTest.h"
#include "cuda_code.h"
#include "ngrams.h"
#include "segmenter.h"
#include "test_data.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace enigmacudatests
{
    TEST_CLASS(MessageTest)
    {
    public:
        TEST_METHOD(DecodeTest)
        {
            string text = DecodeMessage(cipher_string, solution_key_string,
                solution_plug_string);
            Assert::AreEqual(solution_text, text);

            text = DecodeMessage(cipher_EJRSB, key_EJRSB, plugs_EJRSB);
            Assert::AreEqual(text_EJRSB, text);

            text = DecodeMessage(cipher_PFCXY, key_PFCXY, plugs_PFCXY);
            Assert::AreEqual(text_PFCXY, text);

            text = DecodeMessage(cipher_AMERI, key_AMERI, plugs_AMERI);
            Assert::AreEqual(text_AMERI, text);

            text = DecodeMessage(cipher_M4_1, key_M4_1, plugs_M4_1);                        
            Assert::AreEqual(text_M4_1, text);
        }
    
        TEST_METHOD(SegmentTest)    
        {      
          const string  input_text =
            "einspqrzweizusammensetzenzusammenxsetzen";
          const string  correct_segmented_text =
            "EINS pqr ZWEI ZUSAMMENSETZEN ZUSAMMEN X SETZEN";

          WordSegmenter segmenter;
          segmenter.LoadWordsFromFile(WORDS_FILE);
          segmenter.FindSegmentation(input_text);
          string segmented_text = segmenter.GetSegmentedText(true);
          Assert::AreEqual(correct_segmented_text, segmented_text);
        }    
    };
}