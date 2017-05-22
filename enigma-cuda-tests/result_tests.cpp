/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include<iostream>
#include "CppUnitTest.h"
#include "test_helper.h"
#include "test_data.h"


using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace enigmacudatests
{
    TEST_CLASS(ResultsTest)
    {
    private:
        const int counts[4] = { 1, ALPSIZE, ALPSIZE_TO4, ALPSIZE_TO5 };

    public:
        TEST_METHOD(ReduceTest)
        {
            for (int i = 0; i < 4; i++)
            {
                MockResults(counts[i]);
                Result result = GetBestResult(counts[i]);
                Assert::AreEqual(counts[i], result.score);
            }

        }
    };
}