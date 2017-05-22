/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include "CppUnitTest.h"
#include "key.h"


using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace enigmacudatests
{
    Key h_settings{ enigmaHeeres, refB, rotNone, rotI, rotII, rotIII,
    0,0,0,0, 0,0,0,0  };


	Key m4_settings{ enigmaM4, refC_thin, rotGamma, rotVIII, rotVII, rotVI,
		0,0,1,2, 1,2,3,4 };

	Key slow_rings1_settings{ enigmaM4, refC_thin,  rotGamma, rotVIII, rotVII, 
		rotVI, 0, 24, 23, 22, 1, 2, 3, 4 };

	Key slow_rings2_settings{ enigmaM4, refC_thin,  rotGamma, rotVIII, rotVII, 
		rotVI, 25, 24, 23, 22, 1, 2, 3, 4 };

	const string h_key_str = "B:123:AA:AAA";
	const string m4_key_str = "c:g876:BC:BCDE";
	const string slow_rings1_key_str = "c:g876:YXW:BCDE";
	const string slow_rings2_key_str = "c:g876:ZYXW:BCDE";

	const string bad_keys_h[] = {
		//bad char
		"B:123:AB:CDü",
		//wrong key length
		"B:123:A:AAA",
		"B:123:AAA:AAAA",
		//wrong rotor_pos length
		"B:123:AA:AAAA",
		//wrong reflector
		"A:123:AB:CDE",
		//same rotor
		"B:223:AA:AAA",
		//rotor from another machine
		"b:123:AA:AAA",
		"B:623:AA:AAA",
		"B:g23:AA:AAA"
	};

	const string bad_keys_4[] = {
		//wrong key length
		"b:g123:AA:AAA",
		"b:g123:AAAAA:AAAAA",
		//wrong rotor order
		"b:1g23:AA:AAAA"
	};

	const string good_keys_h[] = {
		"B:123:AB:CDE",
		"C:345:AVW:XYZ"
	};

	const string good_keys_3[] = {
		"B:123:AB:CDE",
		"C:872:AVW:XYZ"
	};

	const string good_keys_4[] = {
		"b:b123:AB:CDEF",
		"b:b123:AAB:CDEF",
		"c:g823:AAAB:CDEF",
	};

	
	TEST_CLASS(KeyTest)
	{
	public:
		TEST_METHOD(ToNumTest)
		{
			Assert::AreEqual(int8_t(25), ToNum('Z'));
		};
		
		TEST_METHOD(ToStringTest)
		{
			Assert::AreEqual(h_key_str, h_settings.ToString());
			Assert::AreEqual(m4_key_str, m4_settings.ToString());
			Assert::AreEqual(slow_rings1_key_str, slow_rings1_settings.ToString());
			Assert::AreEqual(slow_rings2_key_str, slow_rings2_settings.ToString());
		}

		TEST_METHOD(FromStringTest)
		{
			Key key;

			key.FromString(h_key_str);
			Assert::IsTrue(h_settings == key);

			key.FromString(m4_key_str);
			Assert::IsTrue(m4_settings == key);

			key.FromString(slow_rings1_key_str);
			Assert::IsTrue(slow_rings1_settings == key);

			key.FromString(slow_rings2_key_str);
			Assert::IsTrue(slow_rings2_settings == key);
		}

		TEST_METHOD(IsValidTest)
		{
			Key key;

			for (string str : bad_keys_h) 
			{
				key.FromString(str, enigmaHeeres);
				Assert::IsFalse(key.IsValid());
			}

			for (string str : bad_keys_4) 
			{
				key.FromString(str, enigmaM4);
				Assert::IsFalse(key.IsValid());
			}

			for (string str : good_keys_h) 
			{
				key.FromString(str, enigmaHeeres);
				Assert::IsTrue(key.IsValid());
			}

			for (string str : good_keys_3) 
			{
				key.FromString(str, enigmaM3);
				Assert::IsTrue(key.IsValid());
			}

			for (string str : good_keys_4) 
			{
				key.FromString(str, enigmaM4);
				Assert::IsTrue(key.IsValid());
			}
		}

		TEST_METHOD(StepTest)
		{
			Key key;

			key.FromString("B:123:AA:AAA");
			for (int i = 0; i < 2000; i++) key.Step();
			Assert::AreEqual((string)"B:123:AA:DCY", key.ToString());

			key.FromString("b:b278:AA:AAAA");
			for (int i = 0; i < 1000; i++) key.Step();
			Assert::AreEqual((string)"b:b278:AA:AGEM", key.ToString());
		}
	};
}