/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include "CppUnitTest.h"
#include "util.h"
#include "cuda_code.h"
#include "settings.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace enigmacudatests
{
    const string cmd_string = "enigma-cuda.exe -M M3"
        " -f B:124:AA:AAA -t B:124:ZZ:ZZZ -x -n 3 -z 2147483646"
        " -e ENRXSI -E XY -p -g 23"
        " -o output.txt 00trigr.gen 00unigr.gen PBNXA.txt";

    const Settings cmd_settings = { "output.txt", enigmaM3, "B:124:AA:AAA",
      "B:124:ZZ:ZZZ", toDuringMessage, 3, 2147483646 ,  "", "ENRXSI", "XY",
      false, skBigram | skTrigram, "00unigr.gen", "", "00trigr.gen", 
      "PBNXA.txt", "B:124:AA:AAA" };
    

    const string resume_text =
        "M3=B:124:AA:AAA=B:124:ZZ:ZZZ=B:124:ZZ:ZZZ=0=1=1=2147483646\n"
        "M3=B:124:YP:HEJ=BJCDFNGZHIKLMUOVPXQYRWST=30884\n";

    const Settings resume_settings = { "", enigmaM3, "B:124:AA:AAA",
        "B:124:ZZ:ZZZ", toBeforeMessage, 1, 2147483646 , "", "", "", true,
      skIC | skBigram | skTrigram, "", "", "", "",
        "B:124:ZZ:ZZZ", 30884, "B:124:YP:HEJ", "BJCDFNGZHIKLMUOVPXQYRWST" };


    TEST_CLASS(SettingsTest)
    {
    private:
        char char_buffer[1000];
        char * ptr_buffer[100];
        int test_argc;
        char** test_argv = &ptr_buffer[0];
        
        //convert command line string to argc, argv
        void StringToArgcArgv(const string & cmdline_string)
        {
            strcpy_s(char_buffer, cmdline_string.c_str());
            test_argv[0] = &char_buffer[0];
            test_argc = 1;
            for (int i=1; i < cmdline_string.length()-1; i++)
                if (char_buffer[i] == ' ')
                {
                    char_buffer[i] = char(0);
                    test_argv[test_argc++] = &char_buffer[i + 1];
                }
        }

    public:
        TEST_METHOD(CmdLineTest)
        {
            Settings sett;

            StringToArgcArgv(cmd_string);
            sett.FromCommandLine(test_argc, test_argv);
            Assert::IsTrue(sett.IsEqual(cmd_settings));
        }

        TEST_METHOD(ResumeReadTest)
        {
            SaveTextToFile(resume_text, RESUME_FILE_NAME);
            Settings sett;
            sett.LoadResumeFile();
            Assert::IsTrue(sett.IsEqual(resume_settings));
        }

        TEST_METHOD(ResumeWriteTest)
        {
            Settings sett = resume_settings;
            sett.SaveResumeFile();
            string saved_text = LoadTextFromFile(RESUME_FILE_NAME);
            remove(RESUME_FILE_NAME);
            Assert::AreEqual(resume_text, saved_text);
        }
    };
}