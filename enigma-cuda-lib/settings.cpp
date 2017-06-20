/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include <string.h>
#include <iostream>
#include <fstream>
#include "vendor/getopt/getopt.h"
#include "settings.h"
#include "iterator.h"
#include "version.h"
#include "cuda_code.h"

const string help_string =
"\r\nusage:\r\n"
"  enigma-cuda.exe <options> <uni/bigram file name> <trigram file name> <ciphertext file name>\r\n"
"\r\noptions:\r\n"
"  - h             show this help\r\n"
"  - v             show version number\r\n"
"  - R             resume operation using the state saved in the 00hc.resume file\r\n"
"  - o <file_name> save output to file\r\n"
"  - M <model>     Enigma model : H, M3 or M4\r\n"
"  - f <key>       first key\r\n"
"  - t <key>       last key\r\n"
"  - x and -a      turnover mode\r\n"
"  - n <count>     number of passes to make\r\n"
"  - z <score>     stop at this score\r\n"
"  - s <plugs>     known plugboard connections, e.g. ABXY\r\n"
"  - e <letters>   exhaustive search with single fixed plug, e.g.ENRXSI\r\n"
"  - E <letters>   exhaustive search with multiple fixed plugs, e.g.EN\r\n"
"  - p             start with random swapping order\r\n"
"  - g <scores>    use scores: 0 = IC, 1 = unigrams, 2 = bigrams, 3 = trigrams. default = 023\r\n"
"  - icuwrmk       ignored but do not result in an error\r\n";


#define VALID_OPTIONS "hvicpxaRM:w:r:m:u:s:f:t:k:n:z:o:e:E:g:"


string OptionGToString(int value)
{
  string result = "";
  if (value & skIC) result.push_back('0');
  if (value & skUnigram) result.push_back('1');
  if (value & skBigram)  result.push_back('2');
  if (value & skTrigram) result.push_back('3');
  return result;
}

int OptionGFromString(const string & str)
{
  int result = 0;
  if (str.find("0") != string::npos) result |= skIC;
  if (str.find("1") != string::npos) result |= skUnigram;
  if (str.find("2") != string::npos) result |= skBigram;
  if (str.find("3") != string::npos) result |= skTrigram;
  return result;
}



void Settings::Clear()
{
    model = enigmaInvalid;
    first_key_string = "";
    last_key_string = "";
    turnover_modes = toBeforeMessage;
    passes_left = 1;
    stop_at_score = INT_MAX;
    exhaust_single_plugs = "";
    known_plugs = "";
    first_pass = true;
    score_kinds = skIC | skBigram | skTrigram;

    //out_file_name = "";
    //unigram_file_name = "";
    //bigram_file_name = "";
    //trigram_file_name = "";
    //ciphertext_file_name = "";

    current_key_string = "";
    best_score = 0;
    best_key_string = "";
    best_pluggoard_string = "";
}

bool Settings::FromCommandLine(int argc, char **argv)
{
    Clear();
    int opt;
    opterr = 0;
    bool resume = false;

    try
    {
        while ((opt = getopt(argc, argv, VALID_OPTIONS)) != -1)
        {
            switch (opt)
            {
            case 'h':
                std::cout << help_string;
                return false;

            case 'v':
                std::cout << VERSION;
                return false;

            case 'R':
                resume = true;
                break;

            case 'f':
                first_key_string = optarg;
                break;

            case 't':
                last_key_string = optarg;
                break;

            case 'e':
              exhaust_single_plugs = optarg;
              break;

            case 'E':
              exhaust_multi_plugs = optarg;
              if (exhaust_multi_plugs.length() != 2)
                throw std::invalid_argument(optarg + 
                  string(" (must be 2 letters)"));
              break;

            case 's':
                known_plugs = optarg;
                break;

            case 'x':
                turnover_modes = toDuringMessage;
                break;

            case 'a':
                turnover_modes = toBeforeMessage | toDuringMessage;
                break;

            case 'n':
                passes_left = std::stoi(optarg);
                break;

            case 'p':
                first_pass = false;
                break;

            case 'g':
              score_kinds = OptionGFromString(optarg);                
              break;

            case 'z':
                stop_at_score = std::stoi(optarg);
                break;

            case 'o': 
                out_file_name = optarg;
                break;

            case 'M': 
                model = StringToEnigmaModel(optarg);
                if (model == enigmaInvalid)
                    throw std::invalid_argument(optarg);
                break;

            case 'i':
            case 'c':
            case 'u':
            case 'w':
            case 'r':
            case 'm':
            case 'k':
                break;

            default:
                throw std::invalid_argument("-" + opt);
            }
        }

        if (argc - optind != 3)         
            throw std::invalid_argument("three file names required");

        trigram_file_name = argv[optind++];
        if (score_kinds | skUnigram) unigram_file_name = argv[optind++];
        else bigram_file_name = argv[optind++];
        ciphertext_file_name = argv[optind];
                
        //load the rest from resume file if enabled
        if (resume) return LoadResumeFile();

        return Validate();
    }

    catch (const std::exception & e)
    {
        std::cerr << "Invalid parameter(s): " << e.what() << std::endl;
        return false;
    }
}

bool Settings::Validate()
{
  try
  {
    //for compatibility
    if (model == enigmaM4)
    {
      if (first_key_string.length() > 2)
      {
        first_key_string[0] = tolower(first_key_string[0]);
        first_key_string[2] = tolower(first_key_string[2]);
      }
      if (last_key_string.length() > 2)
      {
        last_key_string[0] = tolower(last_key_string[0]);
        last_key_string[2] = tolower(last_key_string[2]);
      }
    }

    //defaults
    if (first_key_string == "") first_key_string = key_range_starts[model];
    if (last_key_string == "") last_key_string = key_range_ends[model];
    if (current_key_string == "") current_key_string = first_key_string;

    //single key
    single_key = first_key_string == last_key_string;

    ValidateKey(first_key_string);
    ValidateKey(last_key_string);

    ValidatePlugs(exhaust_single_plugs);

    //known plugs allow self-steckered letters 
    //and thus cannot be validated
    //ValidatePlugs(known_plugs);

    ValidateFile(trigram_file_name);
    ValidateFile(bigram_file_name);
    ValidateFile(ciphertext_file_name);

    return true;
  }

  catch (const std::invalid_argument & e)
  {
    std::cerr << "Invalid parameter: " << e.what() << std::endl;
    return false;
  }
  catch (const std::ifstream::failure & e)
  {
    std::cerr << "File not found: " << e.what() << std::endl;
    return false;
  }
}

void Settings::ValidateKey(const string & key_string)
{
    if (key_string == "") return;

    Key key;
    key.FromString(key_string, model);
    if (!key.IsValid()) throw std::invalid_argument(key_string);
}

void Settings::ValidatePlugs(const string & plugboard_string)
{
    string letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    for (int i = 0; i < plugboard_string.length(); i++)
    {
        size_t p = letters.find(plugboard_string[i]);
        if (p == string::npos)
            throw std::invalid_argument(plugboard_string);
        letters.erase(p, 1);
    }

}
    
void Settings::ValidateFile(const string & file_name)
    
{
    if (file_name == "") return;

    std::ifstream fs(file_name);
    if (!fs) throw std::ifstream::failure(GetAbsolutePath(file_name));
}

string Settings::ReadToken(std::ifstream & fs, char delim)
{
    string token;
    if (!std::getline(fs, token, delim)) 
        throw std::ifstream::failure("'=' not found");
    return token;
}

bool Settings::LoadResumeFile()
{
    try
    {
        Clear();

        std::ifstream fs(RESUME_FILE_NAME);
        if (!fs)
            throw std::ifstream::failure(GetAbsolutePath(RESUME_FILE_NAME));

        //first line
        string model_string = ReadToken(fs);
        model = StringToEnigmaModel(model_string);
        if (model == enigmaInvalid) 
            throw std::invalid_argument("model " + model_string);

        first_key_string = ReadToken(fs);
        last_key_string = ReadToken(fs);
        current_key_string = ReadToken(fs);

        string token = ReadToken(fs);
        switch (token[0])
        {
        case '0': turnover_modes = toBeforeMessage; break;
        case '1': turnover_modes = toDuringMessage; break;
        case '2': turnover_modes = toBeforeMessage | toDuringMessage; break;
        default: throw std::invalid_argument("slow-wheel-mode " + token);
        }

        passes_left = std::stoi(ReadToken(fs));
        first_pass = ReadToken(fs) == "1";
        score_kinds = OptionGFromString(ReadToken(fs));
        stop_at_score = std::stoi(ReadToken(fs, '\n'));
       
        //second line
        if (ReadToken(fs) != model_string)
            throw std::invalid_argument("wrong model " + model_string);
        best_key_string = ReadToken(fs);
        best_pluggoard_string = ReadToken(fs);
        best_score = std::stoi(ReadToken(fs));

        return Validate();
    }
    catch (const std::exception & e)
    {
        std::cerr << "Cannot read Resume file: " << e.what() << std::endl;
        return false;
    }
}

void Settings::SaveResumeFile()
{
    try
    {
        std::ofstream fs(RESUME_FILE_NAME);
        if (!fs)
            throw std::ofstream::failure(GetAbsolutePath(RESUME_FILE_NAME));

        fs << EnigmaModelToString(model) << "=";
        fs << first_key_string << "=";
        fs << last_key_string << "=";
        fs << current_key_string << "=";
        switch (turnover_modes)
        {
        case toBeforeMessage: fs << "0="; break;
        case toDuringMessage: fs << "1="; break;
        case toBeforeMessage | toDuringMessage: fs << "2="; break;
        }
        fs << passes_left << "=";
        fs << (first_pass ? "1" : "0") << "=";
        fs << OptionGToString(score_kinds) << "=";
        fs << stop_at_score << std::endl;
        fs << EnigmaModelToString(model) << "=";
        fs << best_key_string << "=";
        fs << best_pluggoard_string << "=";
        fs << best_score << std::endl;
    }
    catch (const std::exception & e)
    {
        std::cerr << "Cannot write to Resume file: " << e.what() << std::endl;
    }
}

bool Settings::IsEqual(const Settings & sett)
{
    return (
        out_file_name == sett.out_file_name &&
        model == sett.model &&
        first_key_string == sett.first_key_string &&
        last_key_string == sett.last_key_string &&
        turnover_modes == sett.turnover_modes &&
        passes_left == sett.passes_left &&
        stop_at_score == sett.stop_at_score &&
        exhaust_single_plugs == sett.exhaust_single_plugs &&
        known_plugs == sett.known_plugs &&
        first_pass == sett.first_pass &&
        score_kinds == sett.score_kinds &&
        unigram_file_name == sett.unigram_file_name &&
        bigram_file_name == sett.bigram_file_name &&
        trigram_file_name == sett.trigram_file_name &&
        ciphertext_file_name == sett.ciphertext_file_name &&
        current_key_string == sett.current_key_string &&
        best_score == sett.best_score &&
        best_key_string == sett.best_key_string &&
        best_pluggoard_string == sett.best_pluggoard_string);

    return false;
}
