#pragma once

/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include "key.h"

#define RESUME_FILE_NAME "00hc.resume"

struct Settings
{
    //PRINT: -h -v
    //RESUME: -R

    //-o <file_name>
    string out_file_name;

    //-M H
    EnigmaModel model;

    //-f B:123:AA:AAA
    string first_key_string;

    //-t B:543:ZZ:ZZZ
    string last_key_string;

    //-a or -x
    int turnover_modes;

    //-n 3
    int passes_left;

    //-z 4365261
    int stop_at_score;

    //-s ABXY
    string known_plugs;

    //NEW: -e ENRXSI
    string exhaust_single_plugs;

    //NEW: -E EN
    string exhaust_multi_plugs;

    //NEW: -p
    bool first_pass;

    //NEW: -g 0123
    int score_kinds;

    //non-options[0]
    string unigram_file_name;
    string bigram_file_name;

    //non-options[1]
    string trigram_file_name;

    //non-options[2]
    string ciphertext_file_name;

    //IGNORED: -i -c -u -w -r -m -k -x

    //current state
    string current_key_string;
    int best_score;
    string best_key_string;
    string best_pluggoard_string;
    
    bool single_key;

    bool FromCommandLine(int argc, char **argv);
    bool LoadResumeFile();
    void SaveResumeFile();
    bool IsEqual(const Settings & sett);
private:
    void Clear();
    bool Validate();
    void ValidateKey(const string & key_string);
    void ValidatePlugs(const string & plugboard_string);
    void ValidateFile(const string & file_name);
    string ReadToken(std::ifstream & fs, char delim = '=');
};
