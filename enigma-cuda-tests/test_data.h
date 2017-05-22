#pragma once
#include <inttypes.h>
#include <string>
#include "const.h"
using std::string;

//all test data files should be in $(SolutionDir)\data

#define UNI_FILE "..\\..\\..\\data\\00unigr.gen"
#define  BI_FILE "..\\..\\..\\data\\00bigr.gen"
#define TRI_FILE "..\\..\\..\\data\\00trigr.gen"

#define BI_FILE_1943 "..\\..\\..\\data\\Bi_1941_ln.txt"
#define TRI_FILE_1943 "..\\..\\..\\data\\Tri_1941_ln.txt"

#define WORDS_FILE "..\\..\\..\\data\\words.txt"


extern const string cipher_string;
extern const string solution_text;

extern const string solution_key_string;
extern const string solution_plug_string;
extern const int solution_score;

extern const string uni_plug_string;
extern const string tri_plug_string;

extern const int8_t test_order[ALPSIZE];

extern const string cipher_PFCXY;
extern const string key_PFCXY;
extern const string plugs_PFCXY;
extern const string text_PFCXY;

extern const string cipher_AMERI;
extern const string key_AMERI;
extern const string plugs_AMERI;
extern const string text_AMERI;

extern const string cipher_EJRSB;
extern const string key_EJRSB;
extern const string plugs_EJRSB;
extern const string text_EJRSB;
extern const int8_t test_order_EJRSB[ALPSIZE];

extern const string cipher_M4_1;
extern const string key_M4_1;
extern const string plugs_M4_1;
extern const string text_M4_1;

