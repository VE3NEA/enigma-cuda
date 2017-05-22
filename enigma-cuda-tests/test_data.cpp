/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include "test_data.h"

//from http://cryptocellar.org/bgac/keyofE.html
const string cipher_string =
"PBNXASMDAXNOOYHRCZGVVZCBIGIBGWHMXKRRVQCFJCZPTUNSWADDSTIGQQCSAGPKRXXLOMGFXAPHH"
"MRFSDKYTMYPMVROHASQYRWFWVAVGCCUDBIBXXDYZSACJSYOTMWUCNWOMHHJPYWDCCLUPGSWCLMBCZ"
"SSYXPGMGMQXAUFULNOZEQENHEIZZAKLC";
 
const string solution_text = 
"ihykteinsseqsxaqtxviereinsxviergefallenexeinsnullverwuntetexneunzugangxvierne"
"unbestandxfeldlazaretteingesetztinxbhfxutorgoschxbhfxutorgoschxhockxhockxsieg"
"friedsiegfriedxstandartenfuehrer";

const string solution_key_string = "B:524:KZ:YEY";
const string solution_plug_string = "AECFGLHIKPMSNROUQYTW";
const int solution_score = 4365261;

const int8_t test_order[ALPSIZE] = { 
    2, 18, 12, 23, 6, 0, 25, 24, 22, 15, 7, 17, 14, 
    3, 21, 20, 16, 13, 1, 11, 8, 5, 19, 10, 9, 4 };


//plugboard found using IC score (4 pairs correct).
//used at the first call to unigram score function
const string uni_plug_string = "APBKERGLHIMSNTQYUW";

//plugboard found by running ic and unigrams once
const string tri_plug_string = "AEBKGMHILNPTQYRU";




//from http://cryptocellar.org/Enigma/Enigma_ModernBreaking.html
//36 chars score=16787
const string cipher_PFCXY = "PSQDBCSFKHFJOMVCJAUXTOTQBBPBWACHZYXH";
//const string key_PFCXY = "B:432:RIT:VOR";
const string key_PFCXY = "B:432:IT:EOR"; //QT:DVR
const string plugs_PFCXY = "AHBODPEXFNJQKSLRMUTZ";
const string text_PFCXY =   "abendmeldungenentfallenxhartjenstein";

//32 chars score=16082
const string cipher_AMERI = "TDLYXLHUVKOGOTUXNVRBPVICIBWTSTYD";
const string key_AMERI = "B:132:LES:BEN"; //WS:PVN
const string plugs_AMERI = "AYBJDGEHFQIMKOLPNWRT";
const string text_AMERI = "zwoxstafdelamxzwosiebenxaqtxvier";

//18 chars
const string cipher_EJRSB = "UNXXISVILMHHKZPJZU";
//const string key_EJRSB = "B:321:XBM:DOF";
const string key_EJRSB = "B:321:BM:GOF";
const string plugs_EJRSB = "AEBTCFDKGJHMISLVOZUX";
const string text_EJRSB = "tagesmmldungfunken";
//one climber pass = 440861  "BWDTEUFJHXILKSMOPRYZ"
const int8_t test_order_EJRSB[ALPSIZE] = {
    25, 23, 20, 8, 7, 21, 18, 15, 13, 12, 11, 10, 9,
    24, 22, 19, 17, 16, 14, 6, 5, 4, 3, 2, 1, 0 };

//46 chars  score=19385
const string cipher_YYBRW = "CFVUAHZHPIWNUCXTMJGXPMVWKFVHZJTJGXMSSDJYESRCNX";
const string key_YYBRW = "B:341:WGR:TOR"; //CR:WJR
const string plugs_YYBRW = "ACBEHWIPJZKYLUOSQRVX";
const string text_YYBRW =   "wogefeqtsstandquartiermeisterabtxroemeinsberta";

//M4 from http://www.bytereef.org/m4-project-first-break.html
const string cipher_M4_1 =
"NCZWVUSXPNYMINHZXMQXSFWXWLKJAHSHNMCOCCAKUQPMKCSMHKSEINJUSBLKIOSXCKUBHMLLX"
"CSJUSRRDVKOHULXWCCBGVLIYXEOAHXRHKKFVDREWEZLXOBAFGYUJQUKGRTVUKAMEURBVEKSUH"
"HVOYHABCJWMAKLFKLMYFVNRIZRVVRTKOFDANJMOLBGFFLEOPRGTFLVRHOWOPBEKVWMUQFMPWP"
"ARMFHAGKXIIBG";
const string key_M4_1 = "b:b241:AV:VJNA";
const string plugs_M4_1 = "ATBLDFGJHMNWOPQYRZVX";
const string text_M4_1 =
"vonvonjlooksjhffttteinseinsdreizwoyyqnnsneuninhaltxxbeiangriffunterwasser"
"gedruecktywabosxletztergegnerstandnulachtdreinuluhrmarquantonjotaneunacht"
"seyhsdreiyzwozwonulgradyachtsmystossenachxeknsviermbfaelltynnnnnnoooviery"
"sichteinsnull";
