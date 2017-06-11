/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include <iostream>
#include "runner.h"
#include "util.h"
#include "settings.h"
#include "plugboard.h"
#include "cuda_code.h"


bool Runner::Initialize(int max_length)
{
  try
  {
      if (!SelectGpuDevice(2, 1, silent)) return false;
      
      //load ciphertext
      ciphertext = LoadTextFromFile(settings.ciphertext_file_name);
      ciphertext = LettersFromText(ciphertext);
      ciphertext = ciphertext.substr(0, max_length);

      length = (int)ciphertext.length();
      if (length < MIN_MESSAGE_LENGTH)
          throw std::runtime_error("ciphertext too short");

      //ciphertext, wiring, results to device
      InitializeArrays(ciphertext, settings.turnover_modes, 
          settings.score_kinds);

      //ngrams to device
      NgramsToDevice(
          settings.unigram_file_name, 
          settings.bigram_file_name, 
          settings.trigram_file_name);

      //out file
      out_stream = &std::cout;
      if (settings.out_file_name != "")
      {
          out_file_stream.open(settings.out_file_name);
          out_stream = &out_file_stream;
      }

      //init random number generator
      srand((unsigned int)time(NULL));

      //segmenter words
      segmenter.LoadWordsFromFile();

      return true;
  }
  catch (const std::exception & e)
  {
      std::cerr << "Initialization failed: " << e.what() << std::endl;
      return false;
  }
}

bool Runner::Run()
{
  if (!silent)  
    std::cout << "Starting at " << TimeString() << std::endl << std::endl;

  current_pass = 0;

  try
  {
    start_time = clock();
    last_save_time = 0;

    //for each of n passes
    for (; settings.passes_left > 0; settings.passes_left--)
    {
      current_pass++;

      string start_key_string = (current_pass == 1) ?
        settings.current_key_string : settings.first_key_string;

      iterator.SetRange(start_key_string, settings.last_key_string);

      if (settings.first_pass) plugboard.InitializeSwapOrder(ciphertext);
      else plugboard.RandomizeSwapOrder();
      OrderToDevice(plugboard.order);

      //for each rotor order in the range
      do
      {
        if (!silent)
        {
          ShowProgressString(false);
          ShowProgressString(true);
        }
        GenerateScrambler(iterator.key);

        //for each ring settings
        do
        {
          if (settings.exhaust_multi_plugs != "")
            plugboard.StartExhaustive(settings.exhaust_multi_plugs, false);
          else if (settings.exhaust_single_plugs != "")
            plugboard.StartExhaustive(settings.exhaust_single_plugs, true);
          else if (settings.known_plugs != "")
            plugboard.SetFixedPlugs(settings.known_plugs);
          else plugboard.Reset();



          //for each fixed plug
          do
          {
            PlugboardToDevice(plugboard);

            Result result = Climb(length, iterator.key, settings.single_key);

            ProcessResult(result);

            if (settings.best_score >= settings.stop_at_score)
            {
              if (!silent)
              {
                std::cout << "STOPPED at score >= ";
                std::cout << settings.stop_at_score;
                std::cout << std::endl << std::endl;
              }
              return true;
            }

            //{!}
            //plugboard.RandomizeSwapOrder();
            //OrderToDevice(plugboard.order);
          } while (plugboard.NextExahustive());
        } while (iterator.NextRingPosition());
      } while (iterator.NextRotorOrder());

      settings.first_pass = false;
    }


    if (!silent)
    {
      ShowProgressString(false);
      std::cout << "DONE: ";
      std::cout << current_pass << " passes in ";
      std::cout << TimeDiffString(clock() - start_time);
      std::cout << std::endl << std::endl;
    }
    return true;
  }
  catch (const std::exception & e)
  {
    if (!silent) ShowProgressString(false);
    std::cerr << "Runtime error: " << e.what() << std::endl;
    return false;
  }
}

void Runner::ProcessResult(const Result & result)
{
    if (result.score > settings.best_score)
    {
        //update "best_pair" info in settings

        if (settings.single_key)
            settings.best_key_string = iterator.key.ToString();
        else
        {
            Key key = iterator.key;
            key.sett.r_mesg = result.index % ALPSIZE;
            key.sett.m_mesg = (result.index / ALPSIZE) % ALPSIZE;
            key.sett.l_mesg = result.index / ALPSIZE_TO2;
            settings.best_key_string = key.ToString();
        }

        Plugboard plg;
        plg.FromData(result.plugs);
        settings.best_pluggoard_string = plg.ToString();

        settings.best_score = result.score;


        //plain text
        double seg_score = 0;
        string plain_text = DecodeMessage(ciphertext, settings.best_key_string,
            settings.best_pluggoard_string);
        if (segmenter.IsLoaded())
        {          
            seg_score = segmenter.FindSegmentation(plain_text);
            plain_text = segmenter.GetSegmentedText(true);
        }

        //print result
        if (!silent)
        {
          ShowProgressString(false);

          *out_stream << "Spent: " << TimeDiffString(clock() - start_time) << std::endl;
          *out_stream << "Pass:  " << current_pass << std::endl;
          if (settings.exhaust_single_plugs != "" || settings.exhaust_multi_plugs != "")
            *out_stream << "Fixed: " << plugboard.ExahustivePlugsToString() << std::endl;
          *out_stream << "Score: " << settings.best_score << std::endl;
          *out_stream << "Words: " << seg_score << std::endl;
          *out_stream << "Key:   " << settings.best_key_string << std::endl;
          *out_stream << "Plugs: " << settings.best_pluggoard_string << std::endl;
          *out_stream << "Text:  " << plain_text << std::endl;
          *out_stream << std::endl;

          ShowProgressString(true);
        }

    }

    //save Resume file
    int seconds = (clock() - last_save_time) / CLOCKS_PER_SEC;
    if (seconds >= 60 || settings.best_score >= settings.stop_at_score) 
    {
        settings.current_key_string = iterator.key.ToString();
        settings.SaveResumeFile();
        last_save_time = clock();
    }
}

void Runner::ShowProgressString(bool show)
{
  if (show)
  {
    progress_string = TimeDiffString(clock() - start_time) + 
      " pass " + std::to_string(current_pass) +
      " , trying " + iterator.key.ToString().substr(0, 6) + "...";
    std::cout << progress_string;
  }
  else
    for (int i = 0; i < progress_string.length(); ++i) std::cout << "\b \b";
}