#pragma once
#include <string>
#include <vector>
#include <map>
using std::string;

typedef std::map<string, string> Entries;

class IniFile
{
protected:
  void ReadFromFile(const string & file_name);
public:
  std::map<string, Entries> sections;

  IniFile(const string & file_name) { ReadFromFile(file_name); }
  string ReadString(const string & section, const string & key, const string & default);
};