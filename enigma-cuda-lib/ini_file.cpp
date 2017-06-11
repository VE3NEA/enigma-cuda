#include <iostream>
#include <fstream>
#include "ini_file.h"
#include "util.h"


void IniFile::ReadFromFile(const string & file_name)
{
  sections.clear();
  if (!FileExists(file_name)) return;

  string line, section = "", key, value;

  std::ifstream file(file_name);
  while (std::getline(file, line))
  {
    //whitespace or comment
    line = Trim(line);
    if (line == "") continue;
    if (line[0] == ';') continue;
    
    //section
    if (line[0] == '[')
    {
      if (line[line.length() - 1] == ']')       
        section = Trim(line.substr(1, line.length() - 2));
      continue;
    }
    if (section == "") continue;

    //value
    int p = line.find('=');
    if (p == string::npos) continue;
    key = Trim(line.substr(0, p));
    value = Trim(line.substr(p+1));
    sections[section][key] = value;
  }
}

string IniFile::ReadString(const string & section, const string & key, const string & default)
{
  return string();
}
