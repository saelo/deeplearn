//
// File utility functions
//
// Copyright (c) 2016 Samuel Groß
//

#ifndef __FILE_H__
#define __FILE_H__

#include <fstream>

namespace utils {

bool LoadFile(std::string path, std::string* content)
{
    std::ifstream file;

    file.open(path.c_str());
    FAIL_IF(!file.is_open(), "Failed to open file '" << path << "'.", false);

    file.seekg(0, std::ios::end);
    std::ifstream::pos_type fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    content->resize((size_t)fileSize);
    file.read(&content->at(0), fileSize);

    return true;
}

}       // namespace utils

#endif
