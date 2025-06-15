//
// Common classes, macros, etc.
//
// Copyright (c) 2016 Samuel Groß
//

#ifndef __COMMON_H__
#define __COMMON_H__

#include <cstdint>
#include <vector>
#include <iostream>

#if DEBUG
  #define Assert(expr) if (!(expr)) { std::cerr << "Assertion \"" << #expr << "\" failed (in " << __FILE__ << ":" << __LINE__ << "). Aborting" << std::endl; abort(); }
#else
  #define Assert(expr) if (!(expr)) { }
#endif

#define Check(expr, errmsg) if (!(expr)) { std::cerr << "Fatal: " << errmsg << " (in " << __FILE__ << ":" << __LINE__ << ")" << std::endl; abort(); }

#define WARN_IF(expr, warnmsg) if (expr) { std::cerr << "Warning: " << warnmsg << std::endl; }
#define FAIL_IF(expr, errmsg, retval) if (expr) { std::cerr << "Error: " << errmsg << std::endl; return retval; }

#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;   \
  void operator=(const TypeName&) = delete

#endif
