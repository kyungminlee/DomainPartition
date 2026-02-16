#pragma once

#include <vector>

// replacement
class RealNodeScalar {
public:
  RealNodeScalar(int nNode): _data(nNode) {}

  double & operator[](int i) { return _data.at(i); }
  double operator[](int i) const { return _data.at(i); }
  
  std::vector<double> _data;
};