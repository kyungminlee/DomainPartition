#pragma once

#include <vector>
#include "anyprint.hh"

// replacement
template <typename T>
class RealNodeValue {
public:
  RealNodeValue(int nNode): _data(nNode) {}

  T & operator[](int i) { return _data.at(i); }
  T operator[](int i) const { return _data.at(i); }
  
  std::size_t size() const { return _data.size(); }

  decltype(auto) begin() const { return _data.begin(); }
  decltype(auto) end() const { return _data.end(); }
  decltype(auto) begin() { return _data.begin(); }
  decltype(auto) end() { return _data.end(); }

  std::vector<T> _data;
};

using RealNodeScalar = RealNodeValue<double>;







template <typename T>
class anyprint::writer<RealNodeValue<T>> {
public:
  static void write(std::ostream& os, const RealNodeValue<T> & vec) {
    os << "[";
    write_iterable(vec.begin(), vec.end(), os);
    os << "]";
  }
};
