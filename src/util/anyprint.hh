#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <tuple>

namespace anyprint {

  template <typename T>
  class writer {
  public:
    static void write(std::ostream& os, const T& value) {
      os << value;
    }
  };
  
  template <typename ... Ts>
  void write(std::ostream& os, const Ts& ... values) {
    (writer<Ts>::write(os, values), ...);
  }

  template <typename Iter>
  void write_iterable(Iter begin, Iter last, std::ostream & os) {
    char const * sep = "";
    for (auto it = begin; it != last; ++it) {
      os << sep;
      sep = ", ";
      write(os, *it);
    }
  }

  template <typename Iter>
  void write_mappable(Iter begin, Iter last, std::ostream & os) {
    char const * sep = "";
    for (auto it = begin; it != last; ++it) {
      os << sep;
      sep = ", ";
      write(os, it->first);
      os << ": ";
      write(os, it->second);
    }
  }


  template <typename ... Ts>
  void writeln(std::ostream& os, const Ts& ... values) {
    write(os, values...);
    os << std::endl;
  }

  template <typename ... Ts>
  void print(const Ts& ... values) {
    write(std::cout, values...);
    std::cout << std::endl;
  }

  template <typename T>
  class writer<std::vector<T>> {
  public:
    static void write(std::ostream& os, const std::vector<T>& vec) {
      os << "[";
      write_iterable(vec.begin(), vec.end(), os);
      os << "]";
    }
  };

  template <typename K, typename V>
  class writer<std::map<K, V>> {
  public:
    static void write(std::ostream& os, const std::map<K, V>& m) {
      os << "{";
      write_mappable(m.begin(), m.end(), os);
      os << "}";
    }
  };

  template <typename ... Ts>
  class writer<std::tuple<Ts...>> {
    public:
    static void write(std::ostream& os, const std::tuple<Ts...>& t) {
      os << "(";
      std::apply([&os](auto const&... args) {
        size_t n = 0;
        (((n++ ? (os << ", ") : os), write(os, args)), ...);
      }, t);
      os << ")";
    }
  };

  class indentation {
    public:
    indentation(int n): _n(n) {}
    indentation(indentation &&) = default;
    indentation(indentation const &) = default;
    indentation & operator=(indentation &&) = default;
    indentation & operator=(indentation const &) = default;

    template <typename Ch, typename Tr>
    friend std::basic_ostream<Ch, Tr> & operator<<(std::basic_ostream<Ch, Tr> & os, indentation const & indent) {
      for (int i = 0; i < indent._n; ++i) {
        os << Ch(' ');
      }
      return os;
    }
    std::size_t _n;
  };

};