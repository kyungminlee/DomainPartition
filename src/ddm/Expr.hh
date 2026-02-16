#pragma once

#include <iostream>
#include <memory>
#include "anyprint.hh"

struct ExprNode {
  enum class Kind { Literal, Neg, Add, Sub, Mul, Div };
  Kind kind;
  double value;  // used for Literal
  std::shared_ptr<ExprNode> lhs, rhs;  // used for binary ops; lhs only for Neg

  static std::shared_ptr<ExprNode> literal(double v) {
    return std::make_shared<ExprNode>(ExprNode{Kind::Literal, v, nullptr, nullptr});
  }
  static std::shared_ptr<ExprNode> unary(Kind k, std::shared_ptr<ExprNode> operand) {
    return std::make_shared<ExprNode>(ExprNode{k, 0.0, std::move(operand), nullptr});
  }
  static std::shared_ptr<ExprNode> binary(Kind k, std::shared_ptr<ExprNode> l, std::shared_ptr<ExprNode> r) {
    return std::make_shared<ExprNode>(ExprNode{k, 0.0, std::move(l), std::move(r)});
  }
};

class Expr {
public:
  Expr(double v = 0.0) : _value(v), _node(ExprNode::literal(v)) {}
  Expr(int v) : _value(static_cast<double>(v)), _node(ExprNode::literal(static_cast<double>(v))) {}

  explicit operator double() const { return _value; }
  double value() const { return _value; }

  Expr operator-() const {
    return Expr(-_value, ExprNode::unary(ExprNode::Kind::Neg, _node));
  }

  Expr & operator+=(Expr const & rhs) { return *this = *this + rhs; }
  Expr & operator-=(Expr const & rhs) { return *this = *this - rhs; }
  Expr & operator*=(Expr const & rhs) { return *this = *this * rhs; }
  Expr & operator/=(Expr const & rhs) { return *this = *this / rhs; }

  friend Expr operator+(Expr const & a, Expr const & b) {
    if (isZeroLiteral(a._node)) return b;
    if (isZeroLiteral(b._node)) return a;
    return Expr(a._value + b._value, ExprNode::binary(ExprNode::Kind::Add, a._node, b._node));
  }
  friend Expr operator-(Expr const & a, Expr const & b) {
    if (isZeroLiteral(b._node)) return a;
    return Expr(a._value - b._value, ExprNode::binary(ExprNode::Kind::Sub, a._node, b._node));
  }
  friend Expr operator*(Expr const & a, Expr const & b) {
    return Expr(a._value * b._value, ExprNode::binary(ExprNode::Kind::Mul, a._node, b._node));
  }
  friend Expr operator/(Expr const & a, Expr const & b) {
    return Expr(a._value / b._value, ExprNode::binary(ExprNode::Kind::Div, a._node, b._node));
  }

  friend bool operator==(Expr const & a, Expr const & b) { return a._value == b._value; }
  friend bool operator!=(Expr const & a, Expr const & b) { return a._value != b._value; }
  friend bool operator< (Expr const & a, Expr const & b) { return a._value <  b._value; }
  friend bool operator> (Expr const & a, Expr const & b) { return a._value >  b._value; }
  friend bool operator<=(Expr const & a, Expr const & b) { return a._value <= b._value; }
  friend bool operator>=(Expr const & a, Expr const & b) { return a._value >= b._value; }

  friend std::ostream & operator<<(std::ostream & os, Expr const & e) {
    printNode(os, e._node);
    return os;
  }

  std::shared_ptr<ExprNode> const & node() const { return _node; }

private:
  Expr(double v, std::shared_ptr<ExprNode> node) : _value(v), _node(std::move(node)) {}

  static bool isZeroLiteral(std::shared_ptr<ExprNode> const & n) {
    return n && n->kind == ExprNode::Kind::Literal && n->value == 0.0;
  }

  static void printNode(std::ostream & os, std::shared_ptr<ExprNode> const & n) {
    if (!n) { os << "?"; return; }
    switch (n->kind) {
      case ExprNode::Kind::Literal:
        os << n->value;
        break;
      case ExprNode::Kind::Neg:
        os << "(-";
        printNode(os, n->lhs);
        os << ")";
        break;
      case ExprNode::Kind::Add:
        os << "(";
        printNode(os, n->lhs);
        os << " + ";
        printNode(os, n->rhs);
        os << ")";
        break;
      case ExprNode::Kind::Sub:
        os << "(";
        printNode(os, n->lhs);
        os << " - ";
        printNode(os, n->rhs);
        os << ")";
        break;
      case ExprNode::Kind::Mul:
        os << "(";
        printNode(os, n->lhs);
        os << " * ";
        printNode(os, n->rhs);
        os << ")";
        break;
      case ExprNode::Kind::Div:
        os << "(";
        printNode(os, n->lhs);
        os << " / ";
        printNode(os, n->rhs);
        os << ")";
        break;
    }
  }

  double _value;
  std::shared_ptr<ExprNode> _node;
};

template <>
class anyprint::writer<Expr> {
public:
  static void write(std::ostream & os, Expr const & e) {
    os << e;
  }
};
