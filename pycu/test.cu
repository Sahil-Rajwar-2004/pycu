#include <iostream>
#include "intn.cu"
using namespace std;

int main(){
  int16 x(2);
  int32 a(3);
  int64 u(2);

  int16_t v = 2;
  int64_t w = 12;

  cout << x.add(u).getValue() << endl;
  cout << x.sub(u).getValue() << endl;
  cout << v + w << endl;


  cout << a.eq(u) << endl;
//   cout << x.getValue() << " + " << a.getValue() << " = " << x.add(a).getValue() << endl;
//   cout << x.getValue() << " - " << a.getValue() << " = " << x.sub(a).getValue() << endl;
//   cout << x.getValue() << " * " << a.getValue() << " = " << x.mul(a).getValue() << endl;
//   cout << x.getValue() << " / " << a.getValue() << " = " << x.tdiv(a).getValue() << endl;
//   cout << x.getValue() << " // " << a.getValue() << " = " << x.fdiv(a).getValue() << endl;
//   cout << x.getValue() << " % " << a.getValue() << " = " << x.mod(a).getValue() << endl;
//   cout << x.getValue() << " ^ " << a.getValue() << " = " << x.pow(a).getValue() << endl;


//  cout << x.getValue() << " == " << a.getValue() << " = " << x.eq(a) << endl;
//  cout << x.getValue() << " != " << a.getValue() << " = " << x.ne(a) << endl;
//  cout << x.getValue() << " > " << a.getValue() << " = " << x.gt(a) << endl;
//  cout << x.getValue() << " >= " << a.getValue() << " = " << x.ge(a) << endl;
//  cout << x.getValue() << " < " << a.getValue() << " = " << x.lt(a) << endl;
//  cout << x.getValue() << " <= " << a.getValue() << " = " << x.le(a) << endl;

  // cout << x.sub(u).getValue() << endl;
  // cout << x.mul(u).getValue() << endl;
  // cout << x.tdiv(u).getValue() << endl;

  // int16_t x = 2;
  // int64_t y = 12;
  // int16_t z = 13;

  // cout << x+y << endl;
  // cout << x+z << endl;

  return 0;
}
