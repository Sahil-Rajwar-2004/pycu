#include <iostream>
#include "dtypes.cu"

int main(){

  int16 x(1);
  int16 y(2);

  int32 a(2);
  int32 b(3);

  int64 u(1);
  int64 v(3);

  std::cout << x.eq(y) << std::endl;
  std::cout << a.eq(b) << std::endl;
  std::cout << u.ne(v) << std::endl;

  return 0;
}
