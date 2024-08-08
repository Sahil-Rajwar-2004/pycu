#include <cuda_runtime.h>


// CUDA kernels starts from here


__global__ void addKernel(const int16_t *x,const int16_t *y,int16_t *z){ *z = *x + *y; }
__global__ void subKernel(const int16_t *x,const int16_t *y,int16_t *z){ *z = *x - *y; }
__global__ void mulKernel(const int16_t *x,const int16_t *y,int16_t *z){ *z = *x * *y; }
__global__ void tdivKernel(const int16_t *x,const int16_t *y,int16_t *z){ *z = *x / *y; }
__global__ void fdivKernel(const int16_t *x,const int16_t *y,int16_t *z){ *z = *x / *y; }
__global__ void modKernel(const int16_t *x,const int16_t *y,int16_t *z){ *z = *x % *y; }
__global__ void powKernel(const int16_t *x,const int16_t *y,int16_t *z){ *z = pow(*x,*y); }
__global__ void absKernel(const int16_t *x,int16_t *z){
  if(*x < 0){ *z = -*x; }
  else{ *z = *x; }
}
__global__ void negKernel(const int16_t *x,int16_t *z){ *z = -*x; }
__global__ void posKernel(const int16_t *x,int16_t *z){ *z = +*x; }
__global__ void eqKernel(const int16_t *x,const int16_t *y,int16_t *z){ *z = (*x == *y) ? 1 : 0; }
__global__ void neKernel(const int16_t *x,const int16_t *y,int16_t *z){ *z = (*x != *y) ? 1 : 0; }
__global__ void gtKernel(const int16_t *x,const int16_t *y,int16_t *z){ *z = (*x > *y) ? 1 : 0; }
__global__ void geKernel(const int16_t *x,const int16_t *y,int16_t *z){ *z = (*x >= *y) ? 1 : 0; }
__global__ void ltKernel(const int16_t *x,const int16_t *y,int16_t *z){ *z = (*x < *y) ? 1 : 0; }
__global__ void leKernel(const int16_t *x,const int16_t *y,int16_t *z){ *z = (*x <= *y) ? 1 : 0; }

__global__ void addKernel(const int32_t *x,const int32_t *y,int32_t *z){ *z = *x + *y; }
__global__ void subKernel(const int32_t *x,const int32_t *y,int32_t *z){ *z = *x - *y; }
__global__ void mulKernel(const int32_t *x,const int32_t *y,int32_t *z){ *z = *x * *y; }
__global__ void tdivKernel(const int32_t *x,const int32_t *y,int32_t *z){ *z = *x / *y; }
__global__ void fdivKernel(const int32_t *x,const int32_t *y,int32_t *z){ *z = *x / *y; }
__global__ void modKernel(const int32_t *x,const int32_t *y,int32_t *z){ *z = *x % *y; }
__global__ void powKernel(const int32_t *x,const int32_t *y,int32_t *z){ *z = pow(*x,*y); }
__global__ void absKernel(const int32_t *x,int32_t *z){
  if(*x < 0){ *z = -*x; }
  else{ *z = *x; }
}
__global__ void negKernel(const int32_t *x,int32_t *z){ *z = -*x; }
__global__ void posKernel(const int32_t *x,int32_t *z){ *z = +*x; }
__global__ void eqKernel(const int32_t *x,const int32_t *y,int32_t *z){ *z = (*x == *y) ? 1 : 0; }
__global__ void neKernel(const int32_t *x,const int32_t *y,int32_t *z){ *z = (*x != *y) ? 1 : 0; }
__global__ void gtKernel(const int32_t *x,const int32_t *y,int32_t *z){ *z = (*x > *y) ? 1 : 0; }
__global__ void geKernel(const int32_t *x,const int32_t *y,int32_t *z){ *z = (*x >= *y) ? 1 : 0; }
__global__ void ltKernel(const int32_t *x,const int32_t *y,int32_t *z){ *z = (*x < *y) ? 1 : 0; }
__global__ void leKernel(const int32_t *x,const int32_t *y,int32_t *z){ *z = (*x <= *y) ? 1 : 0; }

__global__ void addKernel(const int64_t *x,const int64_t *y,int64_t *z){ *z = *x + *y; }
__global__ void subKernel(const int64_t *x,const int64_t *y,int64_t *z){ *z = *x - *y; }
__global__ void mulKernel(const int64_t *x,const int64_t *y,int64_t *z){ *z = *x * *y; }
__global__ void tdivKernel(const int64_t *x,const int64_t *y,int64_t *z){ *z = *x / *y; }
__global__ void fdivKernel(const int64_t *x,const int64_t *y,int64_t *z){ *z = *x / *y; }
__global__ void modKernel(const int64_t *x,const int64_t *y,int64_t *z){ *z = *x % *y; }
__global__ void powKernel(const int64_t *x,const int64_t *y,int64_t *z){ *z = pow(*x,*y); }
__global__ void absKernel(const int64_t *x,int64_t *z){
  if(*x < 0){ *z = -*x; }
  else{ *z = *x; }
}
__global__ void negKernel(const int64_t *x,int64_t *z){ *z = -*x; }
__global__ void posKernel(const int64_t *x,int64_t *z){ *z = +*x; }
__global__ void eqKernel(const int64_t *x,const int64_t *y,int64_t *z){ *z = (*x == *y) ? 1 : 0; }
__global__ void neKernel(const int64_t *x,const int64_t *y,int64_t *z){ *z = (*x != *y) ? 1 : 0; }
__global__ void gtKernel(const int64_t *x,const int64_t *y,int64_t *z){ *z = (*x > *y) ? 1 : 0; }
__global__ void geKernel(const int64_t *x,const int64_t *y,int64_t *z){ *z = (*x >= *y) ? 1 : 0; }
__global__ void ltKernel(const int64_t *x,const int64_t *y,int64_t *z){ *z = (*x < *y) ? 1 : 0; }
__global__ void leKernel(const int64_t *x,const int64_t *y,int64_t *z){ *z = (*x <= *y) ? 1 : 0; }


// CUDA kernel ends here


// predefined classes
class int16 ;
class int32 ;
class int64 ;




class int16{
  private:
    int16_t *val;
  
  public:
    int16(const int16_t &value){
      cudaMalloc(&val,sizeof(int16_t));
      cudaMemcpy(val,&value,sizeof(int16_t),cudaMemcpyHostToDevice);
    }

    ~int16(){ if(val){ cudaFree(val); } }

    int16_t getValue() const {
      int16_t host_value;
      cudaMemcpy(&host_value,val,sizeof(int16_t),cudaMemcpyDeviceToHost);
      return host_value;
    }

    int16_t *deviceValue() const { return this -> val; }
/*
    int32 to_int32() const {
      int16_t buff = getValue();
      return int32(static_cast<int32_t>(buff));
    }
*/
    int16 add(const int16 &other) const ;
    int16 sub(const int16 &other) const ;
    int16 mul(const int16 &other) const ;
    int16 tdiv(const int16 &other) const ;
    int16 fdiv(const int16 &other) const ;
    int16 mod(const int16 &other) const ;
    int16 pow(const int16 &other) const ;

    int16 abs() const ;
    int16 neg() const ;
    int16 pos() const ;

    int16_t eq(const int16 &other) const ;
    int16_t ne(const int16 &other) const ;
    int16_t gt(const int16 &other) const ;
    int16_t ge(const int16 &other) const ;
    int16_t lt(const int16 &other) const ;
    int16_t le(const int16 &other) const ;
};


class int32{
  private:
   int32_t *val;

  public:
    int32(const int32_t &value){
      cudaMalloc(&val,sizeof(int32_t));
      cudaMemcpy(val,&value,sizeof(int32_t),cudaMemcpyHostToDevice);
    }

    ~int32(){ if(val){ cudaFree(val); } }

    int32_t getValue() const {
      int32_t host_value;
      cudaMemcpy(&host_value,val,sizeof(int32_t),cudaMemcpyDeviceToHost);
      return host_value;
    }

    int32_t *deviceValue() const { return this -> val; }

    int32 add(const int32 &other) const ;
    int32 sub(const int32 &other) const ;
    int32 mul(const int32 &other) const ;
    int32 tdiv(const int32 &other) const ;
    int32 fdiv(const int32 &other) const ;
    int32 mod(const int32 &other) const ;
    int32 pow(const int32 &other) const ;

    int32 abs() const ;
    int32 neg() const ;
    int32 pos() const ;

    int32_t eq(const int32 &other) const ;
    int32_t ne(const int32 &other) const ;
    int32_t gt(const int32 &other) const ;
    int32_t ge(const int32 &other) const ;
    int32_t lt(const int32 &other) const ;
    int32_t le(const int32 &other) const ;
    
};


class int64{
  private:
    int64_t *val;

  public:
    int64(const int64_t &value){
      cudaMalloc(&val,sizeof(int64_t));
      cudaMemcpy(val,&value,sizeof(int64_t),cudaMemcpyHostToDevice);
    }

    ~int64(){ if(val){ cudaFree(val); } }

    int64_t getValue() const {
      int64_t host_value;
      cudaMemcpy(&host_value,val,sizeof(int64_t),cudaMemcpyDeviceToHost);
      return host_value;
    }

    int64_t *deviceValue() const { return this -> val; }

    int64 add(const int64 &other) const ;
    int64 sub(const int64 &other) const ;
    int64 mul(const int64 &other) const ;
    int64 tdiv(const int64 &other) const ;
    int64 fdiv(const int64 &other) const ;
    int64 mod(const int64 &other) const ;
    int64 pow(const int64 &other) const ;

    int64 abs() const ;
    int64 neg() const ;
    int64 pos() const ;

    int64_t eq(const int64 &other) const ;
    int64_t ne(const int64 &other) const ;
    int64_t gt(const int64 &other) const ;
    int64_t ge(const int64 &other) const ;
    int64_t lt(const int64 &other) const ;
    int64_t le(const int64 &other) const ;
};


int16 int16::add(const int16 &other) const {
  int16 result(0);
  addKernel<<<1,1>>>(this -> val,other.val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int16 int16::sub(const int16 &other) const {
  int16 result(0);
  subKernel<<<1,1>>>(this -> val,other.val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int16 int16::mul(const int16 &other) const {
  int16 result(0);
  mulKernel<<<1,1>>>(this -> val,other.val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int16 int16::tdiv(const int16 &other) const {
  int16 result(0);
  tdivKernel<<<1,1>>>(this -> val,other.val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int16 int16::fdiv(const int16 &other) const {
  int16 result(0);
  fdivKernel<<<1,1>>>(this -> val,other.val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int16 int16::mod(const int16 &other) const {
  int16 result(0);
  modKernel<<<1,1>>>(this -> val,other.val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int16 int16::pow(const int16 &other) const {
  int16 result(0);
  powKernel<<<1,1>>>(this -> val,other.val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int16 int16::abs() const {
  int16 result(0);
  absKernel<<<1,1>>>(this -> val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int16 int16::neg() const {
  int16 result(0);
  negKernel<<<1,1>>>(this -> val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int16 int16::pos() const {
  int16 result(0);
  posKernel<<<1,1>>>(this -> val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int16_t int16::eq(const int16 &other) const {
  int16_t host_value;
  int16_t *device_value;
  cudaMalloc(&device_value,sizeof(int16_t));
  eqKernel<<<1,1>>>(this -> val,other.val,device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int16_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int16_t int16::ne(const int16 &other) const {
  int16_t host_value;
  int16_t *device_value;
  cudaMalloc(&device_value,sizeof(int16_t));
  neKernel<<<1,1>>>(this -> val,other.val,device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int16_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int16_t int16::gt(const int16 &other) const {
  int16_t host_value;
  int16_t *device_value;
  cudaMalloc(&device_value,sizeof(int16_t));
  gtKernel<<<1,1>>>(this -> val,other.val,device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int16_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int16_t int16::ge(const int16 &other) const {
  int16_t host_value;
  int16_t *device_value;
  cudaMalloc(&device_value,sizeof(int16_t));
  geKernel<<<1,1>>>(this -> val,other.val,device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int16_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int16_t int16::lt(const int16 &other) const {
  int16_t host_value;
  int16_t *device_value;
  cudaMalloc(&device_value,sizeof(int16_t));
  ltKernel<<<1,1>>>(this -> val,other.val,device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int16_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int16_t int16::le(const int16 &other) const {
  int16_t host_value;
  int16_t *device_value;
  cudaMalloc(&device_value,sizeof(int16_t));
  leKernel<<<1,1>>>(this -> val,other.val,device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int16_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}




int32 int32::add(const int32 &other) const {
  int32 result(0);
  addKernel<<<1,1>>>(this -> val,other.val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int32::sub(const int32 &other) const {
  int32 result(0);
  subKernel<<<1,1>>>(this -> val,other.val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int32::mul(const int32 &other) const {
  int32 result(0);
  mulKernel<<<1,1>>>(this -> val,other.val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int32::tdiv(const int32 &other) const {
  int32 result(0);
  tdivKernel<<<1,1>>>(this -> val,other.val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int32::fdiv(const int32 &other) const {
  int32 result(0);
  fdivKernel<<<1,1>>>(this -> val,other.val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int32::mod(const int32 &other) const {
  int32 result(0);
  modKernel<<<1,1>>>(this -> val,other.val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int32::pow(const int32 &other) const {
  int32 result(0);
  powKernel<<<1,1>>>(this -> val,other.val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int32::abs() const {
  int32 result(0);
  absKernel<<<1,1>>>(this -> val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int32::neg() const {
  int32 result(0);
  negKernel<<<1,1>>>(this -> val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int32::pos() const {
  int32 result(0);
  posKernel<<<1,1>>>(this -> val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32_t int32::eq(const int32 &other) const {
  int32_t host_value;
  int32_t *device_value;
  cudaMalloc(&device_value,sizeof(int32_t));
  eqKernel<<<1,1>>>(this -> val,other.val,device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int32_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int32_t int32::ne(const int32 &other) const {
  int32_t host_value;
  int32_t *device_value;
  cudaMalloc(&device_value,sizeof(int32_t));
  neKernel<<<1,1>>>(this -> val,other.val,device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int32_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int32_t int32::gt(const int32 &other) const {
  int32_t host_value;
  int32_t *device_value;
  cudaMalloc(&device_value,sizeof(int32_t));
  gtKernel<<<1,1>>>(this -> val,other.val,device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int32_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int32_t int32::ge(const int32 &other) const {
  int32_t host_value;
  int32_t *device_value;
  cudaMalloc(&device_value,sizeof(int32_t));
  geKernel<<<1,1>>>(this -> val,other.val,device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int32_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int32_t int32::lt(const int32 &other) const {
  int32_t host_value;
  int32_t *device_value;
  cudaMalloc(&device_value,sizeof(int32_t));
  ltKernel<<<1,1>>>(this -> val,other.val,device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int32_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int32_t int32::le(const int32 &other) const {
  int32_t host_value;
  int32_t *device_value;
  cudaMalloc(&device_value,sizeof(int32_t));
  leKernel<<<1,1>>>(this -> val,other.val,device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int32_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}





int64 int64::add(const int64 &other) const {
  int64 result(0);
  addKernel<<<1,1>>>(this -> val,other.val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int64::sub(const int64 &other) const {
  int64 result(0);
  subKernel<<<1,1>>>(this -> val,other.val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int64::mul(const int64 &other) const {
  int64 result(0);
  mulKernel<<<1,1>>>(this -> val,other.val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int64::tdiv(const int64 &other) const {
  int64 result(0);
  tdivKernel<<<1,1>>>(this -> val,other.val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int64::fdiv(const int64 &other) const {
  int64 result(0);
  fdivKernel<<<1,1>>>(this -> val,other.val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int64::mod(const int64 &other) const {
  int64 result(0);
  modKernel<<<1,1>>>(this -> val,other.val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int64::pow(const int64 &other) const {
  int64 result(0);
  powKernel<<<1,1>>>(this -> val,other.val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int64::abs() const {
  int64 result(0);
  absKernel<<<1,1>>>(this -> val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int64::neg() const {
  int64 result(0);
  negKernel<<<1,1>>>(this -> val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int64::pos() const {
  int64 result(0);
  posKernel<<<1,1>>>(this -> val,result.val);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64_t int64::eq(const int64 &other) const {
  int64_t host_value;
  int64_t *device_value;
  cudaMalloc(&device_value,sizeof(int64_t));
  eqKernel<<<1,1>>>(this -> val,other.val,device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int64_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int64_t int64::ne(const int64 &other) const {
  int64_t host_value;
  int64_t *device_value;
  cudaMalloc(&device_value,sizeof(int64_t));
  neKernel<<<1,1>>>(this -> val,other.val,device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int64_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int64_t int64::gt(const int64 &other) const {
  int64_t host_value;
  int64_t *device_value;
  cudaMalloc(&device_value,sizeof(int64_t));
  gtKernel<<<1,1>>>(this -> val,other.val,device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int64_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int64_t int64::ge(const int64 &other) const {
  int64_t host_value;
  int64_t *device_value;
  cudaMalloc(&device_value,sizeof(int64_t));
  geKernel<<<1,1>>>(this -> val,other.val,device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int64_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int64_t int64::lt(const int64 &other) const {
  int64_t host_value;
  int64_t *device_value;
  cudaMalloc(&device_value,sizeof(int64_t));
  ltKernel<<<1,1>>>(this -> val,other.val,device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int64_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int64_t int64::le(const int64 &other) const {
  int64_t host_value;
  int64_t *device_value;
  cudaMalloc(&device_value,sizeof(int64_t));
  leKernel<<<1,1>>>(this -> val,other.val,device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int64_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}



extern "C"{
  int16 *int16_new(int16_t value){ return new int16(value); }
  void int16_delete(int16 *value){ delete value; }
  int16_t int16_value(int16 *value){ return value -> getValue(); }

  int16 *int16_add(int16 *x,int16 *y){ return new int16(x -> add(*y)); }
  int16 *int16_sub(int16 *x,int16 *y){ return new int16(x -> sub(*y)); }
  int16 *int16_mul(int16 *x,int16 *y){ return new int16(x -> mul(*y)); }
  int16 *int16_tdiv(int16 *x,int16 *y){ return new int16(x -> tdiv(*y)); }
  int16 *int16_fdiv(int16 *x,int16 *y){ return new int16(x -> fdiv(*y)); }
  int16 *int16_mod(int16 *x,int16 *y){ return new int16(x -> mod(*y)); }
  int16 *int16_pow(int16 *x,int16 *y){ return new int16(x -> pow(*y)); }
  int16 *int16_abs(int16 *x){ return new int16(x -> abs()); }
  int16 *int16_neg(int16 *x){ return new int16(x -> neg()); }
  int16 *int16_pos(int16 *x){ return new int16(x -> pos()); }
  int16_t int16_eq(int16 *x,int16 *y){ return x -> eq(*y); }
  int16_t int16_ne(int16 *x,int16 *y){ return x -> ne(*y); }
  int16_t int16_gt(int16 *x,int16 *y){ return x -> gt(*y); }
  int16_t int16_ge(int16 *x,int16 *y){ return x -> ge(*y); }
  int16_t int16_lt(int16 *x,int16 *y){ return x -> lt(*y); }
  int16_t int16_le(int16 *x,int16 *y){ return x -> le(*y); }


  int32 *int32_new(int32_t value){ return new int32(value); }
  void int32_delete(int32 *value){ delete value; }
  int32_t int32_value(int32 *value){ return value -> getValue(); }

  int32 *int32_add(int32 *x,int32 *y){ return new int32(x -> add(*y)); }
  int32 *int32_sub(int32 *x,int32 *y){ return new int32(x -> sub(*y)); }
  int32 *int32_mul(int32 *x,int32 *y){ return new int32(x -> mul(*y)); }
  int32 *int32_tdiv(int32 *x,int32 *y){ return new int32(x -> tdiv(*y)); }
  int32 *int32_fdiv(int32 *x,int32 *y){ return new int32(x -> fdiv(*y)); }
  int32 *int32_mod(int32 *x,int32 *y){ return new int32(x -> mod(*y)); }
  int32 *int32_pow(int32 *x,int32 *y){ return new int32(x -> pow(*y)); }
  int32 *int32_abs(int32 *x){ return new int32(x -> abs()); }
  int32 *int32_neg(int32 *x){ return new int32(x -> neg()); }
  int32 *int32_pos(int32 *x){ return new int32(x -> pos()); }
  int32_t int32_eq(int32 *x,int32 *y){ return x -> eq(*y); }
  int32_t int32_ne(int32 *x,int32 *y){ return x -> ne(*y); }
  int32_t int32_gt(int32 *x,int32 *y){ return x -> gt(*y); }
  int32_t int32_ge(int32 *x,int32 *y){ return x -> ge(*y); }
  int32_t int32_lt(int32 *x,int32 *y){ return x -> lt(*y); }
  int32_t int32_le(int32 *x,int32 *y){ return x -> le(*y); }


  int64 *int64_new(int64_t value){ return new int64(value); }
  void int64_delete(int64 *value){ delete value; }
  int64_t int64_value(int64 *value){ return value -> getValue(); }

  int64 *int64_add(int64 *x,int64 *y){ return new int64(x -> add(*y)); }
  int64 *int64_sub(int64 *x,int64 *y){ return new int64(x -> sub(*y)); }
  int64 *int64_mul(int64 *x,int64 *y){ return new int64(x -> mul(*y)); }
  int64 *int64_tdiv(int64 *x,int64 *y){ return new int64(x -> tdiv(*y)); }
  int64 *int64_fdiv(int64 *x,int64 *y){ return new int64(x -> fdiv(*y)); }
  int64 *int64_mod(int64 *x,int64 *y){ return new int64(x -> mod(*y)); }
  int64 *int64_pow(int64 *x,int64 *y){ return new int64(x -> pow(*y)); }
  int64 *int64_abs(int64 *x){ return new int64(x -> abs()); }
  int64 *int64_neg(int64 *x){ return new int64(x -> neg()); }
  int64 *int64_pos(int64 *x){ return new int64(x -> pos()); }
  int64_t int64_eq(int64 *x,int64 *y){ return x -> eq(*y); }
  int64_t int64_ne(int64 *x,int64 *y){ return x -> ne(*y); }
  int64_t int64_gt(int64 *x,int64 *y){ return x -> gt(*y); }
  int64_t int64_ge(int64 *x,int64 *y){ return x -> ge(*y); }
  int64_t int64_lt(int64 *x,int64 *y){ return x -> lt(*y); }
  int64_t int64_le(int64 *x,int64 *y){ return x -> le(*y); }
}

