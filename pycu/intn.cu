#include <cuda_runtime.h>


// CUDA kernels starts from here


// kernels for 16bit integers

__global__ void addKernel(const int16_t *x,const int16_t *y,int16_t *z){ *z = *x + *y; }
__global__ void addKernel(const int16_t *x,const int32_t *y,int32_t *z){ *z = *x + *y; }
__global__ void addKernel(const int16_t *x,const int64_t *y,int64_t *z){ *z = *x + *y; }
__global__ void subKernel(const int16_t *x,const int16_t *y,int16_t *z){ *z = *x - *y; }
__global__ void subKernel(const int16_t *x,const int32_t *y,int32_t *z){ *z = *x - *y; }
__global__ void subKernel(const int16_t *x,const int64_t *y,int64_t *z){ *z = *x - *y; }
__global__ void mulKernel(const int16_t *x,const int16_t *y,int16_t *z){ *z = *x * *y; }
__global__ void mulKernel(const int16_t *x,const int32_t *y,int32_t *z){ *z = *x * *y; }
__global__ void mulKernel(const int16_t *x,const int64_t *y,int64_t *z){ *z = *x * *y; }
__global__ void tdivKernel(const int16_t *x,const int16_t *y,int16_t *z){ *z = *x / *y; }
__global__ void tdivKernel(const int16_t *x,const int32_t *y,int32_t *z){ *z = *x / *y; }
__global__ void tdivKernel(const int16_t *x,const int64_t *y,int64_t *z){ *z = *x / *y; }
__global__ void fdivKernel(const int16_t *x,const int16_t *y,int16_t *z){ *z = *x / *y; }
__global__ void fdivKernel(const int16_t *x,const int32_t *y,int32_t *z){ *z = *x / *y; }
__global__ void fdivKernel(const int16_t *x,const int64_t *y,int64_t *z){ *z = *x / *y; }
__global__ void modKernel(const int16_t *x,const int16_t *y,int16_t *z){ *z = *x % *y; }
__global__ void modKernel(const int16_t *x,const int32_t *y,int32_t *z){ *z = *x % *y; }
__global__ void modKernel(const int16_t *x,const int64_t *y,int64_t *z){ *z = *x % *y; }
__global__ void powKernel(const int16_t *x,const int16_t *y,int16_t *z){ *z = pow(*x,*y); }
__global__ void powKernel(const int16_t *x,const int32_t *y,int32_t *z){ *z = pow(*x,*y); }
__global__ void powKernel(const int16_t *x,const int64_t *y,int64_t *z){ *z = pow(*x,*y); }
__global__ void absKernel(const int16_t *x,int16_t *z){
  if(*x < 0){ *z = -*x; }
  else{ *z = *x; }
}
__global__ void negKernel(const int16_t *x,int16_t *z){ *z = -*x; }
__global__ void posKernel(const int16_t *x,int16_t *z){ *z = +*x; }

__global__ void eqKernel(const int16_t *x,const int16_t *y,int16_t *z){ *z = (*x == *y) ? 1 : 0; }
__global__ void eqKernel(const int16_t *x,const int32_t *y,int32_t *z){ *z = (*x == *y) ? 1 : 0; }
__global__ void eqKernel(const int16_t *x,const int64_t *y,int64_t *z){ *z = (*x == *y) ? 1 : 0; }
__global__ void neKernel(const int16_t *x,const int16_t *y,int16_t *z){ *z = (*x != *y) ? 1 : 0; }
__global__ void neKernel(const int16_t *x,const int32_t *y,int32_t *z){ *z = (*x != *y) ? 1 : 0; }
__global__ void neKernel(const int16_t *x,const int64_t *y,int64_t *z){ *z = (*x != *y) ? 1 : 0; }
__global__ void gtKernel(const int16_t *x,const int16_t *y,int16_t *z){ *z = (*x > *y) ? 1 : 0; }
__global__ void gtKernel(const int16_t *x,const int32_t *y,int32_t *z){ *z = (*x > *y) ? 1 : 0; }
__global__ void gtKernel(const int16_t *x,const int64_t *y,int64_t *z){ *z = (*x > *y) ? 1 : 0; }
__global__ void geKernel(const int16_t *x,const int16_t *y,int16_t *z){ *z = (*x > *y) ? 1 : 0; }
__global__ void geKernel(const int16_t *x,const int32_t *y,int32_t *z){ *z = (*x >= *y) ? 1 : 0; }
__global__ void geKernel(const int16_t *x,const int64_t *y,int64_t *z){ *z = (*x >= *y) ? 1 : 0; }
__global__ void ltKernel(const int16_t *x,const int16_t *y,int16_t *z){ *z = (*x < *y) ? 1 : 0; }
__global__ void ltKernel(const int16_t *x,const int32_t *y,int32_t *z){ *z = (*x < *y) ? 1 : 0; }
__global__ void ltKernel(const int16_t *x,const int64_t *y,int64_t *z){ *z = (*x < *y) ? 1 : 0; }
__global__ void leKernel(const int16_t *x,const int16_t *y,int16_t *z){ *z = (*x <= *y) ? 1 : 0; }
__global__ void leKernel(const int16_t *x,const int32_t *y,int32_t *z){ *z = (*x <= *y) ? 1 : 0; }
__global__ void leKernel(const int16_t *x,const int64_t *y,int64_t *z){ *z = (*x <= *y) ? 1 : 0; }


// kerneles for 32bit integers 

__global__ void addKernel(const int32_t *x,const int32_t *y,int32_t *z){ *z = *x + *y; }
__global__ void addKernel(const int32_t *x,const int16_t *y,int32_t *z){ *z = *x + *y; }
__global__ void addKernel(const int32_t *x,const int64_t *y,int64_t *z){ *z = *x + *y; }
__global__ void subKernel(const int32_t *x,const int32_t *y,int32_t *z){ *z = *x - *y; }
__global__ void subKernel(const int32_t *x,const int16_t *y,int32_t *z){ *z = *x - *y; } 
__global__ void subKernel(const int32_t *x,const int64_t *y,int64_t *z){ *z = *x - *y; }
__global__ void mulKernel(const int32_t *x,const int32_t *y,int32_t *z){ *z = *x * *y; }
__global__ void mulKernel(const int32_t *x,const int16_t *y,int32_t *z){ *z = *x * *y; }
__global__ void mulKernel(const int32_t *x,const int64_t *y,int64_t *z){ *z = *x * *y; }
__global__ void tdivKernel(const int32_t *x,const int32_t *y,int32_t *z){ *z = *x / *y; }
__global__ void tdivKernel(const int32_t *x,const int16_t *y,int32_t *z){ *z = *x / *y; }
__global__ void tdivKernel(const int32_t *x,const int64_t *y,int64_t *z){ *z = *x / *y; }
__global__ void fdivKernel(const int32_t *x,const int32_t *y,int32_t *z){ *z = *x / *y; }
__global__ void fdivKernel(const int32_t *x,const int16_t *y,int32_t *z){ *z = *x / *y; }
__global__ void fdivKernel(const int32_t *x,const int64_t *y,int64_t *z){ *z = *x / *y; }
__global__ void modKernel(const int32_t *x,const int32_t *y,int32_t *z){ *z = *x % *y; }
__global__ void modKernel(const int32_t *x,const int16_t *y,int32_t *z){ *z = *x % *y; }
__global__ void modKernel(const int32_t *x,const int64_t *y,int64_t *z){ *z = *x % *y; }
__global__ void powKernel(const int32_t *x,const int32_t *y,int32_t *z){ *z = pow(*x,*y); }
__global__ void powKernel(const int32_t *x,const int16_t *y,int32_t *z){ *z = pow(*x,*y); }
__global__ void powKernel(const int32_t *x,const int64_t *y,int64_t *z){ *z = pow(*x,*y); }
__global__ void absKernel(const int32_t *x,int32_t *z){
  if(*x < 0){ *z = -*x; }
  else{ *z = *x; }
}
__global__ void negKernel(const int32_t *x,int32_t *z){ *z = -*x; }
__global__ void posKernel(const int32_t *x,int32_t *z){ *z = +*x; }

__global__ void eqKernel(const int32_t *x,const int32_t *y,int32_t *z){ *z = (*x == *y) ? 1 : 0; }
__global__ void eqKernel(const int32_t *x,const int16_t *y,int32_t *z){ *z = (*x == *y) ? 1 : 0; }
__global__ void eqKernel(const int32_t *x,const int64_t *y,int64_t *z){ *z = (*x == *y) ? 1 : 0; }
__global__ void neKernel(const int32_t *x,const int32_t *y,int32_t *z){ *z = (*x != *y) ? 1 : 0; }
__global__ void neKernel(const int32_t *x,const int16_t *y,int32_t *z){ *z = (*x != *y) ? 1 : 0; }
__global__ void neKernel(const int32_t *x,const int64_t *y,int64_t *z){ *z = (*x != *y) ? 1 : 0; }
__global__ void gtKernel(const int32_t *x,const int32_t *y,int32_t *z){ *z = (*x > *y) ? 1 : 0; }
__global__ void gtKernel(const int32_t *x,const int16_t *y,int32_t *z){ *z = (*x > *y) ? 1 : 0; }
__global__ void gtKernel(const int32_t *x,const int64_t *y,int64_t *z){ *z = (*x > *y) ? 1 : 0; }
__global__ void geKernel(const int32_t *x,const int32_t *y,int32_t *z){ *z = (*x >= *y) ? 1 : 0; }
__global__ void geKernel(const int32_t *x,const int16_t *y,int32_t *z){ *z = (*x >= *y) ? 1 : 0; }
__global__ void geKernel(const int32_t *x,const int64_t *y,int64_t *z){ *z = (*x >= *y) ? 1 : 0; }
__global__ void ltKernel(const int32_t *x,const int32_t *y,int32_t *z){ *z = (*x < *y) ? 1 : 0; }
__global__ void ltKernel(const int32_t *x,const int16_t *y,int32_t *z){ *z = (*x < *y) ? 1 : 0; }
__global__ void ltKernel(const int32_t *x,const int64_t *y,int64_t *z){ *z = (*x < *y) ? 1 : 0; }
__global__ void leKernel(const int32_t *x,const int32_t *y,int32_t *z){ *z = (*x <= *y) ? 1 : 0; }
__global__ void leKernel(const int32_t *x,const int16_t *y,int32_t *z){ *z = (*x <= *y) ? 1 : 0; }
__global__ void leKernel(const int32_t *x,const int64_t *y,int64_t *z){ *z = (*x <= *y) ? 1 : 0; }


// kernels for 64bit integers

__global__ void addKernel(const int64_t *x,const int16_t *y,int64_t *z){ *z = *x + *y; }
__global__ void addKernel(const int64_t *x,const int32_t *y,int64_t *z){ *z = *x + *y; }
__global__ void addKernel(const int64_t *x,const int64_t *y,int64_t *z){ *z = *x + *y; }
__global__ void subKernel(const int64_t *x,const int16_t *y,int64_t *z){ *z = *x - *y; }
__global__ void subKernel(const int64_t *x,const int32_t *y,int64_t *z){ *z = *x - *y; }
__global__ void subKernel(const int64_t *x,const int64_t *y,int64_t *z){ *z = *x - *y; }
__global__ void mulKernel(const int64_t *x,const int16_t *y,int64_t *z){ *z = *x * *y; }
__global__ void mulKernel(const int64_t *x,const int32_t *y,int64_t *z){ *z = *x * *y; }
__global__ void mulKernel(const int64_t *x,const int64_t *y,int64_t *z){ *z = *x * *y; }
__global__ void tdivKernel(const int64_t *x,const int16_t *y,int64_t *z){ *z = *x / *y; }
__global__ void tdivKernel(const int64_t *x,const int32_t *y,int64_t *z){ *z = *x / *y; }
__global__ void tdivKernel(const int64_t *x,const int64_t *y,int64_t *z){ *z = *x / *y; }
__global__ void fdivKernel(const int64_t *x,const int16_t *y,int64_t *z){ *z = *x / *y; }
__global__ void fdivKernel(const int64_t *x,const int32_t *y,int64_t *z){ *z = *x / *y; }
__global__ void fdivKernel(const int64_t *x,const int64_t *y,int64_t *z){ *z = *x / *y; }
__global__ void modKernel(const int64_t *x,const int16_t *y,int64_t *z){ *z = *x % *y; }
__global__ void modKernel(const int64_t *x,const int32_t *y,int64_t *z){ *z = *x % *y; }
__global__ void modKernel(const int64_t *x,const int64_t *y,int64_t *z){ *z = *x % *y; }
__global__ void powKernel(const int64_t *x,const int16_t *y,int64_t *z){ *z = pow(*x,*y); }
__global__ void powKernel(const int64_t *x,const int32_t *y,int64_t *z){ *z = pow(*x,*y); }
__global__ void powKernel(const int64_t *x,const int64_t *y,int64_t *z){ *z = pow(*x,*y); }
__global__ void absKernel(const int64_t *x,int64_t *z){
  if(*x < 0){ *z = -*x; }
  else{ *z = *x; }
}
__global__ void negKernel(const int64_t *x,int64_t *z){ *z = -*x; }
__global__ void posKernel(const int64_t *x,int64_t *z){ *z = +*x; }

__global__ void eqKernel(const int64_t *x,const int16_t *y,int64_t *z){ *z = (*x == *y) ? 1 : 0; }
__global__ void eqKernel(const int64_t *x,const int32_t *y,int64_t *z){ *z = (*x == *y) ? 1 : 0; }
__global__ void eqKernel(const int64_t *x,const int64_t *y,int64_t *z){ *z = (*x == *y) ? 1 : 0; }
__global__ void neKernel(const int64_t *x,const int16_t *y,int64_t *z){ *z = (*x != *y) ? 1 : 0; }
__global__ void neKernel(const int64_t *x,const int32_t *y,int64_t *z){ *z = (*x != *y) ? 1 : 0; }
__global__ void neKernel(const int64_t *x,const int64_t *y,int64_t *z){ *z = (*x != *y) ? 1 : 0; }
__global__ void gtKernel(const int64_t *x,const int16_t *y,int64_t *z){ *z = (*x > *y) ? 1 : 0; }
__global__ void gtKernel(const int64_t *x,const int32_t *y,int64_t *z){ *z = (*x > *y) ? 1 : 0; }
__global__ void gtKernel(const int64_t *x,const int64_t *y,int64_t *z){ *z = (*x > *y) ? 1 : 0; }
__global__ void geKernel(const int64_t *x,const int16_t *y,int64_t *z){ *z = (*x >= *y) ? 1 : 0; }
__global__ void geKernel(const int64_t *x,const int32_t *y,int64_t *z){ *z = (*x >= *y) ? 1 : 0; }
__global__ void geKernel(const int64_t *x,const int64_t *y,int64_t *z){ *z = (*x >= *y) ? 1 : 0; }
__global__ void ltKernel(const int64_t *x,const int16_t *y,int64_t *z){ *z = (*x < *y) ? 1 : 0; }
__global__ void ltKernel(const int64_t *x,const int32_t *y,int64_t *z){ *z = (*x < *y) ? 1 : 0; }
__global__ void ltKernel(const int64_t *x,const int64_t *y,int64_t *z){ *z = (*x < *y) ? 1 : 0; }
__global__ void leKernel(const int64_t *x,const int16_t *y,int64_t *z){ *z = (*x <= *y) ? 1 : 0; }
__global__ void leKernel(const int64_t *x,const int32_t *y,int64_t *z){ *z = (*x <= *y) ? 1 : 0; }
__global__ void leKernel(const int64_t *x,const int64_t *y,int64_t *z){ *z = (*x <= *y) ? 1 : 0; }


// CUDA kernels ends here




class int16 ;
class int32 ;
class int64 ;


class int16{
  private:
    int16_t *value;

  public:
    int16(const int16_t &other){
      cudaMalloc(&value,sizeof(int16_t));
      cudaMemcpy(value,&other,sizeof(int16_t),cudaMemcpyHostToDevice);
    };

    ~int16(){
      if(value){ cudaFree(value); }
    };

    int16_t getValue() const {
      int16_t host_value;
      cudaMemcpy(&host_value,value,sizeof(int16_t),cudaMemcpyDeviceToHost);
      return host_value;
    }

    int16_t *deviceValue() const { return this -> value; }

    int16_t *copyToHost() const {
      int16_t *host_value = new int16_t;
      cudaMemcpy(host_value,value,sizeof(int16_t),cudaMemcpyDeviceToHost);
      return host_value;
    }

    int16 add(const int16 &other) const ;
    int16 sub(const int16 &other) const ;
    int16 mul(const int16 &other) const ;
    int16 tdiv(const int16 &other) const ;
    int16 fdiv(const int16 &other) const ;
    int16 mod(const int16 &other) const ;
    int16 pow(const int16 &other) const;

    int16_t eq(const int16 &other) const ;
    int16_t ne(const int16 &other) const ;
    int16_t gt(const int16 &other) const ;
    int16_t ge(const int16 &other) const ;
    int16_t lt(const int16 &other) const ;
    int16_t le(const int16 &other) const ;
    
    int16 abs() const ;
    int16 neg() const ;
    int16 pos() const ;

    int32 add(const int32 &other) const ;
    int32 sub(const int32 &other) const ;
    int32 mul(const int32 &other) const ;
    int32 tdiv(const int32 &other) const ;
    int32 fdiv(const int32 &other) const ;
    int32 mod(const int32 &other) const ;
    int32 pow(const int32 &other) const ;

    int32_t eq(const int32 &other) const ;
    int32_t ne(const int32 &other) const ;
    int32_t gt(const int32 &other) const ;
    int32_t ge(const int32 &other) const ;
    int32_t lt(const int32 &other) const ;
    int32_t le(const int32 &other) const ;

    int64 add(const int64 &other) const ;
    int64 sub(const int64 &other) const ;
    int64 mul(const int64 &other) const ;
    int64 tdiv(const int64 &other) const ;
    int64 fdiv(const int64 &other) const ;
    int64 mod(const int64 &other) const ;
    int64 pow(const int64 &other) const ;

    int64_t eq(const int64 &other) const ;
    int64_t ne(const int64 &other) const ;
    int64_t gt(const int64 &other) const ;
    int64_t ge(const int64 &other) const ;
    int64_t lt(const int64 &other) const ;
    int64_t le(const int64 &other) const ;
};


class int32{
  private:
  int32_t *value;

  public:
    int32(const int32_t &other){
      cudaMalloc(&value,sizeof(int32_t));
      cudaMemcpy(value,&other,sizeof(int32_t),cudaMemcpyHostToDevice);
    };

    ~int32(){ if(value){ cudaFree(value); } };

    int32_t getValue() const {
      int32_t host_value;
      cudaMemcpy(&host_value,value,sizeof(int32_t),cudaMemcpyDeviceToHost);
      return host_value;
    }

    int32_t *deviceValue() const { return this -> value; }

    int32_t *copyToHost() const {
      int32_t *host_value = new int32_t;
      cudaMemcpy(host_value,value,sizeof(int32_t),cudaMemcpyDeviceToHost);
      return host_value;
    }

    int32 add(const int32 &other) const ;
    int32 sub(const int32 &other) const ;
    int32 mul(const int32 &other) const ;
    int32 tdiv(const int32 &other) const ;
    int32 fdiv(const int32 &other) const ;
    int32 mod(const int32 &other) const ;
    int32 pow(const int32 &other) const ;

    int32_t eq(const int32 &other) const ;
    int32_t ne(const int32 &other) const ;
    int32_t gt(const int32 &other) const ;
    int32_t ge(const int32 &other) const ;
    int32_t lt(const int32 &other) const ;
    int32_t le(const int32 &other) const ;

    int32 abs() const ;
    int32 neg() const ;
    int32 pos() const ;

    int32 add(const int16 &other) const ;
    int32 sub(const int16 &other) const ;
    int32 mul(const int16 &other) const ;
    int32 tdiv(const int16 &other) const ;
    int32 fdiv(const int16 &other) const ;
    int32 mod(const int16 &other) const ;
    int32 pow(const int16 &other) const ;

    int32_t eq(const int16 &other) const ;
    int32_t ne(const int16 &other) const ;
    int32_t gt(const int16 &other) const ;
    int32_t ge(const int16 &other) const ;
    int32_t lt(const int16 &other) const ;
    int32_t le(const int16 &other) const ;

    int64 add(const int64 &other) const ;
    int64 sub(const int64 &other) const ;
    int64 mul(const int64 &other) const ;
    int64 tdiv(const int64 &other) const ;
    int64 fdiv(const int64 &other) const ;
    int64 mod(const int64 &other) const ;
    int64 pow(const int64 &other) const ;

    int64_t eq(const int64 &other) const ;
    int64_t ne(const int64 &other) const ;
    int64_t gt(const int64 &other) const ;
    int64_t ge(const int64 &other) const ;
    int64_t lt(const int64 &other) const ;
    int64_t le(const int64 &other) const ;
};


class int64{
  private:
    int64_t *value;
  public:
    int64(const int64_t &other){
      cudaMalloc(&value,sizeof(int64_t));
      cudaMemcpy(value,&other,sizeof(int64_t),cudaMemcpyHostToDevice);
    }

    ~int64(){ if(value){ cudaFree(value); } }

    int64_t getValue() const {
      int64_t host_value;
      cudaMemcpy(&host_value,value,sizeof(int64_t),cudaMemcpyDeviceToHost);
      return host_value;
    }

    int64_t *deviceValue() const { return this -> value; }

    int64_t *copyToHost() const {
      int64_t *host_value = new int64_t;
      cudaMemcpy(host_value,value,sizeof(int64_t),cudaMemcpyDeviceToHost);
      return host_value;
    }

    int64 add(const int64 &other) const ;
    int64 sub(const int64 &other) const ;
    int64 mul(const int64 &other) const ;
    int64 tdiv(const int64 &other) const ;
    int64 fdiv(const int64 &other) const ;
    int64 mod(const int64 &other) const ;
    int64 pow(const int64 &other) const ;

    int64_t eq(const int64 &other) const ;
    int64_t ne(const int64 &other) const ;
    int64_t gt(const int64 &other) const ;
    int64_t ge(const int64 &other) const ;
    int64_t lt(const int64 &other) const ;
    int64_t le(const int64 &other) const ;

    int64 abs() const ;
    int64 neg() const ;
    int64 pos() const ;

    int64 add(const int32 &other) const ;
    int64 sub(const int32 &other) const ;
    int64 mul(const int32 &other) const ;
    int64 tdiv(const int32 &other) const ;
    int64 fdiv(const int32 &other) const ;
    int64 mod(const int32 &other) const ;
    int64 pow(const int32 &other) const ;

    int64_t eq(const int32 &other) const ;
    int64_t ne(const int32 &other) const ;
    int64_t gt(const int32 &other) const ;
    int64_t ge(const int32 &other) const ;
    int64_t lt(const int32 &other) const ;
    int64_t le(const int32 &other) const ;

    int64 add(const int16 &other) const ;
    int64 sub(const int16 &other) const ;
    int64 mul(const int16 &other) const ;
    int64 tdiv(const int16 &other) const ;
    int64 fdiv(const int16 &other) const ;
    int64 mod(const int16 &other) const ;
    int64 pow(const int16 &other) const ;

    int64_t eq(const int16 &other) const ;
    int64_t ne(const int16 &other) const ;
    int64_t gt(const int16 &other) const ;
    int64_t ge(const int16 &other) const ;
    int64_t lt(const int16 &other) const ;
    int64_t le(const int16 &other) const ;
};




// for int16

int16 int16::add(const int16 &other) const {
  int16 result(0);
  addKernel<<<1,1>>>(this -> value,other.value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int16 int16::sub(const int16 &other) const {
  int16 result(0);
  subKernel<<<1,1>>>(this -> value,other.value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int16 int16::mul(const int16 &other) const {
  int16 result(0);
  mulKernel<<<1,1>>>(this -> value,other.value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int16 int16::tdiv(const int16 &other) const {
  int16 result(0);
  tdivKernel<<<1,1>>>(this -> value,other.value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int16 int16::fdiv(const int16 &other) const {
  int16 result(0);
  fdivKernel<<<1,1>>>(this -> value,other.value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int16 int16::mod(const int16 &other) const {
  int16 result(0);
  modKernel<<<1,1>>>(this -> value,other.value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int16 int16::pow(const int16 &other) const {
  int16 result(0);
  powKernel<<<1,1>>>(this -> value,other.value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int16 int16::abs() const {
  int16 result(0);
  absKernel<<<1,1>>>(this -> value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int16 int16::neg() const {
  int16 result(0);
  negKernel<<<1,1>>>(this -> value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int16 int16::pos() const {
  int16 result(0);
  posKernel<<<1,1>>>(this -> value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int16_t int16::eq(const int16 &other) const {
  int16_t host_value;
  int16_t *device_value;
  cudaMalloc(&device_value,sizeof(int16_t));
  eqKernel<<<1,1>>>(this -> value,other.value,device_value);
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
  neKernel<<<1,1>>>(this -> value,other.value,device_value);
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
  gtKernel<<<1,1>>>(this -> value,other.value,device_value);
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
  geKernel<<<1,1>>>(this -> value,other.value,device_value);
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
  ltKernel<<<1,1>>>(this -> value,other.value,device_value);
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
  leKernel<<<1,1>>>(this -> value,other.value,device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int16_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int32 int16::add(const int32 &other) const {
  int32 result(0);
  addKernel<<<1,1>>>(this -> value,other.deviceValue(),result.deviceValue());
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}


int32 int16::sub(const int32 &other) const {
  int32 result(0);
  subKernel<<<1,1>>>(this -> value,other.deviceValue(),result.deviceValue());
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int16::mul(const int32 &other) const {
  int32 result(0);
  mulKernel<<<1,1>>>(this -> value,other.deviceValue(),result.deviceValue());
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int16::tdiv(const int32 &other) const {
  int32 result(0);
  tdivKernel<<<1,1>>>(this -> value,other.deviceValue(),result.deviceValue());
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int16::fdiv(const int32 &other) const {
  int32 result(0);
  fdivKernel<<<1,1>>>(this -> value,other.deviceValue(),result.deviceValue());
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int16::mod(const int32 &other) const {
  int32 result(0);
  modKernel<<<1,1>>>(this -> value,other.deviceValue(),result.deviceValue());
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int16::pow(const int32 &other) const {
  int32 result(0);
  powKernel<<<1,1>>>(this -> value,other.deviceValue(),result.deviceValue());
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32_t int16::eq(const int32 &other) const {
  int32_t host_value;
  int32_t *device_value;
  cudaMalloc(&device_value,sizeof(int32_t));
  eqKernel<<<1,1>>>(this -> value,other.deviceValue(),device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int32_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int32_t int16::ne(const int32 &other) const {
  int32_t host_value;
  int32_t *device_value;
  cudaMalloc(&device_value,sizeof(int32_t));
  neKernel<<<1,1>>>(this -> value,other.deviceValue(),device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int32_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int32_t int16::gt(const int32 &other) const {
  int32_t host_value;
  int32_t *device_value;
  cudaMalloc(&device_value,sizeof(int32_t));
  gtKernel<<<1,1>>>(this -> value,other.deviceValue(),device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int32_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int32_t int16::ge(const int32 &other) const {
  int32_t host_value;
  int32_t *device_value;
  cudaMalloc(&device_value,sizeof(int32_t));
  geKernel<<<1,1>>>(this -> value,other.deviceValue(),device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int32_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int32_t int16::lt(const int32 &other) const {
  int32_t host_value;
  int32_t *device_value;
  cudaMalloc(&device_value,sizeof(int32_t));
  ltKernel<<<1,1>>>(this -> value,other.deviceValue(),device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int32_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int32_t int16::le(const int32 &other) const {
  int32_t host_value;
  int32_t *device_value;
  cudaMalloc(&device_value,sizeof(int32_t));
  leKernel<<<1,1>>>(this -> value,other.deviceValue(),device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int32_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int64 int16::add(const int64 &other) const {
  int64 result(0);
  addKernel<<<1,1>>>(this -> value,other.deviceValue(),result.deviceValue());
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int16::sub(const int64 &other) const {
  int64 result(0);
  subKernel<<<1,1>>>(this -> value,other.deviceValue(),result.deviceValue());
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int16::mul(const int64 &other) const {
  int64 result(0);
  mulKernel<<<1,1>>>(this -> value,other.deviceValue(),result.deviceValue());
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int16::tdiv(const int64 &other) const {
  int64 result(0);
  tdivKernel<<<1,1>>>(this -> value,other.deviceValue(),result.deviceValue());
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int16::fdiv(const int64 &other) const {
  int64 result(0);
  fdivKernel<<<1,1>>>(this -> value,other.deviceValue(),result.deviceValue());
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int16::mod(const int64 &other) const {
  int64 result(0);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int16::pow(const int64 &other) const {
  int64 result(0);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64_t int16::eq(const int64 &other) const {
  int64_t host_value;
  int64_t *device_value;
  cudaMalloc(&device_value,sizeof(int64_t));
  eqKernel<<<1,1>>>(this -> value,other.deviceValue(),device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int64_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int64_t int16::ne(const int64 &other) const {
  int64_t host_value;
  int64_t *device_value;
  cudaMalloc(&device_value,sizeof(int64_t));
  neKernel<<<1,1>>>(this -> value,other.deviceValue(),device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int64_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int64_t int16::gt(const int64 &other) const {
  int64_t host_value;
  int64_t *device_value;
  cudaMalloc(&device_value,sizeof(int64_t));
  gtKernel<<<1,1>>>(this -> value,other.deviceValue(),device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int64_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int64_t int16::ge(const int64 &other) const {
  int64_t host_value;
  int64_t *device_value;
  cudaMalloc(&device_value,sizeof(int64_t));
  geKernel<<<1,1>>>(this -> value,other.deviceValue(),device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int64_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int64_t int16::lt(const int64 &other) const {
  int64_t host_value;
  int64_t *device_value;
  cudaMalloc(&device_value,sizeof(int64_t));
  ltKernel<<<1,1>>>(this -> value,other.deviceValue(),device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int64_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int64_t int16::le(const int64 &other) const {
  int64_t host_value;
  int64_t *device_value;
  cudaMalloc(&device_value,sizeof(int64_t));
  leKernel<<<1,1>>>(this -> value,other.deviceValue(),device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int64_t),cudaMemcpyDeviceToHost);
  return host_value;
}

// for int32

int32 int32::add(const int32 &other) const {
  int32 result(0);
  addKernel<<<1,1>>>(this -> value,other.value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int32::sub(const int32 &other) const {
  int32 result(0);
  subKernel<<<1,1>>>(this -> value,other.value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int32::mul(const int32 &other) const {
  int32 result(0);
  mulKernel<<<1,1>>>(this -> value,other.value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int32::tdiv(const int32 &other) const {
  int32 result(0);
  tdivKernel<<<1,1>>>(this -> value,other.value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int32::fdiv(const int32 &other) const {
  int32 result(0);
  fdivKernel<<<1,1>>>(this -> value,other.value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int32::mod(const int32 &other) const {
  int32 result(0);
  modKernel<<<1,1>>>(this -> value,other.value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int32::pow(const int32 &other) const {
  int32 result(0);
  powKernel<<<1,1>>>(this -> value,other.value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32_t int32::eq(const int32 &other) const {
  int32_t host_value;
  int32_t *device_value;
  cudaMalloc(&device_value,sizeof(int32_t));
  eqKernel<<<1,1>>>(this -> value,other.value,device_value);
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
  neKernel<<<1,1>>>(this -> value,other.value,device_value);
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
  gtKernel<<<1,1>>>(this -> value,other.value,device_value);
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
  geKernel<<<1,1>>>(this -> value,other.value,device_value);
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
  ltKernel<<<1,1>>>(this -> value,other.value,device_value);
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
  cudaGetLastError();
  cudaDeviceSynchronize();
  leKernel<<<1,1>>>(this -> value,other.value,device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int32_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int32 int32::abs() const {
  int32 result(0);
  absKernel<<<1,1>>>(this -> value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int32::neg() const {
  int32 result(0);
  negKernel<<<1,1>>>(this -> value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int32::pos() const {
  int32 result(0);
  posKernel<<<1,1>>>(this -> value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int32::add(const int16 &other) const {
  int32 result(0);
  addKernel<<<1,1>>>(this -> value,other.deviceValue(),result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int32::sub(const int16 &other) const {
  int32 result(0);
  subKernel<<<1,1>>>(this -> value,other.deviceValue(),result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int32::mul(const int16 &other) const {
  int32 result(0);
  mulKernel<<<1,1>>>(this -> value,other.deviceValue(),result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int32::tdiv(const int16 &other) const {
  int32 result(0);
  tdivKernel<<<1,1>>>(this -> value,other.deviceValue(),result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int32::fdiv(const int16 &other) const {
  int32 result(0);
  fdivKernel<<<1,1>>>(this -> value,other.deviceValue(),result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int32::mod(const int16 &other) const {
  int32 result(0);
  modKernel<<<1,1>>>(this -> value,other.deviceValue(),result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32 int32::pow(const int16 &other) const {
  int32 result(0);
  powKernel<<<1,1>>>(this -> value,other.deviceValue(),result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int32_t int32::eq(const int16 &other) const {
  int32_t host_value;
  int32_t *device_value;
  cudaMalloc(&device_value,sizeof(int32_t));
  eqKernel<<<1,1>>>(this -> value,other.deviceValue(),device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int32_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int32_t int32::ne(const int16 &other) const {
  int32_t host_value;
  int32_t *device_value;
  cudaMalloc(&device_value,sizeof(int32_t));
  neKernel<<<1,1>>>(this -> value,other.deviceValue(),device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int32_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int32_t int32::gt(const int16 &other) const {
  int32_t host_value;
  int32_t *device_value;
  cudaMalloc(&device_value,sizeof(int32_t));
  gtKernel<<<1,1>>>(this -> value,other.deviceValue(),device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int32_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int32_t int32::ge(const int16 &other) const {
  int32_t host_value;
  int32_t *device_value;
  cudaMalloc(&device_value,sizeof(int32_t));
  geKernel<<<1,1>>>(this -> value,other.deviceValue(),device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int32_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int32_t int32::lt(const int16 &other) const {
  int32_t host_value;
  int32_t *device_value;
  cudaMalloc(&device_value,sizeof(int32_t));
  ltKernel<<<1,1>>>(this -> value,other.deviceValue(),device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int32_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int32_t int32::le(const int16 &other) const {
  int32_t host_value;
  int32_t *device_value;
  cudaMalloc(&device_value,sizeof(int32_t));
  leKernel<<<1,1>>>(this -> value,other.deviceValue(),device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int32_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int64 int32::add(const int64 &other) const {
  int64 result(0);
  addKernel<<<1,1>>>(this -> value,other.deviceValue(),result.deviceValue());
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int32::sub(const int64 &other) const {
  int64 result(0);
  subKernel<<<1,1>>>(this -> value,other.deviceValue(),result.deviceValue());
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int32::mul(const int64 &other) const {
  int64 result(0);
  mulKernel<<<1,1>>>(this -> value,other.deviceValue(),result.deviceValue());
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int32::tdiv(const int64 &other) const {
  int64 result(0);
  tdivKernel<<<1,1>>>(this -> value,other.deviceValue(),result.deviceValue());
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int32::fdiv(const int64 &other) const {
  int64 result(0);
  fdivKernel<<<1,1>>>(this -> value,other.deviceValue(),result.deviceValue());
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int32::mod(const int64 &other) const {
  int64 result(0);
  modKernel<<<1,1>>>(this -> value,other.deviceValue(),result.deviceValue());
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int32::pow(const int64 &other) const {
  int64 result(0);
  powKernel<<<1,1>>>(this -> value,other.deviceValue(),result.deviceValue());
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64_t int32::eq(const int64 &other) const {
  int64_t host_value;
  int64_t *device_value;
  cudaMalloc(&device_value,sizeof(int64_t));
  eqKernel<<<1,1>>>(this -> value,other.deviceValue(),device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int64_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int64_t int32::ne(const int64 &other) const {
  int64_t host_value;
  int64_t *device_value;
  cudaMalloc(&device_value,sizeof(int64_t));
  neKernel<<<1,1>>>(this -> value,other.deviceValue(),device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int64_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int64_t int32::gt(const int64 &other) const {
  int64_t host_value;
  int64_t *device_value;
  cudaMalloc(&device_value,sizeof(int64_t));
  gtKernel<<<1,1>>>(this -> value,other.deviceValue(),device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int64_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int64_t int32::ge(const int64 &other) const {
  int64_t host_value;
  int64_t *device_value;
  cudaMalloc(&device_value,sizeof(int64_t));
  geKernel<<<1,1>>>(this -> value,other.deviceValue(),device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int64_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int64_t int32::lt(const int64 &other) const {
  int64_t host_value;
  int64_t *device_value;
  cudaMalloc(&device_value,sizeof(int64_t));
  ltKernel<<<1,1>>>(this -> value,other.deviceValue(),device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int64_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int64_t int32::le(const int64 &other) const {
  int64_t host_value;
  int64_t *device_value;
  cudaMalloc(&device_value,sizeof(int64_t));
  leKernel<<<1,1>>>(this -> value,other.deviceValue(),device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int64_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int64 int64::add(const int64 &other) const {
  int64 result(0);
  addKernel<<<1,1>>>(this -> value,other.deviceValue(),result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int64::sub(const int64 &other) const {
  int64 result(0);
  subKernel<<<1,1>>>(this -> value,other.deviceValue(),result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int64::mul(const int64 &other) const {
  int64 result(0);
  mulKernel<<<1,1>>>(this -> value,other.deviceValue(),result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int64::tdiv(const int64 &other) const {
  int64 result(0);
  tdivKernel<<<1,1>>>(this -> value,other.value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int64::fdiv(const int64 &other) const {
  int64 result(0);
  fdivKernel<<<1,1>>>(this -> value,other.value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int64::mod(const int64 &other) const {
  int64 result(0);
  modKernel<<<1,1>>>(this -> value,other.value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int64::pow(const int64 &other) const {
  int64 result(0);
  powKernel<<<1,1>>>(this -> value,other.value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64_t int64::eq(const int64 &other) const {
  int64_t host_value;
  int64_t *device_value;
  cudaMalloc(&device_value,sizeof(int64_t));
  eqKernel<<<1,1>>>(this -> value,other.value,device_value);
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
  neKernel<<<1,1>>>(this -> value,other.value,device_value);
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
  gtKernel<<<1,1>>>(this -> value,other.value,device_value);
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
  geKernel<<<1,1>>>(this -> value,other.value,device_value);
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
  ltKernel<<<1,1>>>(this -> value,other.value,device_value);
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
  leKernel<<<1,1>>>(this -> value,other.value,device_value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(&host_value,device_value,sizeof(int64_t),cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  return host_value;
}

int64 int64::abs() const {
  int64 result(0);
  absKernel<<<1,1>>>(this -> value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int64::neg() const {
  int64 result(0);
  negKernel<<<1,1>>>(this -> value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int64::pos() const {
  int64 result(0);
  posKernel<<<1,1>>>(this -> value,result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int64::add(const int32 &other) const {
  int64 result(0);
  addKernel<<<1,1>>>(this -> value,other.deviceValue(),result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int64::sub(const int32 &other) const {
  int64 result(0);
  subKernel<<<1,1>>>(this -> value,other.deviceValue(),result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int64::mul(const int32 &other) const {
  int64 result(0);
  mulKernel<<<1,1>>>(this -> value,other.deviceValue(),result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return resultl;
}

int64 int64::tdiv(const int32 &other) const {
  int64 result(0);
  tdivKernel<<<1,1>>>(this -> value,other.deviceValue(),result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int64::fdiv(const int32 &other) const {
  int64 result(0);
  fdivKernel<<<1,1>>>(this -> value,other.deviceValue(),result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int64::mod(const int32 &other) const {
  int64 result(0);
  modKernel<<<1,1>>>(this -> value,other.deviceValue(),result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}

int64 int64::pow(const int32 &other) const {
  int64 result(0);
  powKernel<<<1,1>>>(this -> value,other.deviceValue(),result.value);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return result;
}















