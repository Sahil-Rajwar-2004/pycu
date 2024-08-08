import os
from ctypes import (
  CDLL,
  Structure,
  POINTER,
  c_int16,
  c_int32,
  c_int64
)

if os.name == "nt": raise OSError(f"this library isn't compatible with Windows OS, instead you can use WSL!")
elif os.name == "posix": PATH = os.path.join(os.path.dirname(__file__),"dtypes.so")

if not os.path.exists(PATH): raise FileNotFoundError("couldn't find the file dtyes.so")

lib = CDLL(PATH)

class Int16(Structure): _fields_ = [("value",c_int16)]
class Int32(Structure): _fields_ = [("value",c_int32)]
class Int64(Structure): _fields_ = [("value",c_int64)]

lib.int16_new.argtypes = [c_int16]
lib.int16_new.restype = POINTER(Int16)
lib.int16_delete.argtypes = [POINTER(Int16)]
lib.int16_delete.restype = None
lib.int16_value.argtypes = [POINTER(Int16)]
lib.int16_value.restype = c_int16
lib.int16_add.argtypes = [POINTER(Int16),POINTER(Int16)]
lib.int16_add.restype = POINTER(Int16)
lib.int16_sub.argtypes = [POINTER(Int16),POINTER(Int16)]
lib.int16_sub.restype = POINTER(Int16)
lib.int16_mul.argtypes = [POINTER(Int16),POINTER(Int16)]
lib.int16_mul.restype = POINTER(Int16)
lib.int16_tdiv.argtypes = [POINTER(Int16),POINTER(Int16)]
lib.int16_tdiv.restype = POINTER(Int16)
lib.int16_fdiv.argtypes = [POINTER(Int16),POINTER(Int16)]
lib.int16_fdiv.restype = POINTER(Int16)
lib.int16_mod.argtypes = [POINTER(Int16),POINTER(Int16)]
lib.int16_mod.restype = POINTER(Int16)
lib.int16_pow.argtypes = [POINTER(Int16),POINTER(Int16)]
lib.int16_pow.restype = POINTER(Int16)
lib.int16_abs.argtypes = [POINTER(Int16)]
lib.int16_abs.restype = POINTER(Int16)
lib.int16_neg.argtypes = [POINTER(Int16)]
lib.int16_neg.restype = POINTER(Int16)
lib.int16_pos.argtypes = [POINTER(Int16)]
lib.int16_pos.restype = POINTER(Int16)


lib.int32_new.argtypes = [c_int32]
lib.int32_new.restype = POINTER(Int32)
lib.int32_delete.argtypes = [POINTER(Int32)]
lib.int32_delete.restype = None
lib.int32_value.argtypes = [POINTER(Int32)]
lib.int32_value.restype = c_int32
lib.int32_add.argtypes = [POINTER(Int32),POINTER(Int32)]
lib.int32_add.restype = POINTER(Int32)
lib.int32_sub.argtypes = [POINTER(Int32),POINTER(Int32)]
lib.int32_sub.restype = POINTER(Int32)
lib.int32_mul.argtypes = [POINTER(Int32),POINTER(Int32)]
lib.int32_mul.restype = POINTER(Int32)
lib.int32_tdiv.argtypes = [POINTER(Int32),POINTER(Int32)]
lib.int32_tdiv.restype = POINTER(Int32)
lib.int32_fdiv.argtypes = [POINTER(Int32),POINTER(Int32)]
lib.int32_fdiv.restype = POINTER(Int32)
lib.int32_mod.argtypes = [POINTER(Int32),POINTER(Int32)]
lib.int32_mod.restype = POINTER(Int32)
lib.int32_pow.argtypes = [POINTER(Int32),POINTER(Int32)]
lib.int32_pow.restype = POINTER(Int32)
lib.int32_abs.argtypes = [POINTER(Int32)]
lib.int32_abs.restype = POINTER(Int32)
lib.int32_neg.argtypes = [POINTER(Int32)]
lib.int32_neg.restype = POINTER(Int32)
lib.int32_pos.argtypes = [POINTER(Int32)]
lib.int32_pos.restype = POINTER(Int32)


lib.int64_new.argtypes = [c_int64]
lib.int64_new.restype = POINTER(Int64)
lib.int64_delete.argtypes = [POINTER(Int64)]
lib.int64_delete.restype = None
lib.int64_value.argtypes = [POINTER(Int64)]
lib.int64_value.restype = c_int64
lib.int64_add.argtypes = [POINTER(Int64),POINTER(Int64)]
lib.int64_add.restype = POINTER(Int64)
lib.int64_sub.argtypes = [POINTER(Int64),POINTER(Int64)]
lib.int64_sub.restype = POINTER(Int64)
lib.int64_mul.argtypes = [POINTER(Int64),POINTER(Int64)]
lib.int64_mul.restype = POINTER(Int64)
lib.int64_tdiv.argtypes = [POINTER(Int64),POINTER(Int64)]
lib.int64_tdiv.restype = POINTER(Int64)
lib.int64_fdiv.argtypes = [POINTER(Int64),POINTER(Int64)]
lib.int64_fdiv.restype = POINTER(Int64)
lib.int64_mod.argtypes = [POINTER(Int64),POINTER(Int64)]
lib.int64_mod.restype = POINTER(Int64)
lib.int64_pow.argtypes = [POINTER(Int64),POINTER(Int64)]
lib.int64_pow.restype = POINTER(Int64)
lib.int64_abs.argtypes = [POINTER(Int64)]
lib.int64_abs.restype = POINTER(Int64)
lib.int64_neg.argtypes = [POINTER(Int64)]
lib.int64_neg.restype = POINTER(Int64)
lib.int64_pos.argtypes = [POINTER(Int64)]
lib.int64_pos.restype = POINTER(Int64)




class int16:
    def __init__(self,value): self.__value = lib.int16_new(value)

    def __del__(self): lib.int16_delete(self.__value)

    def __repr__(self): return f"{lib.int16_value(self.__value)}"

    def to_int32(self): return int32(lib.int16_value(self.__value))

    def to_int64(self): return int64(lib.int16_value(self.__value))

    @property
    def value(self): return self.__value

    @property
    def dtype(self): return type(self).__name__

    def __add__(self,other):
        if isinstance(other,int16):
            result = int16(0)
            result.__value = lib.int16_add(self.__value,other.__value)
            return result
        elif isinstance(other,int32): return self.to_int32() + other 
        elif isinstance(other,int64): return self.to_int64() + other
        raise TypeError(f"unsupported operand type for +: '{type(other).__name__}' with 'int16'")

    def __sub__(self,other):
        if isinstance(other,int16):
            result = int16(0)
            result.__value = lib.int16_sub(self.__value,other.__value)
            return result
        elif isinstance(other,int32): return self.to_int32() - other
        elif isinstance(other,int64): return self.to_int64() - other
        raise TypeError(f"unsupported operand type for -: '{type(other).__name__}' with 'int16'")

    def __mul__(self,other):
        if isinstance(other,int16):
            result = int16(0)
            result.__value = lib.int16_mul(self.__value,other.__value)
            return result
        elif isinstance(other,int32): return self.to_int32() * other
        elif isinstance(other,int64): return self.to_int64() * other
        raise TypeError(f"unsupported operand type for *: '{type(other).__name__}' with 'int16'")

    def __truediv__(self,other):
        if isinstance(other,int16):
            result = int16(0)
            result.__value = lib.int16_tdiv(self.__value,other.__value)
            return result
        elif isinstance(other,int32): return self.to_int32() / other
        elif isinstance(other,int64): return self.to_int64() / other
        raise TypeError(f"unsupported operand typr for /: '{type(other).__name__}' with 'int16'")


class int32:
    def __init__(self,value): self.__value = lib.int32_new(value)

    def __del__(self): lib.int32_delete(self.__value)

    def __repr__(self): return f"{lib.int32_value(self.__value)}"

    def to_int64(self): return int64(lib.int32_value(self.__value))

    @property
    def value(self): return self.__value

    @property
    def dtype(self): return type(self).__name__

    def set_value(self,value): self.__value = value

    def __add__(self,other):
        if isinstance(other,int32):
            result = int32(0)
            result.__value = lib.int32_add(self.__value,other.__value)
            return result
        elif isinstance(other,int64): return self.to_int64() + other
        elif isinstance(other,int16): return self + other.to_int32()
        raise TypeError(f"unsupported operand type for +: '{type(other).__name__}' with 'int32'")

    def __sub__(self,other):
        if isinstance(other,int32):
            result = int32(0)
            result.__value = lib.int32_sub(self.__value,other.__value)
            return result
        elif isinstance(other,int64): return self.to_int64() - other
        elif isinstance(other,int16): return self - other.to_int32()
        raise TypeError(f"unsupported operand type for +: '{type(other).__name__}' with 'int32'")

    def __mul__(self,other):
        if isinstance(other,int32):
            result = int32(0)
            result.__value = lib.int32_mul(self.__value,other.__value)
            return result
        elif isinstance(other,int64): return self.to_int64() * other
        elif isinstance(other,int16): return self * other.to_int32() 
        raise TypeError(f"unsupported operand type for +: '{type(other).__name__}' with 'int32'")

    def __truediv__(self,other):
        if isinstance(other,int32):
            result = int32(0)
            result.__value = lib.int32_tdiv(self.__value,other.__value)
            return result
        elif isinstance(other,int64): return self.to_int64() / other
        elif isinstance(other,int16): return self / other.to_int32()
        raise TypeError(f"unsupported operand type for +: '{type(other).__name__}' with 'int32'")


class int64:
    def __init__(self,value): self.__value = lib.int64_new(value)

    def __del__(self): lib.int64_delete(self.__value)

    def __repr__(self): return f"{lib.int64_value(self.__value)}"

    @property
    def value(self): return self.__value

    @property
    def dtype(self): return type(self).__name__

    def __add__(self,other):
        if isinstance(other,int64):
            result = int64(0)
            result.__value = lib.int64_add(self.__value,other.__value)
            return result
        elif isinstance(other,int32) or isinstance(other,int16): return self + other.to_int64()
    
    def __sub__(self,other):
        if isinstance(other,int64):
            result = int64(0)
            result.__value = lib.int64_sub(self.__value,other.__value)
            return result
        elif isinstance(other,int32) or isinstance(other,int16): return self - other.to_int64()

    def __mul__(self,other):
        if isinstance(other,int64):
            result = int64(0)
            result.__value = lib.int64_mul(self.__value,other.__value)
            return result
        elif isinstance(other,int32) or isinstance(other,int16): return self * other.to_int64()



