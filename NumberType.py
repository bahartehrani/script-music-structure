from enum import Enum
import numpy as np

class NumberType(Enum):
    UINT8 = {
        "name": "UINT8",
        "max": 2**8 - 1,
        "min": 0,
        "scale": 2**8 - 1,
        "array": np.uint8
    }
    UINT16 = {
        "name": "UINT16",
        "max": 2**16 - 1,
        "min": 0,
        "scale": 2**16 - 1,
        "array": np.uint16
    }
    UINT32 = {
        "name": "UINT32",
        "max": 2**32 - 1,
        "min": 0,
        "scale": 2**32 - 1,
        "array": np.uint32
    }
    INT8 = {
        "name": "INT8",
        "max": 2**7 - 1,
        "min": -2**7,
        "scale": 2**7 - 1,
        "array": np.int8
    }
    INT16 = {
        "name": "INT16",
        "max": 2**15 - 1,
        "min": -2**15,
        "scale": 2**15 - 1,
        "array": np.int16
    }
    INT32 = {
        "name": "INT32",
        "max": 2**31 - 1,
        "min": -2**31,
        "scale": 2**31 - 1,
        "array": np.int32
    }
    FLOAT32 = {
        "name": "FLOAT32",
        "max": float('inf'),
        "min": float('-inf'),
        "scale": 1,
        "array": np.float32
    }
    FLOAT64 = {
        "name": "FLOAT64",
        "max": float('inf'),
        "min": float('-inf'),
        "scale": 1,
        "array": np.float64
    }

def get_number_type_by_name(object_or_string):
    if isinstance(object_or_string, str):
        for type_ in NumberType:
            if type_.value["name"] == object_or_string:
                return type_
    else:
        return object_or_string
