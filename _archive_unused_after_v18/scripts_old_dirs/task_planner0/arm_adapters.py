# V11 modular facade. Implementation is kept in core.py to preserve the validated V10 behavior.
from .core import LeftArmAdapter, RightArmAdapter, get_left_attr, get_left_offset

__all__ = ['LeftArmAdapter', 'RightArmAdapter', 'get_left_attr', 'get_left_offset']
