from typing import Union, Optional, List, Tuple

CompressType = Optional[Union[str,bool]]
ParallelType = Union[int,bool]
CacheType = Union[bool,str]
SecretsType = Optional[Union[str,dict]]
MipType = Union[int, List[int]]
ShapeType = Tuple[int,int,int]