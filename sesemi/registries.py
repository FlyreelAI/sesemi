#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
from typing import Dict, Generic, TypeVar, Callable, Union

ResultType = TypeVar("ResultType")


class CallableRegistry(Generic[ResultType]):
    def __init__(self):
        self._lut: Dict[str, Callable[..., ResultType]] = {}

    def __call__(self, item: Union[str, Callable[..., ResultType]]):
        """Registers the callable.

        Args:
            item: Either the callable, in which case the name of the object
                is used to register it, or a string name, in which case
                a registration function is returned.

        Returns:
            Either the function that was registered or a registration function.
        """
        if isinstance(item, str):

            def register(x: Callable[..., ResultType]):
                self._lut[item] = x
                return x

            return register
        else:
            self._lut[item.__name__] = item
            return item

    def __getitem__(self, name: str) -> Callable[..., ResultType]:
        """Looks up the registered function with the given name"""
        return self._lut[name]
