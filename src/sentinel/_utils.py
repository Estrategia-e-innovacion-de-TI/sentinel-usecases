"""Internal utility functions for Sentinel."""

from typing import Dict, Optional, Type


def _get_all_subclasses_from_superclass(
    superclass: Type,
) -> Dict[str, Optional[str]]:
    """Recursively collect all non-private subclasses of a superclass.

    Parameters
    ----------
    superclass : type
        The base class to inspect.

    Returns
    -------
    dict
        Mapping of subclass name to its docstring.
    """
    result: Dict[str, Optional[str]] = {}
    for sb in superclass.__subclasses__():
        if sb.__name__[0] != "_":
            result[sb.__name__] = sb.__doc__
        else:
            result.update(_get_all_subclasses_from_superclass(sb))
    return result
