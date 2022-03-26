from typing import Optional, Callable


def async_query(on_success: Optional[ Callable[[int], int] ] = None,
                on_error: Optional[Callable[[int, Exception], None]] = None ) -> None:

    print(on_success)

def test(name):
    print(123)
    return 2

def test1(name1,name2):
    print(123)


