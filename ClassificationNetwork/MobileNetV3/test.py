from typing import Optional, Callable, List

class test:

    def __init__(self, one, two):

        self.one = one
        self.two = two
    
    @staticmethod
    def adjust( m ,n):
        print( m + n)

array = [
            [ 1, 2, 3 ],
            [ 4, 5, 6 ]
        ]

class val:

    def __init__(self, cnf : List[test]):

        pval = cnf[0].one
        print(pval)

arr = [ test(1,2), test(1,2) ]

#val( arr )

for i in arr:
    
    print(i.one)

