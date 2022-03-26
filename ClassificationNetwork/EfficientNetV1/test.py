import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument( '--name', default = 'Li Hua' )
    parser.add_argument( '--year', type = int, default = 18 )
    args = parser.parse_args()
    print(args)
    print(args.name)

pg = [ p for p in range(3) 
        
            if p == 2
     ]

print(pg)
