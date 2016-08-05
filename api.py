from __future__ import print_function
from hTrust import hTrust
from sTrust import sTrust
from MATRI import model as Matri
import sys

usage = """
USAGE: python api.py <ALGORITHM_NAME>
Type of algorithm type can be:
    - sTrust
    - hTrust
"""

def show_usage():
    print(usage)
    sys.exit(1)



if __name__ == "__main__":
    if len(sys.argv) != 2:
        show_usage()
    
    algo = sys.argv[1].lower()
    if algo == "htrust":
        hTrust.init_main()
    elif algo == "strust":
        sTrust.init_main()
    elif algo == "matri":
        Matri.init_main()
    else:
        print("ERROR: invalid algorithm specified")
	show_usage()
        
    
    
    
    
