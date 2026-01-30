import numpy as np


def generate_ramping_file(output_file_path):
    """
    Ramping file rules
    
    1. The first row of all columns must be description of each column (string, the value of string has no specific meaning. It can be whatever you prefer)
    2. The first column must be turns, and the turns must start from 1 and be consecutive
    3. The delimeter of each column must be ','
    4. For sbend, rbend, quadrupole, sextupole, octupole, the second column contains the values of kl.
       For multipole, from the 2nd column, all columns must be arranged according to the order and direction. 
            E.g. for a multipole has k2l values, the columns must be arranged as k0l,k0sl,k1l,k1sl,k2l,k2sl, even though the other values are 0.
       For kicker, the 2nd and 3rd columns contain the values of hkick and vkick.
       For tune exciter,
    """

    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write("turn,k1l\n")

        for i in range(1, 100 + 1):
            f.write(f"{i},{0.2}\n")

    print(f"Ramping file has been generated successfully: {output_file_path}")


if __name__ == '__main__':
    generate_ramping_file(r"C:\Users\changmx\Documents\PASS\para\k1l.csv")
