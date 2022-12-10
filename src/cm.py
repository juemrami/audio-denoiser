import argparse

# create the parser
parser = argparse.ArgumentParser()

# add the arguments
parser.add_argument('-f', type=str, help='file path')
parser.add_argument('-t', type=str, help=' Algorithm Type (SS, WF, or ML) for Spectral Subtraction, Wiener Filter, or Machine Learning respectively')
args = parser.parse_args()

# parse the arguments

# print the values
print(args.f)
print(args.t)