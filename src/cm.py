import argparse
import os
import numpy as np
from traditional import WienerFilter, SpectralSubtraction

# create the parser
parser = argparse.ArgumentParser()
# add the arguments

# parser.print_usage = parser.print_help = lambda *args: None
parser.add_argument('-f', type=str, help='file path', required=False)
parser.add_argument('-t', type=str, help=' Algorithm Type (SS, WF, or ML) for Spectral Subtraction, Wiener Filter, or Machine Learning respectively')
args = parser.parse_args()

file_name = args.f
denoise_type = (args.t)
denoise_name = ''
outfile = file_name.split('.')[0] + '_denoised.wav'
if  denoise_type == 'SS':
    denoise_name = 'Spectral Subtraction'
elif denoise_type == 'WF':
    denoise_name = 'Wiener Filter'
elif denoise_type == 'ML':
    denoise_name = 'Machine Learning'
else:
    print("Please provide a valid denoising algorithm")
    exit()

if not file_name:
    print("Please provide a file name")
    exit()
print('Clean Machine Stating Up...')
print("Input File: ", file_name)
print("Selected: ", denoise_name)

if denoise_type == 'WF':
    try:
        WienerFilter(file_name)
    except Exception as e:
        print(e)
        print("Error: Wiener Filtering failed.")
        exit()
    print ("====Processing Successful====")
    print("Output File: ", outfile)
elif denoise_type == 'SS':
    try:
        SpectralSubtraction(file_name)
    except Exception as e:
        print(e)
        print("Error: Spectral Subtraction failed.")
        exit()
    print ("====Processing Successful====")
    print("Output File: ", outfile)
    pass
elif denoise_type == 'ML':
    denoise_name = 'Machine Learning'


