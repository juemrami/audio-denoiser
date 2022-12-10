import argparse
print('Clean Machine Starting Up...')
from traditional import WienerFilter, SpectralSubtraction

# create the parser
parser = argparse.ArgumentParser()
# add the arguments

# parser.print_usage = parser.print_help = lambda *args: None
parser.add_argument('-f', type=str, help='file path', required=False)
parser.add_argument('-t', type=str, help=' Algorithm Type (SS, WF, or ML) for Spectral Subtraction, Wiener Filter, or Machine Learning respectively')
## add sub argument to -t for ML to specify the segment time
# parser.add_argument('-s', type=float, help='Segment Time for Machine Learning. In seconds. 0 for off', required=False)
args = parser.parse_args()

file_name = args.f
denoise_type : str = (args.t) 
denoise_name = ''
outfile = file_name.split('.')[0] + '_' +denoise_type.lower()+'_denoised.wav'
if  denoise_type == 'SS':
    denoise_name = 'Spectral Subtraction'
    print("Selected: ", denoise_name)
elif denoise_type == 'WF':
    denoise_name = 'Wiener Filter'
    print("Selected: ", denoise_name)
elif denoise_type == 'ML':
    denoise_name = 'Machine Learning'
    print("Selected: ", denoise_name)
    from infer import DeepDenoise
else:
    print("Please provide a valid denoising algorithm")
    exit()

if not file_name:
    print("Please provide a file name")
    exit()
print("Input File: ", file_name)
print("Selected: ", denoise_name)

if denoise_type == 'WF':
    try:
        WienerFilter(file_name)
    except Exception as e:
        print("Error: %d failed. Reason:", denoise_name)
        print(e)
        exit()
    print ("====Processing Successful====")
    print("Output File: ", outfile)
elif denoise_type == 'SS':
    try:
        SpectralSubtraction(file_name)
    except Exception as e:
        print("Error: %d failed. Reason:", denoise_name)
        print(e)
        exit()
    print ("====Processing Successful====")
    print("Output File: ", outfile)
    pass
elif denoise_type == 'ML':
    try:
        # print(int(args.s is not None))
        # time = (0, float(args.s))[int(args.s is not None)]
        DeepDenoise(file_name)
    except Exception as e:
        print("Error: %d failed. Reason:", denoise_name)
        print(e)
        exit()
    print ("====Processing Successful====")
    print("Output File: ", outfile)
    
