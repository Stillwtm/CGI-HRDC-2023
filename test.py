import sys, getopt
import os
import cv2
import glob
import csv
import pandas as pd
from submit.model import model

def main():
    input_path = None
    output_path = None
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print ('test.py -i <inputfile> -o <outputfile>')
    for opt, arg in opts:
        if opt in ['-i']:
            input_path = arg
        elif opt in ['-o']:
            output_path = arg
    if (os.path.exists(input_path) == False):
        print('Input path does not exist')
        return
    if (os.path.splitext(output_path)[1] != '.csv'):
        print('Output file must be a csv file')
        return
    if (os.path.exists(os.path.dirname(output_path)) == False):
        print('Output path does not exist')
        return
    if (os.path.exists(output_path)):
        os.remove(output_path)
    
    DIP = model()
    DIP.load('./submit')
    images_path = glob.glob(os.path.join(input_path, '*.png'))
    header = True
    for image_path in images_path:
        image = cv2.imread(image_path)
        result = DIP.predict(image)
        print(result)
        filename = os.path.basename(image_path)
        data = {'Image':filename,'Hypertensive':result}
        nd = pd.DataFrame(data,index=[0])
        nd.to_csv(output_path, mode='a', header=header, index=False)
        header = False
    print('Done')
    
if __name__ == "__main__":
    main()
    
    