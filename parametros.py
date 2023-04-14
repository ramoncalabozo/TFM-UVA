import numpy as np
import os

def read_output(outfile):
    # archivo con los datos, para luego el entrenamiento y la evaluación
    with open(outfile) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    dataset = ""
    return dataset 

if __name__ == '__main__':
    fileDataset = open("dataset.txt", "w")
    for i in range(1):
        output = "resultados/output"
        output = output + str(i) + ".txt"
        lecturaDataSet = read_output(output)
        fileDataset.write(lecturaDataSet + os.linesep)
        print("TODO CORRECTO")
        


