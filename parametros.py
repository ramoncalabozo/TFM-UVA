import numpy as np
import os

def read_output(outfile):
    
    # variables de entrada
    radioComponent1= ""
    sigmaComponent1 = ""
    radioComponent2= ""
    sigmaComponent2 = ""

    #variables de salida
    aod_mode_1 = ""
    aod_mode_2 = ""
    ssa_mode_1 = ""
    ssa_mode_2 = ""
    g = ""


    dataset = ""
    
    # archivo con los datos, para luego el entrenamiento y la evaluación
    with open(outfile) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    
    i=-1 #i va a recorrer el vector content
    while True:
        i+=1
        if i >= (len(content)): #se ha recorrido todo el fichero
            break
        line=content[i]       
        
        carac = "Parameters of lognormal SD for Particle component 1"
        if carac in line:
            for j in range(2):
                i+=1
                line=content[i]
                separacion = line.split(":")
                if j == 0 :
                    radioComponent1 = separacion[1].strip()
                else :
                    sigmaComponent1 = separacion[1].strip()

        carac = "Parameters of lognormal SD for Particle component 2"
        if carac in line:
            for j in range(2):
                i+=1
                line=content[i]
                separacion = line.split(":")
                if j == 0 :
                    radioComponent2 = separacion[1].strip()
                else :
                    sigmaComponent2 = separacion[1].strip()

    dataset = str(radioComponent1) + " " + str(sigmaComponent1) + "    ---------------   " + str(radioComponent2) + " " + str(sigmaComponent2)
    return dataset 

if __name__ == '__main__':
    fileDataset = open("dataset.txt", "w")
    for i in range(1):
        output = "resultados/output"
        output = output + str(i) + ".txt"
        lecturaDataSet = read_output(output)
        print(lecturaDataSet)
        fileDataset.write(lecturaDataSet + os.linesep)

    fileDataset.close()
        


