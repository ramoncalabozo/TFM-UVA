import random
import os
from os import remove
import numpy as np

def sustituirfichero(temporal, caracteristicaSZA):
    fileSdataScript = open("sdata_Script.txt", "r")
    fileTemporal = open(temporal, "w")
    res = 180 - caracteristicaSZA 
    
    # Sustitución del atributo SZA
    for line in fileSdataScript:
        fileTemporal.write(line.replace('SZA', str(caracteristicaSZA)))
    fileSdataScript.close() 
    fileTemporal.close()
    
    # Sustitución del atributo RES
    with open(temporal, "r+") as file:
        x = file.read()
        
    with open(temporal, "w+") as file:
        x = x.replace("RES",str(res))
        file.write(x)


def numeroAleatorioDistribucionLog(min, max):
    log_min_val = np.log(min)
    log_max_val = np.log(max)

    r = np.random.uniform()
    log_r = log_min_val + r * (log_max_val - log_min_val)
    random_number = np.exp(log_r)
    return random_number



if __name__ == '__main__':
    
    # archivo con los parametros
    fileInput = open("input.txt", "w")
    fileRun = open("fileRun.txt", "w")

    for i in range(10):    
        temporal = "temporal" + str(i) + ".txt"

        # SZA
        caracteristicaSZA = random.randint(20,80)
        res = 180 - caracteristicaSZA
        sustituirfichero(temporal, caracteristicaSZA)
        comando = "grasp settings.yml input.file=" + temporal +" output.segment.stream=resultados/output" + str(i) + ".txt"
        parametros =  "SZA = " + str(caracteristicaSZA)
        parametros =  parametros + " RES = " + str(res)

        # CARACTERISITCA_1
        caracteristica1_radio1 = random.uniform(0.1, 0.7)
        caracteristica1_std1 = random.uniform(0.1, 0.9)
        comando = comando +  " retrieval.constraints.characteristic[1].mode[1].initial_guess.value=[" + str(caracteristica1_radio1) + "," + str(caracteristica1_std1) + "]"
        parametros = parametros + " caracteristica1_radio1 = " + str(caracteristica1_radio1) + " caracteristica1_std1 = " + str(caracteristica1_std1)
        
        caracteristica1_radio2 = random.uniform(0.7, 5.0)
        caracteristica1_std2 = random.uniform(0.1, 0.9)
        comando = comando +  " retrieval.constraints.characteristic[1].mode[2].initial_guess.value=[" + str(caracteristica1_radio2) + "," + str(caracteristica1_std2) + "]"
        parametros = parametros + " caracteristica1_radio2 = " + str(caracteristica1_radio2) + " caracteristica1_std2 = " + str(caracteristica1_std2)
        
        # CARACTERISTICA_2
        caracteristica2_modo1 = numeroAleatorioDistribucionLog(0.0005, 5.0)
        comando = comando +  " retrieval.constraints.characteristic[2].mode[1].initial_guess.value=" + str(caracteristica2_modo1)
        parametros = parametros + " caracteristica2_modo1 = " + str(caracteristica2_modo1)

        caracteristica2_modo2 = numeroAleatorioDistribucionLog(0.0005, 5.0)
        comando = comando +  " retrieval.constraints.characteristic[2].mode[2].initial_guess.value=" + str(caracteristica2_modo2)
        parametros = parametros + " caracteristica2_modo2 = " + str(caracteristica2_modo2)

        # CARACTERISTICA_3
        caracteristica3_modo1_long1 = random.uniform(1.33, 1.6) 
        caracteristica3_modo1_long2 = random.uniform(1.33, 1.6) 
        caracteristica3_modo1_long3 = random.uniform(1.33, 1.6) 
        caracteristica3_modo1_long4 = random.uniform(1.33, 1.6)
        comando = comando +  " retrieval.constraints.characteristic[3].mode[1].initial_guess.value=[" + str(caracteristica3_modo1_long1) + "," + str(caracteristica3_modo1_long2) + "," +  str(caracteristica3_modo1_long3) + "," + str(caracteristica3_modo1_long4) + "]"
        parametros = parametros + " caracteristica3_modo1_long1 = " + str(caracteristica3_modo1_long1) + " caracteristica3_modo1_long2 = " + str(caracteristica3_modo1_long2) + " caracteristica3_modo1_long3 = " + str(caracteristica3_modo1_long3) + " caracteristica3_modo1_long4 = " + str(caracteristica3_modo1_long4)
        
        caracteristica3_modo2_long1 = random.uniform(1.33, 1.6) 
        caracteristica3_modo2_long2 = random.uniform(1.33, 1.6) 
        caracteristica3_modo2_long3 = random.uniform(1.33, 1.6) 
        caracteristica3_modo2_long4 = random.uniform(1.33, 1.6)
        comando = comando +  " retrieval.constraints.characteristic[3].mode[2].initial_guess.value=[" + str(caracteristica3_modo2_long1) + "," + str(caracteristica3_modo2_long2) + "," +  str(caracteristica3_modo2_long3) + "," + str(caracteristica3_modo2_long4) + "]" 
        parametros = parametros + " caracteristica3_modo2_long1 = " + str(caracteristica3_modo2_long1) + " caracteristica3_modo2_long2 = " + str(caracteristica3_modo2_long2) + " caracteristica3_modo2_long3 = " + str(caracteristica3_modo2_long3) + " caracteristica3_modo2_long4 = " + str(caracteristica3_modo2_long4)
        
        # CARACTERISTICA_4
        caracteristica4_modo1_long1 = random.uniform(0.0005, 0.5)
        caracteristica4_modo1_long2 = random.uniform(0.0005, 0.5)
        caracteristica4_modo1_long3 = random.uniform(0.0005, 0.5)
        caracteristica4_modo1_long4 = random.uniform(0.0005, 0.5)
        comando = comando +  " retrieval.constraints.characteristic[4].mode[1].initial_guess.value=[" + str(caracteristica4_modo1_long1) + "," + str(caracteristica4_modo1_long2) + "," +  str(caracteristica4_modo1_long3) + "," + str(caracteristica4_modo1_long4) + "]"
        parametros = parametros + " caracteristica4_modo1_long1 = " + str(caracteristica4_modo1_long1) + " caracteristica4_modo1_long2 = " + str(caracteristica4_modo1_long2) + " caracteristica4_modo1_long3 = " + str(caracteristica4_modo1_long3) + " caracteristica4_modo1_long4 = " + str(caracteristica4_modo1_long4)

        caracteristica4_modo2_long1 = random.uniform(0.0005, 0.5)
        caracteristica4_modo2_long2 = random.uniform(0.0005, 0.5) 
        caracteristica4_modo2_long3 = random.uniform(0.0005, 0.5) 
        caracteristica4_modo2_long4 = random.uniform(0.0005, 0.5) 
        comando = comando +  " retrieval.constraints.characteristic[4].mode[2].initial_guess.value=[" + str(caracteristica4_modo2_long1) + "," + str(caracteristica4_modo2_long2) + "," +  str(caracteristica4_modo2_long3) + "," + str(caracteristica4_modo2_long4) + "]"
        parametros = parametros + " caracteristica4_modo2_long1 = " + str(caracteristica4_modo2_long1) + " caracteristica4_modo2_long2 = " + str(caracteristica4_modo2_long2) + " caracteristica4_modo2_long3 = " + str(caracteristica4_modo2_long3) + " caracteristica4_modo2_long4 = " + str(caracteristica4_modo2_long4)

        # CARACTERISTICA_5
        caracteristica5_modo1 = random.uniform(0.001, 1.0) 
        comando = comando + " retrieval.constraints.characteristic[5].mode[1].initial_guess.value="+ str(caracteristica5_modo1)
        parametros = parametros +  " caracteristica5_modo1 = " + str(caracteristica5_modo1)
        
        caracteristica5_modo2 = random.uniform(0.001, 1.0)
        comando = comando + " retrieval.constraints.characteristic[5].mode[2].initial_guess.value="+ str(caracteristica5_modo2) 
        parametros = parametros +  " caracteristica5_modo2 = " + str(caracteristica5_modo2)
        
        # CARACTERISTICA_6
        caracteristica6 = random.uniform(10, 2000)
        comando = comando + " retrieval.constraints.characteristic[6].mode[1].initial_guess.value="+ str(caracteristica6)
        parametros = parametros + " caracteristica6 = " + str(caracteristica6)
    
        # CARACTERISTICA_7
        caracteristica7 = random.uniform(150, 50000)
        comando = comando + " retrieval.constraints.characteristic[7].mode[1].initial_guess.value="+ str(caracteristica7)
        parametros = parametros + " caracteristica7 = " + str(caracteristica7)
        
        # CARACTERISTICA_8
        caracteristica8_modo1_long1 = random.uniform(0.00099, 1.0) 
        caracteristica8_modo1_long2 = random.uniform(0.00099, 1.0) 
        caracteristica8_modo1_long3 = random.uniform(0.00099, 1.0) 
        caracteristica8_modo1_long4 = random.uniform(0.00099, 1.0) 
        comando = comando +  " retrieval.constraints.characteristic[8].mode[1].initial_guess.value=[" + str(caracteristica8_modo1_long1) + "," + str(caracteristica8_modo1_long2) + "," +  str(caracteristica8_modo1_long3) + "," + str(caracteristica8_modo1_long4) + "]"
        parametros = parametros + " caracteristica8_modo1_long1 = " + str(caracteristica8_modo1_long1) + " caracteristica8_modo1_long2 = " + str(caracteristica8_modo1_long2) + " caracteristica8_modo1_long3 = " + str(caracteristica8_modo1_long3) + " caracteristica8_modo1_long4 = " + str(caracteristica8_modo1_long4)

        caracteristica8_modo2_long1 = random.uniform(0.00099, 1.0) 
        caracteristica8_modo2_long2 = random.uniform(0.00099, 1.0) 
        caracteristica8_modo2_long3 = random.uniform(0.00099, 1.0) 
        caracteristica8_modo2_long4 = random.uniform(0.00099, 1.0) 
        comando = comando +  " retrieval.constraints.characteristic[8].mode[2].initial_guess.value=[" + str(caracteristica8_modo2_long1) + "," + str(caracteristica8_modo2_long2) + "," +  str(caracteristica8_modo2_long3) + "," + str(caracteristica8_modo2_long4) + "]"
        parametros = parametros + " caracteristica8_modo2_long1 = " + str(caracteristica8_modo2_long1) + " caracteristica8_modo2_long2 = " + str(caracteristica8_modo2_long2) + " caracteristica8_modo2_long3 = " + str(caracteristica8_modo2_long3) + " caracteristica8_modo2_long4 = " + str(caracteristica8_modo2_long4)

        caracteristica8_modo3_long1 = random.uniform(0.0005, 1.5) 
        caracteristica8_modo3_long2 = random.uniform(0.0005, 1.5)
        caracteristica8_modo3_long3 = random.uniform(0.0005, 1.5)
        caracteristica8_modo3_long4 = random.uniform(0.0005, 1.5)
        comando = comando +  " retrieval.constraints.characteristic[8].mode[3].initial_guess.value=[" + str(caracteristica8_modo3_long1) + "," + str(caracteristica8_modo3_long2) + "," +  str(caracteristica8_modo3_long3) + "," + str(caracteristica8_modo3_long4) + "]"
        parametros = parametros + " caracteristica8_modo3_long1 = " + str(caracteristica8_modo3_long1) + " caracteristica8_modo3_long2 = " + str(caracteristica8_modo3_long2) + " caracteristica8_modo3_long3 = " + str(caracteristica8_modo3_long3) + " caracteristica8_modo3_long4 = " + str(caracteristica8_modo3_long4)

        fileRun.write(comando + os.linesep)
        fileInput.write(parametros + os.linesep)
        os.system(comando)
        remove(temporal)

    fileRun.close()
    fileInput.close()
