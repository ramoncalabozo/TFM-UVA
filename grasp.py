import random
import os
from os import remove
import numpy as np

def sustituirfichero(temporal, caracteristicaSZA, L1, L2, L3, L4):
    fileSdataScript = open("sdata_Script.txt", "r")
    fileTemporal = open(temporal, "w")
    res = 180 - caracteristicaSZA 
    longitud1 = L1
    longitud2 = L2
    longitud3 = L3
    longitud4 = L4
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

    # Sustitución del atributo L1
    with open(temporal, "r+") as file:
        x = file.read()
        
    with open(temporal, "w+") as file:
        x = x.replace("L1",str(longitud1))
        file.write(x)

 # Sustitución del atributo L2
    with open(temporal, "r+") as file:
        x = file.read()
        
    with open(temporal, "w+") as file:
        x = x.replace("L2",str(longitud2))
        file.write(x)

    # Sustitución del atributo L3
    with open(temporal, "r+") as file:
        x = file.read()
        
    with open(temporal, "w+") as file:
        x = x.replace("L3",str(longitud3))
        file.write(x)

 # Sustitución del atributo L4
    with open(temporal, "r+") as file:
        x = file.read()
        
    with open(temporal, "w+") as file:
        x = x.replace("L4",str(longitud4))
        file.write(x)


def numeroAleatorioDistribucionLog(min, max):
    log_min_val = np.log(min)
    log_max_val = np.log(max)

    r = np.random.uniform()
    log_r = log_min_val + r * (log_max_val - log_min_val)
    random_number = np.exp(log_r)
    return random_number


def numerosAleatoriosArrayOrdenado(min, max):
    longitudes = []
    for i in range (4):
        longitudes.append(random.uniform(min, max))
    longitudes.sort()
    return longitudes
    

if __name__ == '__main__':
    
    # archivo con los parametros
    fileInput = open("input.txt", "w")
    fileRun = open("fileRun.txt", "w")

    for i in range(500):    
        temporal = "temporal" + str(i) + ".txt"

        # Longitudes de HondaL1, L2, L3, L4
        longitudes = numerosAleatoriosArrayOrdenado(0.3, 2.5)
        L1 = longitudes[0]
        L2 = longitudes[1]
        L3 = longitudes[2]
        L4 = longitudes[3]

        # SZA
        caracteristicaSZA = random.uniform(20,80)
        res = 180 - caracteristicaSZA
        sustituirfichero(temporal, caracteristicaSZA, L1, L2, L3, L4)
        comando = "grasp settings.yml input.file=" + temporal +" output.segment.stream=resultados/output" + str(i).rjust(4,'0') + ".txt"
        parametros =  "SZA = " + str(caracteristicaSZA)
        parametros =  parametros + " RES = " + str(res)
        parametros =  parametros + " L1 = " + str(L1)
        parametros =  parametros + " L2 = " + str(L2)
        parametros =  parametros + " L3 = " + str(L3)
        parametros =  parametros + " L4 = " + str(L4)

        # CARACTERISITCA_1
        caracteristica1_radio1 = random.uniform(0.05, 0.7)
        caracteristica1_std1 = random.uniform(0.05, 3.0)
        comando = comando +  " retrieval.constraints.characteristic[1].mode[1].initial_guess.value=[" + str(caracteristica1_radio1) + "," + str(caracteristica1_std1) + "]"
        parametros = parametros + " caracteristica1_radio1 = " + str(caracteristica1_radio1) + " caracteristica1_std1 = " + str(caracteristica1_std1)
        
        caracteristica1_radio2 = random.uniform(0.7, 10.0)
        caracteristica1_std2 = random.uniform(0.05, 3.0)
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
        caracteristica4_modo1_long1 = numeroAleatorioDistribucionLog(0.0005, 0.5)
        caracteristica4_modo1_long2 = numeroAleatorioDistribucionLog(0.0005, 0.5)
        caracteristica4_modo1_long3 = numeroAleatorioDistribucionLog(0.0005, 0.5)
        caracteristica4_modo1_long4 = numeroAleatorioDistribucionLog(0.0005, 0.5)
        comando = comando +  " retrieval.constraints.characteristic[4].mode[1].initial_guess.value=[" + str(caracteristica4_modo1_long1) + "," + str(caracteristica4_modo1_long2) + "," +  str(caracteristica4_modo1_long3) + "," + str(caracteristica4_modo1_long4) + "]"
        parametros = parametros + " caracteristica4_modo1_long1 = " + str(caracteristica4_modo1_long1) + " caracteristica4_modo1_long2 = " + str(caracteristica4_modo1_long2) + " caracteristica4_modo1_long3 = " + str(caracteristica4_modo1_long3) + " caracteristica4_modo1_long4 = " + str(caracteristica4_modo1_long4)

        caracteristica4_modo2_long1 = numeroAleatorioDistribucionLog(0.0005, 0.5)
        caracteristica4_modo2_long2 = numeroAleatorioDistribucionLog(0.0005, 0.5) 
        caracteristica4_modo2_long3 = numeroAleatorioDistribucionLog(0.0005, 0.5) 
        caracteristica4_modo2_long4 = numeroAleatorioDistribucionLog(0.0005, 0.5) 
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

        caracteristica8_modo2_long1 = random.uniform(0.00099, 0.8) 
        caracteristica8_modo2_long2 = random.uniform(0.00099, 0.8) 
        caracteristica8_modo2_long3 = random.uniform(0.00099, 0.8) 
        caracteristica8_modo2_long4 = random.uniform(0.00099, 0.8) 
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
