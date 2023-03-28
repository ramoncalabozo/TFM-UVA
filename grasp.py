import random
import os;

if __name__ == '__main__':

    fileInput = open("input.txt", "w")
    fileRun = open("fileRun.txt", "w")

    for i in range(10):
        parametros= ""
        comando = "grasp settingsForwardBilog_2modos4long.yml input.file=sdata_RadAOD_4long.txt output.segment.stream=resultados/pruebaScript" + str(i) + ".txt"
    
        # CARACTERISITCA_1
        caracteristica1_radio1 = random.uniform(0.1, 0.7)
        caracteristica1_std1 = random.uniform(0.1, 0.9)
        comando = comando +  " retrieval.constraints.characteristic[1].mode[1].initial_guess.value=[" + str(caracteristica1_radio1) + "," + str(caracteristica1_std1) + "]"
        parametros = parametros + "caracteristica1_radio1 = " + str(caracteristica1_radio1) + " caracteristica1_std1 = " + str(caracteristica1_std1)
        
        caracteristica1_radio2 = random.uniform(0.7, 5.0)
        caracteristica1_std2 = random.uniform(0.1, 0.9)
        comando = comando +  " retrieval.constraints.characteristic[1].mode[2].initial_guess.value=[" + str(caracteristica1_radio2) + "," + str(caracteristica1_std2) + "]"
        parametros = parametros + " caracteristica1_radio2 = " + str(caracteristica1_radio2) + " caracteristica1_std2 = " + str(caracteristica1_std2)
    
        # CARACTERISTICA_2
        # PROBABILIDAD 
        
        # CARACTERISTICA_3
        # TIENE QUE SER EL MISMO NÚMERO O DISTINTO?
        caracteristica3_modo1_long1 = random.uniform(1.33, 1.6) 
        caracteristica3_modo1_long2 = random.uniform(1.33, 1.6) 
        caracteristica3_modo1_long3 = random.uniform(1.33, 1.6) 
        caracteristica3_modo1_long4 = random.uniform(1.33, 1.6) 
        # TIENE QUE SER EL MISMO NÚMERO O DISTINTO?
        caracteristica3_modo2_long1 = random.uniform(1.33, 1.6) 
        caracteristica3_modo2_long2 = random.uniform(1.33, 1.6) 
        caracteristica3_modo2_long3 = random.uniform(1.33, 1.6) 
        caracteristica3_modo2_long4 = random.uniform(1.33, 1.6) 
        # CARACTERISTICA_4
        # TIENE QUE SER EL MISMO NÚMERO O DISTINTO?
        caracteristica4_modo1_long1 = random.uniform(0.0005, 0.5)
        caracteristica4_modo1_long2 = random.uniform(0.0005, 0.5)
        caracteristica4_modo1_long3 = random.uniform(0.0005, 0.5)
        caracteristica4_modo1_long4 = random.uniform(0.0005, 0.5)
        # TIENE QUE SER EL MISMO NÚMERO O DISTINTO?
        caracteristica4_modo2_long1 = random.uniform(0.0005, 0.5)
        caracteristica4_modo2_long2 = random.uniform(0.0005, 0.5) 
        caracteristica4_modo2_long3 = random.uniform(0.0005, 0.5) 
        caracteristica4_modo2_long4 = random.uniform(0.0005, 0.5) 
        
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
        caracteristica6 = random.uniform(150, 50000)
        # CARACTERISTICA_8
        caracteristica8_modo1_long1 = random.uniform(0.00099, 1.0) 
        caracteristica8_modo1_long2 = random.uniform(0.00099, 1.0) 
        caracteristica8_modo1_long3 = random.uniform(0.00099, 1.0) 
        caracteristica8_modo1_long4 = random.uniform(0.00099, 1.0) 
        
        caracteristica8_modo2_long1 = random.uniform(0.00099, 1.0) 
        caracteristica8_modo2_long2 = random.uniform(0.00099, 1.0) 
        caracteristica8_modo2_long3 = random.uniform(0.00099, 1.0) 
        caracteristica8_modo2_long4 = random.uniform(0.00099, 1.0) 
        
        caracteristica8_modo3_long1 = random.uniform(0.00099, 1.0) 
        caracteristica8_modo3_long2 = random.uniform(0.00099, 1.0)
        caracteristica8_modo3_long3 = random.uniform(0.00099, 1.0)
        caracteristica8_modo3_long4 = random.uniform(0.00099, 1.0)

        print(comando)

        fileRun.write(comando + os.linesep)
        fileInput.write(parametros + os.linesep)

    fileRun.close()
    fileInput.close()
