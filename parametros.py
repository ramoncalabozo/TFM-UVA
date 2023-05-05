import numpy as np
import os

def read_output(outfile):
        
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

        carac = "% of spherical particles for Particle component 1"
        if carac in line:
            i+=1
            line=content[i]
            separacion = line.split("   ")
            esfericidadComponent1 = separacion[1].strip()
        
        carac = "% of spherical particles for Particle component 2"
        if carac in line:
            i+=1
            line=content[i]
            separacion = line.split("   ")
            esfericidadComponent2 = separacion[1].strip()
        
        carac = "Aerosol volume concentration (um^3/um^2 or um^3/um^3)"
        if carac in line:
            for j in range(2):
                i+=1
                line=content[i]
                separacion = line.split("   ")
                if j == 0 :
                    concentracionComponent1 = separacion[1].strip()
                else :
                    concentracionComponent2 = separacion[1].strip()
        
        carac = "Wavelength (um), AOD_Particle_mode_1 (unitless or 1/um)"
        if carac in line:
            for j in range(4):
                i+=1
                line=content[i].strip()
                separacion = line.split("   ")
                if j == 0 :
                    L1 = separacion[0].strip()
                    aod_mode_1_L1  = separacion[1].strip()
                elif j == 1 :
                    L2 = separacion[0].strip()
                    aod_mode_1_L2 = separacion[1].strip()
                elif j == 2 :
                    L3 = separacion[0].strip()
                    aod_mode_1_L3 = separacion[1].strip()
                else:
                    L4 = separacion[0].strip()
                    aod_mode_1_L4 = separacion[1].strip()

        carac = "Wavelength (um), AOD_Particle_mode_2 (unitless or 1/um)"
        if carac in line:
            for j in range(4):
                i+=1
                line=content[i].strip()
                separacion = line.split("   ")
                if j == 0 :
                    aod_mode_2_L1  = separacion[1].strip()
                elif j == 1 :
                    aod_mode_2_L2 = separacion[1].strip()
                elif j == 2 :
                    aod_mode_2_L3 = separacion[1].strip()
                else:
                    aod_mode_2_L4 = separacion[1].strip()
      
        carac = "Wavelength (um), SSA_Particle_mode_1"
        if carac in line:
            for j in range(4):
                i+=1
                line=content[i].strip()
                separacion = line.split("   ")
                if j == 0 :
                    ssa_mode_1_L1  = separacion[1].strip()
                elif j == 1 :
                    ssa_mode_1_L2 = separacion[1].strip()
                elif j == 2 :
                    ssa_mode_1_L3 = separacion[1].strip()
                else:
                    ssa_mode_1_L4 = separacion[1].strip()

        carac = "Wavelength (um), SSA_Particle_mode_2"
        if carac in line:
            for j in range(4):
                i+=1
                line=content[i].strip()
                separacion = line.split("   ")
                if j == 0 :
                    ssa_mode_2_L1  = separacion[1].strip()
                elif j == 1 :
                    ssa_mode_2_L2 = separacion[1].strip()
                elif j == 2 :
                    ssa_mode_2_L3 = separacion[1].strip()
                else:
                    ssa_mode_2_L4 = separacion[1].strip()

        carac = "Wavelength (um), REAL Ref. Index for Particle component 1"
        if carac in line:
            for j in range(4):
                i+=1
                line=content[i].strip()
                separacion = line.split("       ")
                if j == 0 :
                    indRefReal_mode1_L1  = separacion[1].strip()
                elif j == 1 :
                    indRefReal_mode1_L2 = separacion[1].strip()
                elif j == 2 :
                    indRefReal_mode1_L3 = separacion[1].strip()
                else:
                    indRefReal_mode1_L4 = separacion[1].strip()

        carac = "Wavelength (um), REAL Ref. Index for Particle component 2"
        if carac in line:
            for j in range(4):
                i+=1
                line=content[i].strip()
                separacion = line.split("       ")
                if j == 0 :
                    indRefReal_mode2_L1  = separacion[1].strip()
                elif j == 1 :
                    indRefReal_mode2_L2 = separacion[1].strip()
                elif j == 2 :
                    indRefReal_mode2_L3 = separacion[1].strip()
                else:
                    indRefReal_mode2_L4 = separacion[1].strip()
        
        carac = "Wavelength (um), IMAG Ref. Index for Particle component 1"
        if carac in line:
            for j in range(4):
                i+=1
                line=content[i].strip()
                separacion = line.split("   ")
                if j == 0 :
                    indRefImag_mode1_L1  = separacion[1].strip()
                elif j == 1 :
                    indRefImag_mode1_L2 = separacion[1].strip()
                elif j == 2 :
                    indRefImag_mode1_L3 = separacion[1].strip()
                else:
                    indRefimag_mode1_L4 = separacion[1].strip()

        carac = "Wavelength (um), IMAG Ref. Index for Particle component 2"
        if carac in line:
            for j in range(4):
                i+=1
                line=content[i].strip()
                separacion = line.split("   ")
                if j == 0 :
                    indRefImag_mode2_L1  = separacion[1].strip()
                elif j == 1 :
                    indRefImag_mode2_L2 = separacion[1].strip()
                elif j == 2 :
                    indRefImag_mode2_L3 = separacion[1].strip()
                else:
                    indRefImag_mode2_L4 = separacion[1].strip()

        
        carac="Phase Matrix"
        if carac in line:
            fase=[]
            scat=[]
            Asym=[]
            i=i+4
            conta=0
            j = 0
            while True:
                i+=1
                line=content[i]
                if np.logical_not(line[0].isdigit()):   
                    if j != 0:
                        asimetria=np.trapz(np.cos(scat)*np.sin(scat)*fase,scat)/2
                        Asym.append(asimetria)
                    conta=conta+1
                    fase=[]
                    scat=[]
                    
                    if line[0]=='-':
                        asimetria=np.trapz(np.cos(scat)*np.sin(scat)*fase,scat)/2
                        Asym.append(asimetria)
                        break
                    i+=1
                    j+=1
                    continue

                line=line.split()
                scat.append(float(line[1])*np.pi/180)
                fase.append(float(line[2]))


    # Modo 1 -- L1
    dataset =  str(radioComponent1) + " " + str(sigmaComponent1) + " " + str(esfericidadComponent1) + " " + str(concentracionComponent1) + " " + str(L1) + " " + str(indRefReal_mode1_L1) + " " + str(indRefImag_mode1_L1) + " --- " + str(aod_mode_1_L1) + " " + str(ssa_mode_1_L1) + " " + str(Asym[0]) + "\n"
    # Modo 1 -- L2
    dataset += str(radioComponent1) + " " + str(sigmaComponent1) + " " + str(esfericidadComponent1) + " " + str(concentracionComponent1) + " " + str(L2) + " " + str(indRefReal_mode1_L2) + " " + str(indRefImag_mode1_L2) + " --- " + str(aod_mode_1_L2) + " " + str(ssa_mode_1_L2) + " " + str(Asym[2]) + "\n"
    # Modo 1 -- L3
    dataset += str(radioComponent1) + " " + str(sigmaComponent1) + " " + str(esfericidadComponent1) + " " + str(concentracionComponent1) + " " + str(L3) + " " + str(indRefReal_mode1_L3) + " " + str(indRefImag_mode1_L3) + " --- " + str(aod_mode_1_L3) + " " + str(ssa_mode_1_L3) + " " + str(Asym[4]) + "\n"
    # Modo 1 -- L4
    dataset += str(radioComponent1) + " " + str(sigmaComponent1) + " " + str(esfericidadComponent1) + " " + str(concentracionComponent1) + " " + str(L4) + " " + str(indRefReal_mode1_L4) + " " + str(indRefimag_mode1_L4) + " --- " + str(aod_mode_1_L4) + " " + str(ssa_mode_1_L4) + " " + str(Asym[6]) + "\n"
    
    # Modo 2 -- L1
    dataset += str(radioComponent2) + " " + str(sigmaComponent2) + " " + str(esfericidadComponent2) + " " + str(concentracionComponent2) + " " + str(L1) + " " + str(indRefReal_mode2_L1) + " " + str(indRefImag_mode2_L1) + " --- " + str(aod_mode_2_L1) + " " + str(ssa_mode_2_L1) + " " + str(Asym[1]) + "\n"
    # Modo 2 -- L2
    dataset += str(radioComponent2) + " " + str(sigmaComponent2) + " " + str(esfericidadComponent2) + " " + str(concentracionComponent2) + " " + str(L2) + " " + str(indRefReal_mode2_L2) + " " + str(indRefImag_mode2_L2) + " --- " + str(aod_mode_2_L2) + " " + str(ssa_mode_2_L2) + " " + str(Asym[3]) + "\n"
    # Modo 2 -- L3
    dataset += str(radioComponent2) + " " + str(sigmaComponent2) + " " + str(esfericidadComponent2) + " " + str(concentracionComponent2) + " " + str(L3) + " " + str(indRefReal_mode2_L3) + " " + str(indRefImag_mode2_L3) + " --- " + str(aod_mode_2_L3) + " " + str(ssa_mode_2_L3) + " " + str(Asym[5]) + "\n"
    # Modo 2 -- L4
    dataset += str(radioComponent2) + " " + str(sigmaComponent2) + " " + str(esfericidadComponent2) + " " + str(concentracionComponent2) + " " + str(L4) + " " + str(indRefReal_mode2_L4) + " " + str(indRefImag_mode2_L4) + " --- " + str(aod_mode_2_L4) + " " + str(ssa_mode_2_L4) + " " + str(Asym[7])  + "\n" 
    
    return dataset 

if __name__ == '__main__':
    fileDataset = open("dataset.txt", "w")
    erroresGrasp = open("erroresGrasp.txt", "w")
    for i in range(50000):
        output = "resultados/output"
        output = output + str(i).rjust(5,'0') + ".txt"
        try:
            lecturaDataSet = read_output(output)
            fileDataset.write(lecturaDataSet)
        except:
            erroresGrasp.write("Error " + str(i).rjust(5,'0') + ".txt\n")

    fileDataset.close()
    erroresGrasp.close()
        


