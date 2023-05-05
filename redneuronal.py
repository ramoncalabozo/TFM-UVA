import tensorflow as tf
from tensorflow import keras
import numpy as np

NUM = 50000

def crearModelo(entradaEntrenamiento, salidaEntrenamiento, entradaTest, salidaTest):

    # Crear el modelo de red neuronal
    model = keras.Sequential([
        # Capa oculta con 16 neuronas y función de activación ReLU
        keras.layers.Dense(64, activation='relu', input_shape=(7,)),
        # Dropout con tasa de abandono del 20%
        keras.layers.Dropout(0.2),
        # Capa oculta con 8 neuronas y función de activación ReLU
        keras.layers.Dense(32, activation='relu'),
        # Capa densa de salida con 3 neuronas y función de activación softmax
        keras.layers.Dense(3)
    ])

    # Compilar el modelo con optimizador Adam y función de pérdida categorical crossentropy
    model.compile(optimizer='adam',
                loss='mse',
                metrics=['accuracy'])
    
     # Entrenar el modelo con los datos de entrenamiento
    history = model.fit(entradaEntrenamiento, salidaEntrenamiento, epochs=50, batch_size=32, validation_split=0.2)

    print("\n\n\nEVALUACIÓN: \n")
     # Evaluar el modelo con los datos de test
    test_loss, test_acc = model.evaluate(entradaTest, salidaTest)
    print("Test_accuracy: " +  str(test_acc) + "\n" + "Test_losser " + str(test_loss))
          

def datosEntrada(archivoDataSet):
    numEntrenamiento = NUM * 0.7    
    entradaEntrenamiento = []
    salidaEntrenamiento = []
    entradaTest = []
    salidaTest = []
    fileData = open(archivoDataSet, "r")
    content = fileData.readlines()

    j = 0
    for line in content:
        entrada = []
        salida = []
        line = line.replace("\n","")
        divisionEntradaSalida = line.split(" --- ")
        parametrosEntrada = divisionEntradaSalida[0].split(" ")
        for i in range(7):
            entrada.append(float(parametrosEntrada[i]))

        parametrosSalida = divisionEntradaSalida[1].split(" ")
        for i in range(3):
            salida.append(float(parametrosSalida[i]))

        if j < numEntrenamiento:
            entradaEntrenamiento.append(entrada)
            salidaEntrenamiento.append(salida)
        else:
            entradaTest.append(entrada)
            salidaTest.append(salida)
        j = j + 1  

    entradaEntrenamiento = np.array(entradaEntrenamiento)
    salidaEntrenamiento = np.array(salidaEntrenamiento)
    entradaTest = np.array(entradaTest)
    salidaTest = np.array(salidaTest)
    return entradaEntrenamiento, salidaEntrenamiento, entradaTest, salidaTest


def normalizarDatos(entrada):
    radioComponent1Max = 0.7
    sigmaComponent1Max = 3.0
    esfericidadComponent1Max = 1.0
    concentracionComponent1Max = 5.0

    indRefReal_mode1_L1Max = 1.6
    indRefReal_mode1_L2Max = 1.6
    indRefReal_mode1_L3Max = 1.6
    indRefReal_mode1_L4Max = 1.6

    indRefImag_mode1_L1Max = 0.5
    indRefImag_mode1_L2Max = 0.5
    indRefImag_mode1_L3Max = 0.5
    indRefImag_mode1_L4Max = 0.5

    radioComponent2Max = 10.0
    sigmaComponent2Max = 3.0
    esfericidadComponent2Max = 1.0
    concentracionComponent2Max = 5.0

    indRefReal_mode2_L1Max = 1.6
    indRefReal_mode2_L2Max = 1.6
    indRefReal_mode2_L3Max = 1.6
    indRefReal_mode2_L4Max = 1.6

    indRefImag_mode2_L1Max = 0.5
    indRefImag_mode2_L2Max = 0.5
    indRefImag_mode2_L3Max = 0.5
    indRefImag_mode2_L4Max = 0.5

    L1Max = 2.5
    L2Max = 2.5
    L3Max = 2.5
    L4Max = 2.5

    j = 0
    for i in range(len(entrada)): 
      if j <= 3:
        entrada[i][0] = entrada[i][0] / radioComponent1Max
        entrada[i][1] = entrada[i][1] / sigmaComponent1Max
        entrada[i][2] = entrada[i][2] / esfericidadComponent1Max
        entrada[i][3] = entrada[i][3] / concentracionComponent1Max
        if j == 0:
            entrada[i][4] = entrada[i][4] / L1Max
            entrada[i][5] = entrada[i][5] / indRefReal_mode1_L1Max
            entrada[i][6] = entrada[i][6] / indRefImag_mode1_L1Max
        if j == 1:
            entrada[i][4] = entrada[i][4] / L2Max
            entrada[i][5] = entrada[i][5] / indRefReal_mode1_L2Max
            entrada[i][6] = entrada[i][6] / indRefImag_mode1_L2Max
        if j == 2:
            entrada[i][4] = entrada[i][4] / L3Max
            entrada[i][5] = entrada[i][5] / indRefReal_mode1_L3Max
            entrada[i][6] = entrada[i][6] / indRefImag_mode1_L3Max
        if j == 3:
            entrada[i][4] = entrada[i][4] / L4Max
            entrada[i][5] = entrada[i][5] / indRefReal_mode1_L4Max
            entrada[i][6] = entrada[i][6] / indRefImag_mode1_L4Max
    else:
        entrada[i][0] = entrada[i][0] / radioComponent2Max
        entrada[i][1] = entrada[i][1] / sigmaComponent2Max
        entrada[i][2] = entrada[i][2] / esfericidadComponent2Max
        entrada[i][3] = entrada[i][3] / concentracionComponent2Max
        if j == 4:
            entrada[i][4] = entrada[i][4] / L1Max
            entrada[i][5] = entrada[i][5] / indRefReal_mode2_L1Max
            entrada[i][6] = entrada[i][6] / indRefImag_mode2_L1Max
        if j == 5:
            entrada[i][4] = entrada[i][4] / L2Max
            entrada[i][5] = entrada[i][5] / indRefReal_mode2_L2Max
            entrada[i][6] = entrada[i][6] / indRefImag_mode2_L2Max
        if j == 6:
            entrada[i][4] = entrada[i][4] / L3Max
            entrada[i][5] = entrada[i][5] / indRefReal_mode2_L3Max
            entrada[i][6] = entrada[i][6] / indRefImag_mode2_L3Max
        if j == 7:
            entrada[i][4] = entrada[i][4] / L4Max
            entrada[i][5] = entrada[i][5] / indRefReal_mode2_L4Max
            entrada[i][6] = entrada[i][6] / indRefImag_mode2_L4Max
            j = -1                    
        
        j += 1
    
    return entrada

if __name__ == '__main__':
    archivoDataSet = "dataset.txt"
    entradaEntrenamiento, salidaEntrenamiento, entradaTest, salidaTest = datosEntrada(archivoDataSet)
    entradaEntrenamientoNorm = normalizarDatos(entradaEntrenamiento)
    entradaTestNorm = normalizarDatos(entradaTest)
    modeloEntrenado = crearModelo(entradaEntrenamientoNorm, salidaEntrenamiento, entradaTestNorm, salidaTest)

