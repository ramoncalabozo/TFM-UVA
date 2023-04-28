import tensorflow as tf
from tensorflow import keras
import numpy as np

NUM = 500

def crearModelo(entradaEntrenamiento, salidaEntrenamiento, entradaTest, salidaTest):

    # Crear el modelo de red neuronal
    model = keras.Sequential([
        # Capa oculta con 16 neuronas y función de activación ReLU
        keras.layers.Dense(16, activation='relu', input_shape=(7,)),
        # Dropout con tasa de abandono del 20%
        keras.layers.Dropout(0.2),
        # Capa oculta con 8 neuronas y función de activación ReLU
        keras.layers.Dense(8, activation='relu'),
        # Capa densa de salida con 3 neuronas y función de activación softmax
        keras.layers.Dense(3, activation='softmax')
    ])

    # Compilar el modelo con optimizador Adam y función de pérdida categorical crossentropy
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
     # Entrenar el modelo con los datos de entrenamiento
    history = model.fit(entradaEntrenamiento, salidaEntrenamiento, epochs=50, batch_size=32, validation_split=0.2)

     # Evaluar el modelo con los datos de test
    test_loss, test_acc = model.evaluate(entradaTest, salidaTest)
    print("Test accuracy: " +  str(test_acc) + " Test losser " + str(test_loss))
          

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

if __name__ == '__main__':
    archivoDataSet = "dataset.txt"
    entradaEntrenamiento, salidaEntrenamiento, entradaTest, salidaTest = datosEntrada(archivoDataSet)
    modeloEntrenado = crearModelo(entradaEntrenamiento, salidaEntrenamiento, entradaTest, salidaTest)

