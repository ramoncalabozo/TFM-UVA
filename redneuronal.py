import tensorflow as tf
from tensorflow import keras
import numpy as np

NUM = 372360 - 33995  # Numero de registros reales -  el numero de registros con aod < 2

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

    # Añadir checkpoint 
    checkpoint =  tf.keras.callbacks.ModelCheckpoint('MejorVersionModelo.h5', monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=False, mode="auto", period=1)
   #ESCRIBO EN UN FICHERO LOS LOGROS QUE VA HACIENDO EL MODELO
    csv_logger = tf.keras.callbacks.CSVLogger('log_model.txt', append=True, separator=',')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

    # Entrenar el modelo con los datos de entrenamiento
    history = model.fit(entradaEntrenamiento, salidaEntrenamiento, epochs=500, batch_size=32, validation_split=0.1,callbacks=[checkpoint,early_stopping, reduce_lr,csv_logger])
    
    print("\n\n\nEVALUACIÓN: \n")
    model=tf.keras.models.load_model( 'MejorVersionModelo.h5')
     # Evaluar el modelo con los datos de test
    test_loss, test_acc = model.evaluate(entradaTest, salidaTest)
    print("Test_accuracy: " +  str(test_acc) + "\n" + "Test_losser " + str(test_loss))
          

def recolectarDatos(archivoDataSet):
    numEntrenamiento = NUM * 0.8
    entradaEntrenamiento = []
    salidaEntrenamiento = []
    entradaTest = []
    salidaTest = []
    fileData = open(archivoDataSet, "r")
    content = fileData.readlines()

    j = 1
    for line in content:
        entrada = []
        salida = []
        line = line.replace("\n","")
        divisionEntradaSalida = line.split(" --- ")
        parametrosEntrada = divisionEntradaSalida[0].split(" ")
        parametrosSalida = divisionEntradaSalida[1].split(" ")
        
        if float(parametrosSalida[0]) < 2:
            for i in range(7):
                entrada.append(float(parametrosEntrada[i]))

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


def normalizarDatosEntradas(entrada):
    radioComponent1Max = 0.7
    sigmaComponent1Max = 3.0
    esfericidadComponent1Max = 100
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
    esfericidadComponent2Max = 100
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


def normalizarDatosSalida(salida):
    aodMax = 2
    for i in range(len(salida)):        
       salida[i][0] = salida[i][0] / aodMax
       
    return salida


if __name__ == '__main__':
    archivoDataSet = "dataset.txt"
    entradaEntrenamiento, salidaEntrenamiento, entradaTest, salidaTest = recolectarDatos(archivoDataSet)
    entradaEntrenamientoNorm = normalizarDatosEntradas(entradaEntrenamiento)
    entradaTestNorm = normalizarDatosEntradas(entradaTest)
    salidaEntrenamientoNorm = normalizarDatosSalida(salidaEntrenamiento)
    salidaTestNorm = normalizarDatosSalida(salidaTest)
    modeloEntrenado = crearModelo(entradaEntrenamientoNorm, salidaEntrenamientoNorm, entradaTestNorm, salidaTestNorm)

