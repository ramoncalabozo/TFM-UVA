import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def crearModelo(entradaEntrenamiento, salidaEntrenamiento, entradaTest, salidaTest, escala):
    # Crear el modelo de red neuronal 
    model = keras.Sequential([
        # Capa oculta con 64 neuronas y función de activación tanh
        keras.layers.Dense(64, activation='tanh', input_shape=(7,)),
        # Dropout con tasa de abandono del 20% NO!!
        # keras.layers.Dropout(0.2),
        # Capa oculta con 64 neuronas y función de activación tanh
        keras.layers.Dense(64, activation='tanh'),
        # keras.layers.Dropout(0.005),
        # Capa oculta con 64 neuronas y función de activación tanh
        keras.layers.Dense(64, activation='tanh'),
        # Capa densa de salida con 3 neuronas
        keras.layers.Dense(3)
    ])
    
# GENERO LOS PESOS QUE LE VOY A DAR A CADA UNO DE LOS 3 VALORES DE SALIDA
    suma=np.sum(salidaEntrenamiento[:,0])+np.sum(salidaEntrenamiento[:,1])+np.sum(salidaEntrenamiento[:,2])
    loss_weights = [suma/np.sum(salidaEntrenamiento[:,0]), suma/np.sum(salidaEntrenamiento[:,1]),suma/np.sum(salidaEntrenamiento[:,2])]
# NORMALIZO LOS PESOS AL PRIMER VALOR
    loss_weights=loss_weights/loss_weights[0]

    # opt = keras.optimizers.SGD(learning_rate=0.001)
    opt ='adam'
    # Compilar el modelo con optimizador Adam y función de pérdida mse
    model.compile(optimizer=opt,
            loss=['mse','mse','mse'],
                loss_weights=loss_weights,
                metrics=['accuracy'])

    # Añadir checkpoint 
    checkpoint =  tf.keras.callbacks.ModelCheckpoint('MejorVersionModelo.h5', monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=False, mode="auto", period=1)
   #ESCRIBO EN UN FICHERO LOS LOGROS QUE VA HACIENDO EL MODELO
    csv_logger = tf.keras.callbacks.CSVLogger('log_model.txt', append=True, separator=',')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

    # Entrenar el modelo con los datos de entrenamiento
    history = model.fit(entradaEntrenamiento, salidaEntrenamiento, epochs=500, batch_size=8, shuffle=True, validation_split=0.1,callbacks=[checkpoint,early_stopping, reduce_lr,csv_logger])
    # history = model.fit(entradaEntrenamiento, salidaEntrenamiento, epochs=500, batch_size=512, shuffle=True, callbacks=[csv_logger]) #esto si se quiere sin validation_split ni callbacks
    # model.save('VersionFinalModelo.h5') #esto si no se usa el validation_split entonces se guarda el ultimo modelo

    # VEO LOS RESULTADOS DEL MODELO
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    
    print("\n\n\nEVALUACIÓN: \n")
    model=tf.keras.models.load_model( 'MejorVersionModelo.h5')
    # model=tf.keras.models.load_model( 'VersionFinalModelo.h5') #este si se usa el modelo sin callbacks ni validation_split
     # Evaluar el modelo con los datos de test
    test_loss, test_acc = model.evaluate(entradaTest, salidaTest)
    print("Test_accuracy: " +  str(test_acc) + "\n" + "Test_losser " + str(test_loss))


    prediccion = model.predict(entradaTest)
    plt.plot((salidaTest[:,0]),prediccion[:,0],'.')
    plt.title('AOD')
    plt.ylabel('$AOD_{pred}$')
    plt.xlabel('$AOD_{ref}$')
    
    p=np.polyfit(salidaTest[:,0],prediccion[:,0],1) 
    r=np.corrcoef(salidaTest[:,0],prediccion[:,0])
    r2=r[0,1]*r[1,0]
    
    plt.plot([0, 2],[p[0]*0+p[1],p[0]*2+p[1]],'k')
    plt.text(0, 2, 'y='+str(np.round(p[1]*1000)/1000)+'+'+str(np.round(p[0]*1000)/1000)+'x', fontsize = 12)
    plt.text(0, 1.8, '$r^2$='+str(np.round(r2*1000)/1000), fontsize = 12)
    plt.show()
    
# EL SSA
    plt.plot((salidaTest[:,1]),prediccion[:,1],'.')
    plt.title('SSA')
    plt.ylabel('$SSA_{pred}$')
    plt.xlabel('$SSA_{ref}$')
    
    p=np.polyfit(salidaTest[:,1],prediccion[:,1],1) 
    r=np.corrcoef(salidaTest[:,1],prediccion[:,1])
    r2=r[0,1]*r[1,0]
    
    plt.plot([0, 1],[p[0]*0+p[1],p[0]*1+p[1]],'k')
    plt.text(0, 1, 'y='+str(np.round(p[1]*1000)/1000)+'+'+str(np.round(p[0]*1000)/1000)+'x', fontsize = 12)
    plt.text(0, 0.9, '$r^2$='+str(np.round(r2*1000)/1000), fontsize = 12)
    plt.show()

# EL G
    plt.plot((salidaTest[:,2]),prediccion[:,2],'.')
    plt.title('g')
    plt.ylabel('$g_{pred}$')
    plt.xlabel('$g_{ref}$')
    
    p=np.polyfit(salidaTest[:,2],prediccion[:,2],1) 
    r=np.corrcoef(salidaTest[:,2],prediccion[:,2])
    r2=r[0,1]*r[1,0]
    
    plt.plot([0, 1],[p[0]*0+p[1],p[0]*1+p[1]],'k')
    plt.text(0, 1, 'y='+str(np.round(p[1]*1000)/1000)+'+'+str(np.round(p[0]*1000)/1000)+'x', fontsize = 12)
    plt.text(0, 0.9, '$r^2$='+str(np.round(r2*1000)/1000), fontsize = 12)
    plt.show()

# LAS DIFERNECIAS
    diferenciaAOD=salidaTest[:,0]-prediccion[:,0]
    diferenciaSSA=salidaTest[:,1]-prediccion[:,1]
    diferenciaG=salidaTest[:,2]-prediccion[:,2]
    Nbins=200


    plt.plot((entradaTest[:,0])*escala[0],diferenciaAOD,'.')
    plt.title('Radio')
    plt.xlabel('Radio')
    plt.ylabel('$AOD_{pred}$-$AOD_{ref}$')
    plt.show()

    plt.plot((entradaTest[:,1])*escala[1],diferenciaAOD,'.')
    # plt.title('Concentracion')
    plt.xlabel('Sigma')
    plt.ylabel('$AOD_{pred}$-$AOD_{ref}$')
    plt.show()

    plt.plot((entradaTest[:,2])*escala[2],diferenciaAOD,'.')
    # plt.title('difAOD')
    plt.xlabel('esfericiad')
    plt.ylabel('$AOD_{pred}$-$AOD_{ref}$')
    plt.show()

    plt.plot((entradaTest[:,3])*escala[3],diferenciaAOD,'.')
    plt.xlabel('Concentracion')
    plt.ylabel('$AOD_{pred}$-$AOD_{ref}$')
    plt.show()

    plt.plot((entradaTest[:,4])*escala[4],diferenciaAOD,'.')
    plt.title('difAOD')
    plt.xlabel('longitud de onda')
    plt.ylabel('$AOD_{pred}$-$AOD_{ref}$')
    plt.show()

    plt.plot((entradaTest[:,5])*escala[5],diferenciaAOD,'.')
    plt.xlabel('RRI')
    plt.ylabel('$AOD_{pred}$-$AOD_{ref}$')
    plt.show()

    plt.plot((entradaTest[:,6])*escala[6],diferenciaAOD,'.')
    plt.xlabel('IMI')
    plt.ylabel('$AOD_{pred}$-$AOD_{ref}$')
    plt.show()
    
# LOS HISTOGRAMAS
# EL AOD    
    n=plt.hist(diferenciaAOD,bins=Nbins)
    plt.title('AOD')
    plt.ylabel('N')
    plt.xlabel('$AOD_{pred}$-$AOD_{ref}$')

    MBE=np.mean(diferenciaAOD) 
    Md=np.median(diferenciaAOD) 
    STD=np.std(diferenciaAOD) 
    plt.text(np.min(diferenciaAOD), np.max(n[0])*0.95, 'Mean='+str(np.round(MBE*1000)/1000), fontsize = 12)
    plt.text(np.min(diferenciaAOD), np.max(n[0])*0.88, 'Median='+str(np.round(Md*1000)/1000), fontsize = 12)
    plt.text(np.min(diferenciaAOD), np.max(n[0])*0.81, 'STD='+str(np.round(STD*1000)/1000), fontsize = 12)
    plt.show()
    
# EL SSA    
    n=plt.hist(diferenciaSSA,bins=Nbins)
    plt.title('SSA')
    plt.ylabel('N')
    plt.xlabel('$SSA_{pred}$-$SSA_{ref}$')

    MBE=np.mean(diferenciaSSA) 
    Md=np.median(diferenciaSSA) 
    STD=np.std(diferenciaSSA) 
    plt.text(np.min(diferenciaSSA), np.max(n[0])*0.95, 'Mean='+str(np.round(MBE*1000)/1000), fontsize = 12)
    plt.text(np.min(diferenciaSSA), np.max(n[0])*0.88, 'Median='+str(np.round(Md*1000)/1000), fontsize = 12)
    plt.text(np.min(diferenciaSSA), np.max(n[0])*0.81, 'STD='+str(np.round(STD*1000)/1000), fontsize = 12)
    plt.show()
    
    
# EL G    
    n=plt.hist(diferenciaG,bins=Nbins)
    plt.title('g')
    plt.ylabel('N')
    plt.xlabel('$g_{pred}$-$g_{ref}$')

    MBE=np.mean(diferenciaG) 
    Md=np.median(diferenciaG) 
    STD=np.std(diferenciaG) 
    plt.text(np.min(diferenciaG), np.max(n[0])*0.95, 'Mean='+str(np.round(MBE*1000)/1000), fontsize = 12)
    plt.text(np.min(diferenciaG), np.max(n[0])*0.88, 'Median='+str(np.round(Md*1000)/1000), fontsize = 12)
    plt.text(np.min(diferenciaG), np.max(n[0])*0.81, 'STD='+str(np.round(STD*1000)/1000), fontsize = 12)
    plt.show()

    
def recolectarDatos(archivoDataSet, numEntrenamiento):
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


def normalizarDatosEntradas(parametros):
    cicle = 0

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

    for entrada in parametros:
        cicle = cicle % 8
        
        if cicle <= 3:
            entrada[0] = entrada[0] / radioComponent1Max
            entrada[1] = entrada[1] / sigmaComponent1Max
            entrada[2] = entrada[2] / esfericidadComponent1Max
            entrada[3] = entrada[3] / concentracionComponent1Max
            if cicle == 0:
                entrada[4] = entrada[4] / L1Max
                entrada[5] = entrada[5] / indRefReal_mode1_L1Max
                entrada[6] = entrada[6] / indRefImag_mode1_L1Max
            if cicle == 1:
                entrada[4] = entrada[4] / L2Max
                entrada[5] = entrada[5] / indRefReal_mode1_L2Max
                entrada[6] = entrada[6] / indRefImag_mode1_L2Max
            if cicle == 2:
                entrada[4] = entrada[4] / L3Max
                entrada[5] = entrada[5] / indRefReal_mode1_L3Max
                entrada[6] = entrada[6] / indRefImag_mode1_L3Max
            if cicle == 3:
                entrada[4] = entrada[4] / L4Max
                entrada[5] = entrada[5] / indRefReal_mode1_L4Max
                entrada[6] = entrada[6] / indRefImag_mode1_L4Max
        else:
            entrada[0] = entrada[0] / radioComponent2Max
            entrada[1] = entrada[1] / sigmaComponent2Max
            entrada[2] = entrada[2] / esfericidadComponent2Max
            entrada[3] = entrada[3] / concentracionComponent2Max
            if cicle == 4:
                entrada[4] = entrada[4] / L1Max
                entrada[5] = entrada[5] / indRefReal_mode2_L1Max
                entrada[6] = entrada[6] / indRefImag_mode2_L1Max
            if cicle == 5:
                entrada[4] = entrada[4] / L2Max
                entrada[5] = entrada[5] / indRefReal_mode2_L2Max
                entrada[6] = entrada[6] / indRefImag_mode2_L2Max
            if cicle == 6:
                entrada[4] = entrada[4] / L3Max
                entrada[5] = entrada[5] / indRefReal_mode2_L3Max
                entrada[6] = entrada[6] / indRefImag_mode2_L3Max
            if cicle == 7:
                entrada[4] = entrada[4] / L4Max
                entrada[5] = entrada[5] / indRefReal_mode2_L4Max
                entrada[6] = entrada[6] / indRefImag_mode2_L4Max  
        cicle +=1                     

    return parametros

if __name__ == '__main__':
    archivoDataSet = "dataset.txt"
    
    numeroEntrenamiento = 372360 * 0.8
    modulo = numeroEntrenamiento % 8
    numeroEntrenamiento = numeroEntrenamiento + (8 - modulo)
    
    entradaEntrenamiento, salidaEntrenamiento, entradaTest, salidaTest = recolectarDatos(archivoDataSet, numeroEntrenamiento)
    
    # entradaEntrenamiento = normalizarDatosEntradas(entradaEntrenamiento)
    # entradaTest = normalizarDatosEntradas(entradaTest)

# QUITO LOS DATOS CON AODs MAYORES  DE 2
    entradaEntrenamiento=entradaEntrenamiento[salidaEntrenamiento[:,0]<2,:]
    salidaEntrenamiento=salidaEntrenamiento[salidaEntrenamiento[:,0]<2,:]
    entradaTest=entradaTest[salidaTest[:,0]<2,:]
    salidaTest=salidaTest[salidaTest[:,0]<2,:]

# QUITO LOS DATOS CON RADIOS Y SIGMAS MENORES DE 0.1
    salidaEntrenamiento=salidaEntrenamiento[entradaEntrenamiento[:,1]>=0.1,:]
    entradaEntrenamiento=entradaEntrenamiento[entradaEntrenamiento[:,1]>=0.1,:]
    salidaTest=salidaTest[entradaTest[:,1]>=0.1,:]
    entradaTest=entradaTest[entradaTest[:,1]>=0.1,:]

    salidaEntrenamiento=salidaEntrenamiento[entradaEntrenamiento[:,1]>=0.1,:]
    entradaEntrenamiento=entradaEntrenamiento[entradaEntrenamiento[:,1]>=0.1,:]
    salidaTest=salidaTest[entradaTest[:,1]>=0.1,:]
    entradaTest=entradaTest[entradaTest[:,1]>=0.1,:]

# INIZIALIZO LOS VALORES DE ENTRADA QUE VAN A NORMALIZARSE
    entradaEntrenamientoNorm = (entradaEntrenamiento)
    entradaTestNorm = (entradaTest)

# escala será el valor con el que voy a normalizar cada variable    
    escala=np.zeros(7)
    for n in range(7): #barro para cada variable de entrada
        # concateno todos los valores disponibles de la variable n
        conca=np.append(entradaEntrenamiento[:,n],entradaTest[:,n])
        # busco el máximo valor de la variable n
        maxi=np.max(conca)
        # normalizo a ese valor maximo
        entradaEntrenamientoNorm[:,n]=(entradaEntrenamientoNorm[:,n])/maxi
        entradaTestNorm[:,n]=(entradaTestNorm[:,n])/maxi
        # guardo ese valor de normalizacion en escala
        escala[n]=maxi

    modeloEntrenado = crearModelo(entradaEntrenamiento, salidaEntrenamiento, entradaTest, salidaTest, escala)

