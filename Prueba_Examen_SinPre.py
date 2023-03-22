import csv
import getopt
import sys
import numpy as np
import pandas as pd
import sklearn as sk
import imblearn
import pickle
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

k=1
d=1
p='./'
f="datasetForTheExam_SubGrupo1.csv"
oFile="output.out"
a=''


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('ARGV   :',sys.argv[1:])
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'o:k:d:p:a:f:h',['output=','path=','iFile','h'])
    except getopt.GetoptError as err:
        print('ERROR:',err)
        sys.exit(1)
    print('OPTIONS   :',options)

    for opt,arg in options:
        if opt in ('-o','--output'):
            oFile = arg
        elif opt in ('-p', '--path'):
            p = arg
        elif opt in ('-f', '--file'):
            f = arg
        elif opt in ('-h','--help'):
            print(' -o outputFile \n -k numberOfItems \n -d distanceParameter \n -p inputFilePath \n -f inputFileName \n -a nombreAlgoritmo \n')
            exit(1)

    if p == './':
        iFile=p+str(f)
    else:
        iFile = p+"/" + str(f)
    # astype('unicode') does not work as expected

    def coerce_to_unicode(x):
        if sys.version_info < (3, 0):
            if isinstance(x, str):
                return unicode(x, 'utf-8')
            else:
                return unicode(x)
        else:
            return str(x)

    #Abrir el fichero .csv y cargarlo en un dataframe de pandas
    ml_dataset = pd.read_csv(iFile)

    #comprobar que los datos se han cargado bien. Cuidado con las cabeceras, la primera línea por defecto la considerara como la que almacena los nombres de los atributos
    # comprobar los parametros por defecto del pd.read_csv en lo referente a las cabeceras si no se quiere lo comentado

    #Printear los datos de las 5 primeras filas 
    ml_dataset=ml_dataset[['Area','Perimeter','Compactness','kernelLength','KernelWidth','AsymmetryCoeff','KernelGrooveLength','Class']]
    print("Forma dataframe:")
    print(ml_dataset.shape)




    # Según el tipo de feature, lo guardamos en el array correspondiente, y luego las columnas categoriales las pasamos a unicode (formato 'utf-8'). Los valores numericos los pasamos a floats

    categorical_features = ml_dataset.select_dtypes(include=['object'])
    numerical_features= ml_dataset.select_dtypes(include='number')

    #categorical_features = []
    #numerical_features = []
    text_features = []
    for feature in categorical_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

    for feature in text_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

    for feature in numerical_features:
        if ml_dataset[feature].dtype == np.dtype('M8[ns]') or (
                hasattr(ml_dataset[feature].dtype, 'base') and ml_dataset[feature].dtype.base == np.dtype('M8[ns]')):
            ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature])
        else:
            ml_dataset[feature] = ml_dataset[feature].astype('double')


    #Debemos cambiar lo que está entre comillas por los valores que pueden existir tipo 'perro':0 y 'gato':1. TARGET es la columna sobre la que queremos hacer la prediccion
    target_map = {'1': 0, '2': 1,'3':2}
    ml_dataset['__target__'] = ml_dataset['Class'].replace(target_map)
    print(ml_dataset.shape)
    print(ml_dataset.head(5))
    del ml_dataset['Class']

    # Eliminar datos nulos.
    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]
    print(f)
    print(ml_dataset.head(5))

    #Dividir la muestra en train y test, poniendo la proporcion de test, si queremos estratificarlo utilizamos la última opción
    train, test = train_test_split(ml_dataset,test_size=0.2,random_state=42,stratify=ml_dataset[['__target__']])
    print(train.head(5))
    #Contar valores en train
    print("Valores en train: "+str(train['__target__'].value_counts()))
    #Contar valores en test
    print("Valores en test: "+str(test['__target__'].value_counts()))

    #Ahora manejamos los missing values
    drop_rows_when_missing = []

    #Elegimos que hacer cuando hay missing values, en este caso pondremos la media, Ejemplo:[{'feature': 'num_var45_ult1', 'impute_with': 'MEAN'}]
    impute_when_missing = []

    # Por cada atributo que se encuentre en la lista, si se da el caso de que tienen un missing value, se borra esa fila
    for feature in drop_rows_when_missing:
        train = train[train[feature].notnull()]
        test = test[test[feature].notnull()]
        print('Dropped missing records in %s' % feature)

    # Dentro de las columnas que están en el array impute_when_missing, cuando se encuentre un missing value se sustituirá por la función que tenga cada valor
    for feature in impute_when_missing:
        if feature['impute_with'] == 'MEAN':
            v = train[feature['feature']].mean()
        elif feature['impute_with'] == 'MEDIAN':
            v = train[feature['feature']].median()
        elif feature['impute_with'] == 'CREATE_CATEGORY':
            v = 'NULL_CATEGORY'
        elif feature['impute_with'] == 'MODE':
            v = train[feature['feature']].value_counts().index[0]
        elif feature['impute_with'] == 'CONSTANT':
            v = feature['value']
        train[feature['feature']] = train[feature['feature']].fillna(v)
        test[feature['feature']] = test[feature['feature']].fillna(v)
        print('Imputed missing values in feature %s with value %s' % (feature['feature'], coerce_to_unicode(v)))


    #Columnas que se quieren reescalar, ejemplo: 'num_var45_ult1': 'AVGSTD'
    rescale_features = {}
    
    #Para cada atributo dentro del array anterior, realizar el método de reescalar que tenga
    for (feature_name, rescale_method) in rescale_features.items():
        if rescale_method == 'MINMAX':
            _min = train[feature_name].min()
            _max = train[feature_name].max()
            scale = _max - _min
            shift = _min
        else:
            shift = train[feature_name].mean()
            scale = train[feature_name].std()
        #Se borra la columna que no tenga varianza
        if scale == 0.:
            del train[feature_name]
            del test[feature_name]
            print('Feature %s was dropped because it has no variance' % feature_name)
        else:
            print('Rescaled %s' % feature_name)
            train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
            test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale

    #Separa el target de los label, tanto en test como train, trainX son los atributos y trainY es la clase
    trainX = train.drop('__target__', axis=1)
    #trainY = train['__target__']

    testX = test.drop('__target__', axis=1)
    #testY = test['__target__']

    trainY = np.array(train['__target__'])
    testY = np.array(test['__target__'])

    # Utilizamos undersample para reducir el número de casos que es mayoría y así estén balanceados
    #undersample = RandomUnderSampler(sampling_strategy=0.5)#la mayoria va a estar representada el doble de veces

    #trainXUnder,trainYUnder = undersample.fit_resample(trainX,trainY)
    #testXUnder,testYUnder = undersample.fit_resample(testX, testY)
    #Aquí hacemos el barrido de hiperparámetros
    print(trainX)
    print(trainY)               # Aquí creamos el modelo knn, siendo k=n_neighbors, weights como decidimos que valgan las distancias, uniform cada una vale lo mismo, y distance que valgan 
    clf = KNeighborsClassifier(n_neighbors=3,
                                weights='uniform',
                                algorithm='auto',
                                leaf_size=30,
                                p=1)

                        # Se especifica que las clases están balanceadas
    clf.class_weight = "balanced"

                        # Aquí entrenamos el modelo

    clf.fit(trainX, trainY)


                    # Build up our result dataset

                    # The model is now trained, we can apply it to our test set:

    predictions = clf.predict(testX)
    probas = clf.predict_proba(testX)
                    

    predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
    cols = [
                            u'probability_of_value_%s' % label
                            for (_, label) in sorted([(int(target_map[label]), label) for label in target_map])
                        ]
    probabilities = pd.DataFrame(data=probas, index=testX.index, columns=cols)

                    # Build scored dataset
    results_test = testX.join(predictions, how='left')
    results_test = results_test.join(probabilities, how='left')
    results_test = results_test.join(test['__target__'], how='left')
    results_test = results_test.rename(columns= {'__target__': 'Especie'})

    i=0
    for real,pred in zip(testY,predictions):
        print(real,pred)
        i+=1
        if i>5:
            break
                        #Printeamos las estadisticas que queramos
    f_score=f1_score(testY, predictions, average=None)
    f_score_avg=(f_score[0]+f_score[1]+f_score[2])/3
    rec=recall_score(testY,predictions,average=None)
    rec_avg=(rec[0]+rec[1]+rec[2])/3
    prec=precision_score(testY,predictions,average=None)
    prec_avg=(prec[0]+prec[1]+prec[2])/3
    inv_map = { target_map[label] : label for label in target_map}
    predictions.map(inv_map)
    print(f1_score(testY, predictions, average=None))
    print(classification_report(testY,predictions))
    print(confusion_matrix(testY, predictions, labels=[1,0]))

print("bukatu da")