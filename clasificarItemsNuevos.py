# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


model=""
p="./"
r=0
supervisado=1

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('ARGV   :',sys.argv[1:])
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'p:m:r:s:f:h',['path=','model=','pre=','testFile=','h'])
    except getopt.GetoptError as err:
        print('ERROR:',err)
        sys.exit(1)
    print('OPTIONS   :',options)

    for opt,arg in options:
        if opt in ('-p','--path'):
            p = arg
        elif opt in ('-f', '--file'):
            f = arg
        elif opt in ('-r', '--pre'):
            r = int(arg)
        elif opt in ('-s', '--super'):
            supervisado = int(arg)
        elif opt in ('-m', '--model'):
            m = arg
        elif opt in ('-h','--help'):
            print(' -p modelAndTestFilePath \n -m modelFileName \n -f testFileName \n -r preprocesado \n -s supervisado')
            exit(1)

    
    if p == './':
        model=p+str(m)
        iFile = p+ str(f)
    else:
        model=p+"/"+str(m)
        iFile = p+"/" + str(f)

    def coerce_to_unicode(x):
        if sys.version_info < (3, 0):
            if isinstance(x, str):
                return unicode(x, 'utf-8')
            else:
                return unicode(x)
        else:
            return str(x)
            
    #Abrir el fichero .csv con las instancias a predecir y que no contienen la clase y cargarlo en un dataframe de pandas para hacer la prediccion
    y_test=pd.DataFrame()
    testX = pd.read_csv(iFile)
    if(r==1):
        #Columnas a tener en cuenta en el test: testX=testX[['']]
        categorical_features = testX.select_dtypes(include=['object','category'])
        numerical_features= testX.select_dtypes(include='number')

        #categorical_features = []
        #numerical_features = []
        text_features = []
        for feature in categorical_features:
            testX[feature] = testX[feature].apply(coerce_to_unicode)

        for feature in text_features:
            testX[feature] = testX[feature].apply(coerce_to_unicode)

        for feature in numerical_features:
            if testX[feature].dtype == np.dtype('M8[ns]') or (
                    hasattr(testX[feature].dtype, 'base') and testX[feature].dtype.base == np.dtype('M8[ns]')):
                testX[feature] = datetime_to_epoch(testX[feature])
            else:
                testX[feature] = testX[feature].astype('double')
        drop_rows_when_missing = []

        #Elegimos que hacer cuando hay missing values, en este caso pondremos la media, Ejemplo:[{'feature': 'num_var45_ult1', 'impute_with': 'MEAN'}]
        impute_when_missing = [{'feature': 'Largo de sepalo', 'impute_with': 'MODE'},
                            {'feature': 'Ancho de sepalo', 'impute_with': 'MODE'},
                            {'feature': 'Largo de petalo', 'impute_with': 'MODE'},
                            {'feature': 'Ancho de petalo', 'impute_with': 'MODE'}]

        # Por cada atributo que se encuentre en la lista, si se da el caso de que tienen un missing value, se borra esa fila
        for feature in drop_rows_when_missing:
            testX = testX[testX[feature].notnull()]
            print('Dropped missing records in %s' % feature)

        # Dentro de las columnas que están en el array impute_when_missing, cuando se encuentre un missing value se sustituirá por la función que tenga cada valor
        for feature in impute_when_missing:
            if feature['impute_with'] == 'MEAN':
                v = testX[feature['feature']].mean()
            elif feature['impute_with'] == 'MEDIAN':
                v = testX[feature['feature']].median()
            elif feature['impute_with'] == 'CREATE_CATEGORY':
                v = 'NULL_CATEGORY'
            elif feature['impute_with'] == 'MODE':
                v = testX[feature['feature']].value_counts().index[0]
            elif feature['impute_with'] == 'CONSTANT':
                v = testX['value']
            testX[feature['feature']] = testX[feature['feature']].fillna(v)
            print('Imputed missing values in feature %s with value %s' % (feature['feature'], coerce_to_unicode(v)))


        #Columnas que se quieren reescalar, ejemplo: 'num_var45_ult1': 'AVGSTD'
        rescale_features = {'Largo de sepalo':'AVGSTD','Ancho de sepalo':'AVGSTD', 'Largo de petalo':'AVGSTD', 'Ancho de petalo':'AVGSTD'}
        
        #Para cada atributo dentro del array anterior, realizar el método de reescalar que tenga
        for (feature_name, rescale_method) in rescale_features.items():
            if rescale_method == 'MINMAX':
                _min = testX[feature_name].min()
                _max = testX[feature_name].max()
                scale = _max - _min
                shift = _min
            else:
                shift = testX[feature_name].mean()
                scale = testX[feature_name].std()
            #Se borra la columna que no tenga varianza
            if scale == 0.:
                del testX[feature_name]
                print('Feature %s was dropped because it has no variance' % feature_name)
            else:
                print('Rescaled %s' % feature_name)
                testX[feature_name] = (testX[feature_name] - shift).astype(np.float64) / scale

    #print(supervisado)
    #if(supervisado==0):
    #    del testX['Especie']

    clf = pickle.load(open(model, 'rb'))
    predictions = clf.predict(testX)
    probas = clf.predict_proba(testX)
    y_test['preds'] = predictions
    predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
    results_test = testX.join(predictions, how='left')
    
    print(results_test)
    

    
