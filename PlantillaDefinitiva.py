# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
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
f=""
oFile="output.out"
a=''
balanced=0
pre=0

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('ARGV   :',sys.argv[1:])
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'o:k:d:p:r:b:a:f:h',['output=','k=','d=','r=','path=','a=','b=','iFile','h'])
    except getopt.GetoptError as err:
        print('ERROR:',err)
        sys.exit(1)
    print('OPTIONS   :',options)

    for opt,arg in options:
        if opt in ('-o','--output'):
            oFile = arg
        elif opt == '-k':
            K = int(arg)
        elif opt == '-b':
            balanced = int(arg)
        elif opt == '-r':
            pre = int(arg)
        elif opt ==  '-d':
            D = int(arg)
        elif opt ==  '-a':
            a = arg
        elif opt in ('-p', '--path'):
            p = arg
        elif opt in ('-f', '--file'):
            f = arg
        elif opt in ('-h','--help'):
            print(' -o outputFile \n -k numberOfItems \n -d distanceParameter \n -p inputFilePath \n -f Fichero csv con la muestra \n -a nombreAlgoritmo tiene que ser KNN o decisionTree\n -b Si queremos balancear la muestra o no \n-r Si queremos preprocesar los datos = 1\n')
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

    ml_dataset=ml_dataset[['Largo de sepalo','Ancho de sepalo','Largo de petalo','Ancho de petalo','Especie']]
    #Printear los datos de las 5 primeras filas 
    print(ml_dataset.shape)
    
     # Según el tipo de feature, lo guardamos en el array correspondiente, y luego las columnas categoriales las pasamos a unicode (formato 'utf-8'). Los valores numericos los pasamos a floats

    categorical_features = ml_dataset.select_dtypes(include=['object','category'])
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
    #Para cambiar una variable de cat a num, debemos hacer esto:
    cat_columns = categorical_features.columns
    #for col in cat_columns:
    #    ml_dataset[col] = pd.factorize(ml_dataset[col])[0]

    #Debemos cambiar lo que está entre comillas por los valores que pueden existir tipo 'perro':0 y 'gato':1. TARGET es la columna sobre la que queremos hacer la prediccion
    target_map = {'Iris-setosa': 0, 'Iris-versicolor': 1,'Iris-virginica':2}

    ml_dataset['__target__'] = ml_dataset['Especie'].map({'Iris-setosa' :0,'Iris-versicolor' :1,'Iris-virginica':2}).astype(int)
    #Si los atributos son numéricos: el target map sin comillas, y la columna target así: ml_dataset['__target__'] = ml_dataset['Especie'].map(target_map)
    print(ml_dataset.head(5))
    del ml_dataset['Especie']

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
    if(pre==1):
        #Ahora manejamos los missing values
        drop_rows_when_missing = []

        #Elegimos que hacer cuando hay missing values, en este caso pondremos la media, Ejemplo:[{'feature': 'num_var45_ult1', 'impute_with': 'MEAN'}]
        impute_when_missing = [{'feature': 'Largo de sepalo', 'impute_with': 'MODE'},
                            {'feature': 'Ancho de sepalo', 'impute_with': 'MODE'},
                            {'feature': 'Largo de petalo', 'impute_with': 'MODE'},
                            {'feature': 'Ancho de petalo', 'impute_with': 'MODE'}]

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
        rescale_features = {'Largo de sepalo':'AVGSTD','Ancho de sepalo':'AVGSTD', 'Largo de petalo':'AVGSTD', 'Ancho de petalo':'AVGSTD'}
        
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
    # Si es multiclass se tiene que utilizar como sampling strategy 'not minority', si es binaria se puede utilizar como automático 0.5
    if(balanced==1):
        undersample = RandomUnderSampler(sampling_strategy='not majority')
        trainX,trainY = undersample.fit_resample(trainX,trainY)
        testXU,testY = undersample.fit_resample(testX, testY)

    best_fscore=0
    K_best=0
    D_best=0
    peso=""
    J_best=0
    #Aquí hacemos el barrido de hiperparámetros
    print(trainX)
    print(trainY)
    if (a=='KNN'):
        #Creamos el csv:
        with open('datosKNN.csv','w',newline='')as archivo_csv:
            writer=csv.writer(archivo_csv,delimiter=',')
            writer.writerow(['Combinación','Precisión','Recall','F-Score'])
            k=1
            while(k<=K):
                d=1
                while(d<=D):
                    s=0
                    weight='uniform'
                    while(s<=1):
                        print("Comprobamos el algoritmo con el hiperparámetro K:"+str(k)+" y la P: "+str(d)+", y el weight establecido en "+weight)
                        # Aquí creamos el modelo knn, siendo k=n_neighbors, weights como decidimos que valgan las distancias, uniform cada una vale lo mismo, y distance que valgan 
                        clf = KNeighborsClassifier(n_neighbors=k,
                                            weights=weight,
                                            algorithm='auto',
                                            leaf_size=30,
                                            p=d)

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
                        writer.writerow(['k= '+str(k)+'p= '+str(d)+' '+weight,str(prec_avg),str(rec_avg),str(f_score_avg)])
                        print(classification_report(testY,predictions))
                        print(confusion_matrix(testY, predictions, labels=[1,0]))
                        weight='distance'
                        s=s+1
                        if(f_score_avg>best_fscore):
                            best_fscore=f_score_avg
                            K_best=k
                            D_best=d
                            peso=weight
                        nombreModel = str(a)+"_"+str(k)+"_"+str(d)+"_"+peso+".sav"
                        saved_model = pickle.dump(clf, open(nombreModel,"wb"))
                        print("Modelo guardado correctamente empleando Pickle")

                    d=d+1
                k=k+2
    elif a == "decisionTree":
        with open('datosDecisionTree.csv','w',newline='')as archivo_csv:
            writer=csv.writer(archivo_csv,delimiter=',')
            writer.writerow(['Combinación','Precisión','Recall','F-Score'])
            k=3
            while(k<=K):
                d=1
                while(d<=2):
                    j=1 
                    while(j<=2):
                        clf = DecisionTreeClassifier(
                                        random_state = 1337,
                                        criterion = 'gini',
                                        splitter = 'best',
                                        max_depth = k,
                                        min_samples_leaf = d,
                                        min_samples_split=j,
                                )

                            # Explica lo que se hace en este paso
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
                            
                        inv_map = { target_map[label] : label for label in target_map}
                        predictions.map(inv_map)
                        print("Esta es la prueba de max_depth: "+str(k)+" min_samples_leaf: "+str(d)+"min_samples_split: "+str(j))
                        print(f1_score(testY, predictions, average=None))
                        print(classification_report(testY,predictions))
                        print(confusion_matrix(testY, predictions, labels=[1,0]))
                        f_score=f1_score(testY, predictions, average=None)
                        f_score_avg=(f_score[0]+f_score[1]+f_score[2])/3
                        rec=recall_score(testY,predictions,average=None)
                        rec_avg=(rec[0]+rec[1]+rec[2])/3
                        prec=precision_score(testY,predictions,average=None)
                        prec_avg=(prec[0]+prec[1]+prec[2])/3
                        writer.writerow(['k= '+str(k)+' min_sample_length= '+str(d)+' min_sample_split= '+str(j),str(prec_avg),str(rec_avg),str(f_score_avg)])
                        j=j+1
                        if(f_score_avg>best_fscore):
                            best_fscore=f_score_avg
                            K_best=k
                            D_best=d
                            J_best=j
                    d=d+1
                    if(d==3):
                        writer.writerow([])
                k=k+3
                    
    else:
        print("Solo tenemos dos algoritmos disponibles KNN o decisionTree, debe indicar uno")            

    if(a=="KNN"):
        print("El mejor modelo de KNN tiene un f_score de: "+str(best_fscore)+" , y es el que tiene k= "+str(K_best)+" y d= "+str(D_best)+"con peso= "+str(peso))
        clf = KNeighborsClassifier(n_neighbors=K_best,
                                            weights=peso,
                                            algorithm='auto',
                                            leaf_size=30,
                                            p=D_best)
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
                            
        inv_map = { target_map[label] : label for label in target_map}
        predictions.map(inv_map)        

        nombreModel = str(a)+"_"+str(K_best)+"_"+str(D_best)+"_"+str(peso)+".sav"
        saved_model = pickle.dump(clf, open(nombreModel,"wb"))
        print("Modelo guardado correctamente empleando Pickle")

    else:
        print("El mejor modelo de Decision Tree tiene un f_score de : "+str(best_fscore)+" , con max_depth= "+str(K_best)+" min_sample_length= "+str(D_best)+" y min_sample_split= "+str(J_best))
        clf = DecisionTreeClassifier(
                                        random_state = 1337,
                                        criterion = 'gini',
                                        splitter = 'best',
                                        max_depth = K_best,
                                        min_samples_leaf = D_best,
                                        min_samples_split=J_best,
                                )
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
                            
        inv_map = { target_map[label] : label for label in target_map}
        predictions.map(inv_map)        
        nombreModel = str(a)+"_"+str(K_best)+"_"+str(D_best)+"_"+str(J_best)+".sav"
        saved_model = pickle.dump(clf, open(nombreModel,"wb"))
        print("Modelo guardado correctamente empleando Pickle")
print("bukatu da")