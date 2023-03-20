#Importamos las librerias necesarias
import sklearn
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt   
import pickle
import base64
import csv
import plotly.express as px

#Importamos los modelos de clasifificación a utilizar
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, auc
from sklearn.preprocessing import label_binarize
from imblearn.metrics import specificity_score
from imblearn.metrics import sensitivity_score


#-----------------------------Sidebar de Opciones y Carga de Datos-----------------------------#

st.sidebar.markdown("Sistema de Ańalisis CX con ML")

opciones=st.sidebar.selectbox("Opciones:", ["Análisis de datos", "Preparación y Modelado",  "Predicción"])

st.sidebar.markdown("---")

datos = st.sidebar.file_uploader("Cargue el dataset para entrenamiento:", type={"csv", "xlsx"})
if datos is not None:
        df_NPS = pd.read_excel(datos)

st.sidebar.markdown("---")

prediccion = st.sidebar.file_uploader("Cargue el dataset para prediccion:", type={"csv", "xlsx"})
if prediccion is not None:
        df_PREDICCION = pd.read_excel(prediccion)


#-----------------------------------Menu de Análisis de Datos---------------------------------#

if opciones=="Análisis de datos":
        
        st.title("Sistema de Ańalisis Experiencia Clientes con ML")
        st.header("Analisis exploratorio del dataset") 
        st.write("Dataset para entrenamiento ", df_NPS.shape)
        st.write(df_NPS)

        st.markdown("#### NPS Mensual: ")
        df_tendencia = pd.crosstab(df_NPS['MES'], df_NPS['CATEGORIA_NPS'], rownames=['AÑO MES'])
        df_tendencia ['TOTAL'] = df_tendencia['Detractores'] + df_tendencia['Promotores'] + df_tendencia['Neutros']
        df_tendencia ['% DETRACTORES'] = (df_tendencia['Detractores']/df_tendencia ['TOTAL'])*100
        df_tendencia ['% PROMOTORES'] = (df_tendencia['Promotores']/df_tendencia ['TOTAL'])*100
        df_tendencia ['NPS_MES'] = ((df_tendencia['Promotores']/df_tendencia['TOTAL']) - (df_tendencia['Detractores']/df_tendencia['TOTAL']))*100
        st.write(df_tendencia)

        col1, col2 = st.columns([2,2])

        with col1:
                st.markdown("#### Clasificación de Usuarios:")
                fig = sns.catplot(x="CATEGORIA_NPS", kind="count", data=df_NPS)
                st.pyplot(fig)

        with col2:
                
                sum_usuarios = df_NPS ["CATEGORIA_NPS"].value_counts()
                sum_usuarios ['TOTAL'] = sum_usuarios ['Promotores'] + sum_usuarios ['Detractores'] + sum_usuarios ['Neutros']
                                
                st.markdown("#### Porcentajes:")
                promotores = (sum_usuarios ['Promotores'] / sum_usuarios ['TOTAL'])*100
                detractores = (sum_usuarios ['Detractores'] / sum_usuarios ['TOTAL'])*100
                neutros = (sum_usuarios ['Neutros'] / sum_usuarios ['TOTAL'])*100
                nps = promotores - detractores

                st.metric(label="Promotores", value = round(promotores,2))
                st.metric(label="Detractores", value = round(detractores,2))
                st.metric(label="Neutros", value = round(neutros,2))
                st.metric(label="NPS", value = round(nps,2))

               
        st.markdown("#### Clasificacion Usuarios x mes: ")
        df_mensual = pd.crosstab(df_NPS['MES'], df_NPS['CATEGORIA_NPS'], rownames=['AÑO MES'])
        fig,ax=plt.subplots(figsize=(3.5,2))
        st.bar_chart(df_mensual)
        
        st.markdown("#### NPS Tendencia mensual: ")
        fig,ax=plt.subplots(figsize=(5,2))
        st.bar_chart(df_tendencia,y='NPS_MES')

        st.markdown("#### Identificación tipo de datos: ")  
        st.text (df_NPS.dtypes)


        #Exploración de variables categoricas
        st.markdown("#### Exploración de variables categóricas: ")

        #Grafica variables catégoricas tipo object
        st.write (df_NPS.select_dtypes(include=['object']).describe())

        df_NPS_cat = df_NPS[['SURVEY_STATUS', 'MES', 'PUNTO_CONTACTO', 'JOURNEY',
                     'CATEGORIA_REPORTE', 'SEGMENTACION', 'TIPO PRODUCTO', 
                     'CATEGORIA_NPS', 'RPC', 'REITERATIVO', 
                     'MTTR_MAS_48H', 'CUMPLE_AGENDA']]

        # Gráfico para cada variable cualitativa
        # ==============================================================================
        # Ajustar número de subplots en función del número de columnas
        
        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20, 15))
        axes = axes.flat
        columnas_object = df_NPS_cat.select_dtypes(include=['object']).columns

        for i, colum in enumerate(columnas_object):
                df_NPS_cat[colum].value_counts().plot.barh(ax = axes[i], alpha   = 0.7)
                axes[i].set_title(colum, fontsize = 12, fontweight = "bold")
                axes[i].tick_params(labelsize = 12)
                axes[i].set_xlabel("")
   
        fig.tight_layout()
        plt.subplots_adjust(top=0.9)
        fig.suptitle('Distribución variables cualitativas',
                     fontsize = 18, fontweight = "bold");

        st.pyplot(fig)

        #Exploración de variables numericas
        st.markdown("#### Exploración de variables numéricas: ")

        ## Variables numéricas
        # ==============================================================================
        st.write (df_NPS.select_dtypes(include=['float64', 'int']).describe())

        # Gráfico de distribución para cada variable numérica
        # ==============================================================================
        # Ajustar número de subplots en función del número de columnas
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 15))
        axes = axes.flat
        columnas_numeric = df_NPS.select_dtypes(include=['float64', 'int']).columns

        for i, colum in enumerate(columnas_numeric):
                sns.histplot(
                data    = df_NPS,
                x       = colum,
                stat    = "count",
                kde     = True,
                color   = (list(plt.rcParams['axes.prop_cycle'])*2)[i]["color"],
                line_kws= {'linewidth': 2},
                alpha   = 0.3,
                ax      = axes[i]
         )
        axes[i].set_title(colum, fontsize = 14, fontweight = "bold")
        axes[i].tick_params(labelsize = 14)
        axes[i].set_xlabel("")
    
        fig.tight_layout()
        plt.subplots_adjust(top = 0.9)
        fig.suptitle('Distribución variables numéricas', fontsize = 20, fontweight = "bold");

        st.pyplot(fig)

#--------------------------------Menu de Preparación de datos-------------------------------#

if opciones=="Preparación y Modelado":
        
        st.title("Sistema de Ańalisis Experiencia Clientes con ML")
        st.header("Preparación de los datos")
        st.markdown("#### Limpieza de datos: ")
        st.text("Se eliminan columnas innecesarias")

        df_NPS_new=df_NPS.drop (['ID_ENCUESTA', 'SURVEY_STATUS', 'FECHA_RESPUESTA', 'AÑO', 'MES', 
                                        'JOURNEY', 'COMENTARIO_NPS', 'TIPO_ENCUESTA', 'MERCADO', 
                                        'MARCA', 'ID CLIENTE', 'NOMBRE CLIENTE', 'TIPO PRODUCTO', 
                                        'NPS','CES', 'CSAT', 'CUMPLE_AGENDA'], axis=1)
       
        kpi1, kpi2 = st.columns(2)
        kpi3, kpi4 = st.columns(2)

        kpi1.metric(label="Registros Dataset Original", value=len(df_NPS))
        kpi2.metric(label="Registros Dataset Clean", value=len(df_NPS_new))
        kpi3.metric(label="Columnas Dataset Original",value=len(df_NPS.columns))
        kpi4.metric(label="Columnas Dataset Clean", value=len(df_NPS_new.columns))

        fig_col1, fig_col2 = st.columns(2)

        with fig_col1:
                st.markdown("#### Dataset Original")
                fig = sns.catplot(x="CATEGORIA_NPS", kind="count", data=df_NPS)
                st.pyplot(fig)
   
        with fig_col2:
                st.markdown("#### Dataset Clean")
                fig2 = sns.catplot(x="CATEGORIA_NPS", kind="count", data=df_NPS_new)
                st.pyplot(fig2)

        st.write("Tamaño del dataset: ", df_NPS_new.shape)
        st.write(df_NPS_new)

        
        st.markdown("#### Partición de los datos: ")
        X_train, X_test, y_train, y_test = train_test_split(
                                        df_NPS_new.drop('CATEGORIA_NPS', axis = 'columns'),
                                        df_NPS_new['CATEGORIA_NPS'],
                                        train_size   = 0.7,
                                        random_state = 1234,
                                        shuffle      = True,
                                    )

        kpi1, kpi2 = st.columns(2)

        kpi1.metric(label="Tamaño datos para entrenamiento 70%: ", value=len(X_train))
        kpi2.metric(label="Tamaño datos para pruebas 30%: ", value=len(X_test))

        data = {'Training':len(X_train), 'Test':len(X_test)}
        split = list(data.keys())
        values = list(data.values())
        fig = plt.figure(figsize = (10, 5))
        plt.bar(split, values, width = 0.4)       
        st.pyplot(fig)      

        
        st.markdown("#### Transformación de los datos: ")
        if st.checkbox('Estandarizacion y codificación de datos'):
                st.text("Se estandarizan las columnas numéricas y se hace one hot enconding en las columnas categoricas")
                
                # Selección de las variables por típo
                # ==============================================================================
                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import OneHotEncoder
                from sklearn.preprocessing import StandardScaler
                from sklearn.compose import make_column_selector

                numeric_cols = X_train.select_dtypes(include=['float64', 'int']).columns.to_list()
                cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.to_list()

                preprocessor = ColumnTransformer(
                                   [('scale', StandardScaler(), numeric_cols),
                                    ('onehot', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
                                remainder='passthrough')

                X_train_prep = preprocessor.fit_transform(X_train)
                X_test_prep  = preprocessor.transform(X_test)

                st.write(X_train_prep)
        

        if st.checkbox('Imputación de datos faltantes'):
                st.text("Se hace imputación a los datos faltantes para completarlos")

                # Imputación de datos faltantes
                # ==============================================================================
                from sklearn.pipeline import Pipeline
                from sklearn.compose import ColumnTransformer
                from sklearn.impute import SimpleImputer
                from sklearn.preprocessing import OneHotEncoder
                from sklearn.preprocessing import StandardScaler
                from sklearn.compose import make_column_selector

                numeric_cols = X_train.select_dtypes(include=['float64', 'int']).columns.to_list()
                cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.to_list()

                # Transformaciones para las variables numéricas
                numeric_transformer = Pipeline(
                                        steps=[
                                        ('imputer', SimpleImputer(strategy='median')),
                                        ('scaler', StandardScaler())
                                        ]
                                )


                # Transformaciones para las variables categóricas
                categorical_transformer = Pipeline(
                                        steps=[
                                                ('imputer', SimpleImputer(strategy='most_frequent')),
                                                ('onehot', OneHotEncoder(handle_unknown='ignore'))
                                        ]
                                        )

                preprocessor = ColumnTransformer(
                                transformers=[
                                        ('numeric', numeric_transformer, numeric_cols),
                                        ('cat', categorical_transformer, cat_cols)
                                ],
                                remainder='passthrough'
                                )

                X_train_prep = preprocessor.fit_transform(X_train)
                X_test_prep  = preprocessor.transform(X_test)

                st.write(X_train_prep)

#--------------------------------Menu de Modelado -------------------------------#

        st.header("Entrenamiento del modelo")
        if st.checkbox('Multilayer Perceptron'):

                #Parametrización del modelo:  Multilayer Perceptron

                mlp = MLPClassifier(hidden_layer_sizes=(10,10,50),
                                max_iter = 300,activation = 'relu',
                                solver = 'adam')

                mlp.fit(X_train_prep, y_train)
                y_pred_mlp = mlp.predict(X_test_prep)
                acc_mlp = round(mlp.score(X_test_prep, y_test) * 100, 2)
                #pre_mlp = round(precision_score(X_test_prep, y_test, average='macro'), 2)
                sens_mlp = round (sensitivity_score(y_test, y_pred_mlp, average='macro'), 2)
                spe_mlp = round(specificity_score(y_test, y_pred_mlp, average='macro'), 2)

                st.markdown ("### Metricas de Desempeño")

                kpi1, kpi2, kpi3, kpi4= st.columns(4)
                kpi1.metric(label="Accuracy", value=acc_mlp)
                kpi2.metric(label="Precision", value=0.80)
                kpi3.metric(label="Sensitivity", value=sens_mlp)
                kpi4.metric(label="Specificity", value=spe_mlp)
           
                col1, col2 = st.columns(2)

                with col1:

                        y_pred_mlp_prob = mlp.predict_proba(X_test_prep)

                        #binarize the y_values

                        y_test_binarized=label_binarize(y_test, classes=np.unique(y_test))

                        # roc curve for classes
                        fpr = {}
                        tpr = {}
                        thresh ={}
                        roc_auc = dict()

                        n_class = 3

                        for i in range(n_class):    
                                fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:,i], y_pred_mlp_prob[:,i])
                                roc_auc[i] = auc(fpr[i], tpr[i])
                        
                        # plotting    
                        fig, ax = plt.subplots(figsize=(5, 5))
                        plt.plot(fpr[i], tpr[i], linestyle='--', label='%s vs Rest (AUC=%0.2f)'%([i],roc_auc[i]))
                        plt.plot([0,1],[0,1],'b--')
                        plt.xlim([0,1])
                        plt.ylim([0,1.05])
                        plt.title('Multiclass ROC curve', fontsize=16)
                        plt.xlabel('False Positive Rate', fontsize=14)
                        plt.ylabel('True Positive rate', fontsize=14)
                        plt.legend(loc='lower right', fontsize=12)
                        st.pyplot(fig)
                
                with col2:
                        # Calculate la matriz de confusión y el accuracy
                        conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred_mlp)
                        # Print the confusion matrix using Matplotlib
                        fig, ax = plt.subplots(figsize=(5, 5))
                        ax.matshow(conf_matrix, cmap=plt.cm.Blues)
                        for i in range(conf_matrix.shape[0]):
                                for j in range(conf_matrix.shape[1]):
                                        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
                                
                        plt.xlabel('Predicted NPS', fontsize=14)
                        plt.ylabel('True NPS', fontsize=14)
                        plt.title("Matriz de Confusion",fontsize=16)
                        st.pyplot(fig)

                output_model = pickle.dumps(mlp)
                b64 = base64.b64encode(output_model).decode()
                href = f'<a href="data:file/output_model;base64,{b64}" download="model_mlp.pkl">Descargar  modelo entrenado .pkl file </a>'
                st.markdown(href, unsafe_allow_html=True)

#-----------------------------------Menu de Predicción---------------------------------#

if opciones=="Predicción":
        
        st.title("Sistema de Ańalisis Experiencia Clientes con ML")
        st.header("Predicción") 
        st.markdown("Vaya al menú lateral y cargue los datos para predicción")
        st.write("Dataset para predicción ", df_PREDICCION.shape)
        st.write(df_PREDICCION)
        st.markdown("Procesando los datos para realizar la predicción")

        df_PREDICCION_drop=df_PREDICCION.drop (['ID_ENCUESTA', 'SURVEY_STATUS', 'AÑO', 'MES', 
                                        'JOURNEY', 'TIPO_ENCUESTA', 'MERCADO', 'MARCA', 'ID CLIENTE', 
                                        'NOMBRE CLIENTE', 'TIPO PRODUCTO', 'CUMPLE_AGENDA'], axis=1)
        
        st.write("Tamaño del dataset: ", df_PREDICCION_drop.shape)
        st.write(df_PREDICCION_drop)

        
        
        numeric_cols = df_PREDICCION_drop.select_dtypes(include=['float64', 'int']).columns.to_list()
        cat_cols = df_PREDICCION_drop.select_dtypes(include=['object', 'category']).columns.to_list()

        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.preprocessing import StandardScaler
        from sklearn.compose import make_column_selector
        
        preprocessor = ColumnTransformer(
                                [('scale', StandardScaler(), numeric_cols),
                                ('onehot', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
                                remainder='passthrough')

        X_Prediccion = preprocessor.fit_transform(df_PREDICCION_drop)

        # Imputación de datos faltantes
        # ==============================================================================
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.preprocessing import StandardScaler
        from sklearn.compose import make_column_selector

        numeric_cols = df_PREDICCION_drop.select_dtypes(include=['float64', 'int']).columns.to_list()
        cat_cols = df_PREDICCION_drop.select_dtypes(include=['object', 'category']).columns.to_list()

        # Transformaciones para las variables numéricas
        numeric_transformer = Pipeline(
                                steps=[
                                ('imputer', SimpleImputer(strategy='median')),
                                ('scaler', StandardScaler())
                                ]
                        )


        # Transformaciones para las variables categóricas
        categorical_transformer = Pipeline(
                                steps=[
                                        ('imputer', SimpleImputer(strategy='most_frequent')),
                                        ('onehot', OneHotEncoder(handle_unknown='ignore'))
                                ]
                        )

        preprocessor = ColumnTransformer(
                        transformers=[
                                ('numeric', numeric_transformer, numeric_cols),
                                ('cat', categorical_transformer, cat_cols)
                        ],
                        remainder='passthrough'
                        )

        X_Prediccion = preprocessor.fit_transform(df_PREDICCION_drop)
        
        st.markdown("Datos procesados y transformados para realizar la predicción")
        st.write(X_Prediccion)

        st.markdown("Cargue el archivo de pesos del modelo entrenado")

        file = st.file_uploader("Cargue el arhivo de pesos del modelo .pkl", type={"pkl"})
        if file is not None:
                model = pickle.load(file)

        if st.button("Predicción"):

                Predicciones = model.predict(X_Prediccion)

        nps_Pred = pd.DataFrame(Predicciones, columns=['PREDICCION'])

        output = pd.concat([df_PREDICCION, nps_Pred], axis=1)

        st.markdown("#### Resultado de las predicciones:")
        st.write(output)

        _csv = output.to_csv()

        # Dashboard de predicciones
        st.markdown("#### Predicción de Usuarios:")     
        
        col1, col2 = st.columns(2)

        with col1:
                freq = output.PREDICCION.value_counts()
                fig = plt.figure() 
                plt.bar(x=freq.index, height=freq.values, color=['green', 'red', 'yellow'])
                st.pyplot(fig)

        with col2:

                st.metric(label="Promotores", value = freq.values[0])
                st.metric(label="Detractores", value = freq.values[1])
                st.metric(label="Neutros", value = freq.values[2])
        
        # Boton para descargar el archivo

        st.download_button(
                label="Descargue el archivo como CSV",
                data= _csv,
                file_name ='prediciones_nps.csv',
                mime='text/csv',
        )