#!/usr/bin/env python
# coding: utf-8

# In[35]:

#Packages
import pandas as pd
import math as mt
from scipy.stats import zscore
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
import warnings
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn import preprocessing
from streamlit import components
import os
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import random
import seaborn as sb
from streamlit_plotly_events import plotly_events
import plotly.express as px
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import pairwise_distances


from streamlit.errors import StreamlitAPIException

import myOwnLib as my
import myOwnLib
from sklearn.metrics import pairwise_distances
from umap import UMAP
from yellowbrick.cluster import KElbowVisualizer, kelbow_visualizer

import sys as sys
# st.set_page_config(layout="wide")
st.markdown('<p style="font-size: 48px; font-weight: bold;">Agrupamentos TJSP/ICMC-USP</p>', unsafe_allow_html=True)

df           = ''
ngram        = ''
matriz_cores = ''
feature      = ''
vec_size     = ''
algoritmo    = ''
n_groups     = ''
myPallete    = ''
ns           = ''
myEps        = ''

processo     = ''
kviz         = ''
processado   = False
data         = ''

hide_github_icon = """
<style>
.css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK{ display: none; } #MainMenu{ visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
</style>
"""
st.markdown(hide_github_icon,unsafe_allow_html=True)

st.markdown("""
        <style>
               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-right: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)

#Remove Warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None

# 1. as sidebar menu
with st.sidebar:
    image = "./img/tjsp-1.png"
    st.image(image, use_column_width=True)

    selected = option_menu("TJSP", ["Auto", "Cluster", 'Vizinhos'],
        icons=['house', 'gear'], menu_icon="cast", default_index=1)


    if selected=='Cluster':
        with st.container():
            st.markdown('<h1 style="font-family: Consolas; font-size: 24px;">Configure o algoritmo!</h1>',
                        unsafe_allow_html=True)
        options = [' ', 'K-Means', 'K-Medoids', 'Fuzzy', 'SOM', 'DBSCAN']
        algoritmo = st.selectbox('Algoritmos', options)
        if algoritmo in ['K-Means', 'K-Medoids', 'SOM', 'Fuzzy']:
            with st.container():
                n_groups = st.slider("Número de agrupamentos:", 2, 100, 3)
        elif algoritmo in ['DBSCAN']:
            myEps = st.slider("EPS:", 1, 10, 3)
            myEps = myEps / 10
            ns = st.slider("Amostras:", 0, 40, 10)

        feature_type = [' ', 'Hash', 'Counting', 'TFIDF']
        # feature_type = [' ', 'Hash', 'Counting', 'TFIDF', 'Doc2Vec', 'Word2Vec']
        feature = st.selectbox('Feature', feature_type)

        RD = [' ', 't-SNE', 'ISOMAP', 'UMAP']
        # feature_type = [' ', 'Hash', 'Counting', 'TFIDF', 'Doc2Vec', 'Word2Vec']
        myRD = st.selectbox('Redução Dimensionalidade', RD)

        measure = [' ', 'hamming', 'euclidean', 'cosine']
        # feature_type = [' ', 'Hash', 'Counting', 'TFIDF', 'Doc2Vec', 'Word2Vec']
        myMeasure = st.selectbox('Distância', measure)

        if feature in ['Hash', 'Counting', 'TFIDF']:
            ngram_options = [1, 2, 3, 4, 5, 6]
            ngram = st.multiselect("NGRAMS:", ngram_options)
            vec_size = st.slider("Dimensão do vetor:", 128, 2048, 1024)

        arquivo = st.file_uploader("Selecione um arquivo", type=["csv", "txt", "xlsx"])

        if arquivo is not None:
            # Verificar o tipo de arquivo enviado
            tipo_arquivo = arquivo.name.split('.')[-1]

            # Processar o arquivo enviado
            if tipo_arquivo == 'csv':
                df = pd.read_csv(arquivo)
            elif tipo_arquivo == 'txt':
                df = pd.read_table(arquivo)
            elif tipo_arquivo in ['xls', 'xlsx']:
                df = pd.read_excel(arquivo)
            else:
                st.error(
                    "Tipo de arquivo não suportado. Por favor, selecione um arquivo CSV, TXT ou Excel (XLS ou XLSX).")

        st.markdown("""---""")
    else:
        st.markdown("""---""")
        processo = st.text_input("Processo:", placeholder="Número do Processo")
        k_elements = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        kviz = st.selectbox('K vizinhos', k_elements)
        arquivo = st.file_uploader("Selecione um arquivo", type=["csv", "txt", "xlsx"])

        if arquivo is not None:
            # Verificar o tipo de arquivo enviado
            tipo_arquivo = arquivo.name.split('.')[-1]

            # Processar o arquivo enviado
            if tipo_arquivo == 'csv':
                df = pd.read_csv(arquivo)
            elif tipo_arquivo == 'txt':
                df = pd.read_table(arquivo)
            elif tipo_arquivo in ['xls', 'xlsx']:
                df = pd.read_excel(arquivo)
            else:
                st.error(
                    "Tipo de arquivo não suportado. Por favor, selecione um arquivo CSV, TXT ou Excel (XLS ou XLSX).")

        st.markdown("""---""")

if selected=='Cluster':
    try:
        st.set_option('deprecation.showPyplotGlobalUse', False)
        # st.write('Processos a serem agrupados:')
        setup = pd.DataFrame(
            [
                {"Algoritmo": algoritmo, "N Clusters": n_groups, "Feature": feature, "Dimensão": vec_size, "Redução Dim": myRD, "Distância": myMeasure},
            ]
        )
        st.table(setup)
        if st.button('Processar'):
            # for batch in tqdm(data_loader, desc='Training Progress'):
            labels, features, fullFeatures = my.processCluster(df, algoritmo, n_groups, feature, vec_size, [min(ngram), max(ngram)], myRD, myMeasure)
            print(str(len(features)))
            print(type(labels))
            processado = True
            if algoritmo=='SOM':
                rotulos = [0] * len(features)
                lab = 0
                for x in labels:
                    for l in x:
                        # print(l)
                        rotulos[l] = lab
                    lab = lab + 1
                labels = np.array(rotulos)
            print(labels)
            data = {
                'X': features[:,0],
                'Y': features[:,1],
                'label' : labels.astype(str),
                'processo' : df['numero_processo']
            }
            st.markdown('<hr>', unsafe_allow_html=True)
            # sb.scatterplot(data, x='X', y='Y', hue='label', palette=myPallete)
            # st.pyplot()

            fig = px.scatter(data, x='X', y='Y', color='label',  hover_data=['processo'], title='Agrupamentos '+ algoritmo)
            st.plotly_chart(fig)

            # selected_points = plotly_events(fig)

            # print(selected_points)
            st.markdown('<hr>', unsafe_allow_html=True)
            myDF = df[['numero_processo', 'formatado']]
            myDF['label'] =labels
            # myDF['status']=[True]*myDF.shape[0]

            st.dataframe(myDF)


            def convert_df_to_csv(df):
                csv = df.to_csv(index=False)
                return csv


            # Convertendo o DataFrame para CSV
            csv = convert_df_to_csv(myDF)

            # Criando um botão para baixar o arquivo CSV
            st.download_button(
                label="Baixar dados como CSV",
                data=csv,
                file_name='cluster_'+algoritmo+'_'+str(n_groups)+'_'+feature+'.csv',
                mime='text/csv',
            )

    except ValueError as e:
        st.write(e)
    except StreamlitAPIException as e:
        st.write(e)


if selected=='Vizinhos':
    try:
        if st.button('Processar'):
            # for batch in tqdm(data_loader, desc='Training Progress'):
            labels, features, fullFeatures = my.processCluster(df)
            data = {
                'X': features[:,0],
                'Y': features[:,1],
                'label' : labels.astype(str),
                # 'label' : labels,
                'processo' : df['numero_processo']
            }
            matriz_distancias = pairwise_distances(features)
            import numpy as np



            data = pd.DataFrame(data)
            idxTarget = data.loc[data.processo == str(processo)].index
            distancias = np.linalg.norm(matriz_distancias - matriz_distancias[idxTarget, :], axis=1)

            indices_k_mais_proximos = np.argsort(distancias)[:kviz]

            data2 = data.iloc[indices_k_mais_proximos]
            data2.loc[data2.processo == str(processo), 'label'] = 'target'

            fig = px.scatter(data2, x='X', y='Y', color='label', hover_data=['processo'],
                             color_discrete_sequence=px.colors.qualitative.Set1,
                             title='K vizinhos próximos do Processo ' + str(processo))
            fig.update_traces(marker_size=10)
            st.plotly_chart(fig)

    except ValueError as e:
        st.write(e)
    except StreamlitAPIException as e:
        st.write(e)


if selected=='Auto':
    try:
        if st.button('Processar'):
            tfidf = myOwnLib.getFeature('TFIDF', 2048, (2,3))
            vecFeat=tfidf.fit_transform(df['formatado'])
            um = UMAP(n_neighbors=15, n_components=2, metric='hamming', min_dist=0.1, random_state=42)
            vecFeatRd = um.fit_transform(vecFeat)

            knee, wcss = myOwnLib.getKnee(vecFeatRd)
            plt.figure(figsize=(10, 6))
            sb.lineplot(x=range(3, 30), y=wcss, marker='o')
            plt.axvline(x=knee, color='r', linestyle='--', label=f'Ponto de Cotovelo: {knee}')
            plt.title('Método do Cotovelo')
            plt.xlabel('Número de Clusters')
            plt.ylabel('WCSS')
            plt.legend()
            st.pyplot(plt)

            labels, features, fullFeatures = my.processCluster(df, 'K-Means', knee, 'TFIDF', 2048, (2,3))
            data = {
                'X': features[:,0],
                'Y': features[:,1],
                'label' : labels.astype(str),
                'processo' : df['numero_processo']
            }
            st.markdown('<hr>', unsafe_allow_html=True)
            plt.figure(figsize=(10, 8))
            sb.scatterplot(data, x='X', y='Y', hue='label')
            plt.xlabel('DIM 1')  # Alterar rótulo do eixo x
            plt.ylabel('DIM 2')  # Alterar rótulo do eixo y
            plt.title(f"Clusters - {algoritmo} - UMAP(hamming))")
            st.pyplot(plt)

            st.markdown('<hr>', unsafe_allow_html=True)
            myDF = df[['numero_processo', 'formatado']]
            myDF['label'] =labels
            # myDF['status']=[True]*myDF.shape[0]

            st.dataframe(myDF)




    except ValueError as e:
        st.write(e)
    except StreamlitAPIException as e:
        st.write(e)
