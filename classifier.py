import os

entrada = "textos"
tipo_teste, texto_teste = [], []
tipo_treino, texto_treino = [], []

# Leitura dos arquivos
for pastas in os.listdir(entrada):
    diretorio = os.listdir(entrada+"\{0}".format(pastas))
    for arquivos in diretorio:
        dir = str(entrada + '\\' + pastas + '\\' + arquivos)
        file = open(str(dir), 'r', encoding='utf-8').read()
        if pastas == 'testes':
            texto_teste.append(file)
            if "carta" in arquivos:
                tipo_teste.append("carta_de_amor")
            elif "receita" in arquivos:
                tipo_teste.append("receita_de_bolo")
        else:
            texto_treino.append(file)
            if "carta" in arquivos:
                tipo_treino.append("carta_de_amor")
            elif "receita" in arquivos:
                tipo_treino.append("receita_de_bolo")

# Prepara Treino
import pandas as pd
df_treino = pd.DataFrame()
df_treino['tipo'] = tipo_treino
df_treino['texto'] = texto_treino

#Carrega palavras neutras
stopWords = list(open("stopWords.txt", 'r').read())

# Vetorização de palavras
from sklearn.feature_extraction.text import CountVectorizer
vetor_modelo = CountVectorizer(analyzer='word', stop_words=stopWords)
vetor_treino = vetor_modelo.fit_transform(df_treino['texto'])

# Importa Naive Bayes
import sklearn.naive_bayes as sk
modelo = sk.MultinomialNB()
modelo.fit(vetor_treino, df_treino['tipo'])

# Prepara teste
df_teste = pd.DataFrame()
df_teste['texto'] = texto_teste
df_teste['tipo'] = tipo_teste

# Transforma teste em vetor
vetor_teste = vetor_modelo.transform(df_teste['texto'])

# Predicao dos testes
predicao = modelo.predict(vetor_teste)

# Avaliacao de resultado
from sklearn import metrics
avalicao = metrics.accuracy_score(predicao, df_teste['tipo'])
print(predicao)
print("Resultado da predição: {0}%".format(int(avalicao*100)))
