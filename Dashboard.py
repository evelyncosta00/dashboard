import plotly.subplots as sp
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as ticker 
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.axis import Axis 
from sqlalchemy import create_engine
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import requests 
import plotly.express as px

st.title('Prevendo valor barril de Petróleo Brent :moneybag:')
st.write ('O objetivo deste projeto é analisar os dados históricos de preços do petróleo Brent do IPEA, com o intuito de desenvolver insights relevantes e construir um modelo de Machine Learning para prever os preços futuros do petróleo.')
st.divider()
st.write ('Para aprimorar nosso modelo, buscamos entender melhor os fatores que influenciam sua variação. Inicialmente, analisamos uma base de dados contendo informações históricas de preços do petróleo ao longo do tempo.')

df_preco_petro_1 = pd.read_csv(r'https://raw.githubusercontent.com/evelyncosta00/prevendo_valor_barril_petroleo/main/valor_barril_silver.csv')
df_preco_petro_1.drop(columns ='Unnamed: 0', inplace=True)
df_preco_petro_1['Preco'] = df_preco_petro_1['Preco'].apply(lambda x: f"${x:.2f}")
df_preco_petro_1['Data'] =  pd.to_datetime(df_preco_petro_1['Data'], format='%Y-%m-%d') 
df_preco_petro_1['Data'] = df_preco_petro_1['Data'].dt.strftime('%d/%m/%Y')  
df_preco_petro_1 = df_preco_petro_1.head(10)

st.write ('Base de dados com os preços do barril de petróleo Brent ao longo do tempo:')
#tabela1
st.table(df_preco_petro_1)

st.title('Análise Exploratória dos dados (EDA)')

st.divider()

# Grafico de janela

# Carregando os dados
df_preco_petro = pd.read_csv('https://raw.githubusercontent.com/evelyncosta00/prevendo_valor_barril_petroleo/main/valor_barril_silver.csv')
df_preco_petro['Data'] = pd.to_datetime(df_preco_petro['Data'], format='%Y-%m-%d')

# Definindo as janelas de tempo
windows = ['3M', '6M', '12M', '5Y']

fig, axs = plt.subplots(nrows=len(windows)//2, ncols=2, figsize=(10, 10))

for i, window in enumerate(windows):
    # Filtrando os dados para a janela de tempo atual
    end_date = df_preco_petro['Data'].max()
    start_date = end_date - pd.DateOffset(months=int(window[:-1])) if window[-1] == 'M' else end_date - pd.DateOffset(years=int(window[:-1]))
    filtered_df = df_preco_petro[(df_preco_petro['Data'] >= start_date) & (df_preco_petro['Data'] <= end_date)]

    # Plotando os dados filtrados
    axs[i//2, i%2].plot(filtered_df['Data'], filtered_df['Preco'], color='purple')
    axs[i//2, i%2].set_title(f'Preço do Brent ao longo do tempo ({window})')
    axs[i//2, i%2].set_xlabel('Data')
    axs[i//2, i%2].set_ylabel('Preço')

    # Formatando o eixo y para 'YYYYMM'
    date_format = mdates.DateFormatter('%Y%m')
    axs[i//2, i%2].xaxis.set_major_formatter(date_format)

plt.tight_layout()

# Adicionando o gráfico ao Streamlit
st.pyplot(fig)

st.write ('O gráfico acima mostra a evolução dos preços do petróleo Brent ao longo do tempo, considerando diferentes janelas de tempo. Podemos observar que o preço do petróleo apresenta flutuações significativas, com períodos de alta e baixa ao longo dos anos.')

st.divider()


#with coluna2:

st.title ('Tendência, Sazonalidade e Resíduos')

#st.image(r'C:\Users\ofici\OneDrive\Documentos\Study\Data_Science_Projects\Studies\Portifolio\dashboard\freq.png',caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
#grafico de tendências 

# Criar uma cópia do DataFrame
df_copy = df_preco_petro.copy()

# Certifique-se de que a cópia do DataFrame está indexada por uma coluna de data
df_copy.index = pd.to_datetime(df_copy['Data'])

# Resample para frequência diária, use ffill() para preencher quaisquer lacunas
df_copy = df_copy.resample('D').ffill()

# Decomposição sazonal
decomposition = seasonal_decompose(df_copy['Preco'], model='additive')

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

fig, axs = plt.subplots(4, 1, figsize=(12,8))
axs[0].plot(df_copy['Preco'], label='Original', color='purple')
axs[0].legend(loc='best')
axs[1].plot(trend, label='Tendência', color='purple')
axs[1].legend(loc='best')
axs[2].plot(seasonal,label='Sazonalidade', color='purple')
axs[2].legend(loc='best')
axs[3].plot(residual, label='Resíduos', color='purple')
axs[3].legend(loc='best')
plt.tight_layout()

# Adicionando o gráfico ao Streamlit
st.pyplot(fig)
st.divider()

st.write ('No entanto, reconhecendo a complexidade do mercado de petróleo e a interconexão com outros fatores econômicos globais, expandimos nossa análise para incluir variáveis adicionais que podem impactar o preço do petróleo, escorando a busca principalmente em oferta e demanda. \n\n Assim, identificamos que indicadores econômicos, como o crescimento do PIB global, tem uma influência substancial no comportamento de longo prazo dos preços do petróleo.')

st.write ('Na busca por fontes complementares para enriquecer nosso projeto, identificamos os seguintes fatores que exercem influência nos preços do petróleo Brent:')

st.write ('1.Oferta e Demanda: As flutuações na oferta global de petróleo, determinadas pelas decisões de produção da OPEP (Organização dos Países Exportadores de Petróleo) e eventos geopolíticos, desempenham um papel crucial nos preços do petróleo. A dinâmica entre oferta e demanda é essencial para compreender as variações nos valores do petróleo Brent.')

st.write ('2.Economia Global: O crescimento econômico global exerce uma influência significativa na demanda por petróleo. Durante períodos de expansão econômica robusta, a procura por petróleo tende a aumentar, impulsionando os preços para cima. A análise dos indicadores econômicos globais é fundamental para prever tendências futuras nos preços do petróleo.')

st.write ('3.Estoques de Petróleo: Os níveis de estoques de petróleo, tanto nos países produtores quanto nos consumidores, desempenham um papel fundamental na determinação dos preços do petróleo. Grandes aumentos nos estoques podem exercer pressão para baixo nos preços, enquanto grandes reduções podem impulsioná-los para cima. O monitoramento contínuo dos estoques é essencial para compreender a dinâmica do mercado petrolífero.')

st.write ('4.Política da OPEP: As decisões da OPEP e de seus membros sobre cotas de produção têm um impacto direto nos preços do petróleo. Cortes na produção podem resultar em aumentos nos preços, enquanto aumentos na produção podem causar quedas nos preços. A inclusão de bases de dados da OPEP enriquece nossa análise, permitindo uma compreensão mais profunda das influências políticas sobre o mercado de petróleo.')

st.write ('Os dados da OPEP e indicadores globais foram extraídos de: https://asb.opec.org/data/ASB_Data.php')

st.divider()

st.title('Conectando features')

st.write ('Após identificar as bases de dados consideradas potencialmente influentes, iniciamos o tratamento de cada uma utilizando tanto Python quanto MySQL. Isso incluiu etapas como limpeza, transformação e integração dos dados. Ao finalizar essas etapas, combinamos todas as bases através de um join, resultando no DataFrame que você vê abaixo:')

df_base_final = pd.read_csv(r'https://raw.githubusercontent.com/evelyncosta00/prevendo_valor_barril_petroleo/main/base_final_completa.csv')
df_base_final = df_base_final.head(10)

st.write ('Base de dados com as features selecionadas:')

st.table(df_base_final)

st.divider()

st.title('Matriz de correlação')
st.write ('Esta é a matriz de correlação das variáveis que foram selecionadas, tratadas e transformadas em novas variáveis. Ela evidencia as interações entre todas as variáveis, proporcionando uma representação estatística das associações entre os diferentes atributos. Essa visualização é fundamental para identificar padrões e insights cruciais no conjunto de dados, contribuindo para a compreensão das relações entre as variáveis e facilitando o processo de análise e tomada de decisão.')
#st.image(r'C:\Users\ofici\OneDrive\Documentos\Study\Data_Science_Projects\Studies\Portifolio\dashboard\st.image().png',caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
import seaborn as sns
import matplotlib.pyplot as plt

# numeric_df = df.select_dtypes(include='number')

df_corr = df_base_final[[
'PRECO'
,'MIN_producao_petroliferos'
,'MIN_producao_mundial_gas_natural'
,'MIN_demanda_petroleo'
,'AVG_prod_petroleo_opep'
,'AVG_pib_growth'
,'MIN_demanda_gas_natural'
,'DESV_producao_petroleo'
,'AVG_producao_petroleo'
,'MIN_producao_petroleo'
,'MAX_demanda_petroleo'
,'AVG_estoques_petroleo'
,'MAX_producao_petroleo'
,'DESV_estoques_petroleo'
,'AVG_exportacoes_petroleo'
,'MAX_estoques_petroleo'
,'MIN_exportacoes_gas_natural'
,'AVG_importacoes_gas_natural'
,'MAX_exportacoes_gas_natural'
,'AVG_demanda_gas_natural'
,'DESV_importacoes_gas_natural']]
corr = df_corr.corr()

# Create a new figure and set its size
plt.figure(figsize = (20, 10))

# Generate a heatmap in seaborn
sns.heatmap(corr, cmap ='Purples',annot=True)

# Use Streamlit's pyplot() function to display the figure
st.pyplot(plt)

st.divider()
st.title('Prevendo valor do Barril de Petróleo Brent')

st.write('Realizamos testes com algoritmos de boosting para determinar a performance ótima com base nas variáveis disponíveis. Os algoritmos examinados foram LightGBM e Xgboost, contudo, o modelo que apresentou o melhor desempenho foi o MLPRegressor, uma rede neural artificial com múltiplas camadas.')

##############################################################################################################

st.write('Os dados foram divididos em conjuntos de treinamento (80%) e teste (20%). Ao rodar o modelo de redes neurais, foram definidos parâmetros específicos que desempenham um papel crucial em seu desempenho. Para o MLPRegressor, os seguintes parâmetros foram selecionados: hidden_layer_sizes=(32, 16), activation="relu", solver="adam" e max_iter=500. Estes parâmetros influenciam diretamente na arquitetura da rede neural, na função de ativação das camadas ocultas, no algoritmo utilizado para a otimização dos pesos e no número máximo de iterações durante o treinamento.')

df = pd.read_csv(r'https://raw.githubusercontent.com/evelyncosta00/prevendo_valor_barril_petroleo/main/base_final_completa.csv')

df = df.sort_values('ANO')

# Definir a variável target e as variáveis explicativas
target = 'PRECO'
features = df.columns.drop([target, 'ANO', 'PAIS'])

# Separar as variáveis explicativas e a variável target
X = df[features]
y = df[target]

# Preencher os valores NaN com a média da coluna
numeric_features = X.select_dtypes(include=[np.number]).columns
X[numeric_features] = X[numeric_features].fillna(X[numeric_features].mean())

# One-hot encode categorical variables
X = pd.get_dummies(X)

# Criar conjuntos de treinamento, teste e validação
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Guardar os índices
train_index = X_train.index
val_index = X_val.index
test_index = X_test.index

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Criar o modelo de rede neural
model = MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu', solver='adam', max_iter=500)

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões no conjunto de treinamento, validação e teste
predictions_train = model.predict(X_train)
predictions_val = model.predict(X_val)
predictions_test = model.predict(X_test)

# Adicionar as previsões de volta ao DataFrame original
df.loc[train_index, 'Previsoes'] = predictions_train
df.loc[val_index, 'Previsoes'] = predictions_val
df.loc[test_index, 'Previsoes'] = predictions_test

# Calcular as métricas de erro para o modelo principal
mse_train = mean_squared_error(y_train, predictions_train)
mse_val = mean_squared_error(y_val, predictions_val)
mse_test = mean_squared_error(y_test, predictions_test)

rmse_train = np.sqrt(mse_train)
rmse_val = np.sqrt(mse_val)
rmse_test = np.sqrt(mse_test)

r2_train = r2_score(y_train, predictions_train)
r2_val = r2_score(y_val, predictions_val)
r2_test = r2_score(y_test, predictions_test)

#print(f'Train: MSE = {mse_train}, RMSE = {rmse_train}, R^2 = {r2_train}')
#print(f'Test: MSE = {mse_test}, RMSE = {rmse_test}, R^2 = {r2_test}')
#print(f'Validation: MSE = {mse_val}, RMSE = {rmse_val}, R^2 = {r2_val}')

# Normalizar todos os dados para a validação cruzada
X_scaled = scaler.transform(X)

# Executar a validação cruzada
scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')

# Calcular as métricas de erro para a validação cruzada
mse_cv = -scores.mean()
rmse_cv = np.sqrt(mse_cv)
r2_cv = r2_score(y, model.predict(X_scaled))

# Crie um dicionário com os resultados
results = {
    'MSE': [mse_train, mse_test, mse_val, mse_cv],
    'RMSE': [rmse_train, rmse_test, rmse_val, rmse_cv],
    'R^2': [r2_train, r2_test, r2_val, r2_cv]
}

# Crie um DataFrame a partir do dicionário
df_results = pd.DataFrame(results, index=['Train', 'Test', 'Validation', 'Cross Validation'])


# Converter o DataFrame para HTML e centralizá-lo
df_html = df_results.to_html(index=False).replace('<table border="1" class="dataframe">','<table style="margin-left:auto;margin-right:auto;">')

# Exibir o DataFrame no Streamlit
st.write(df_html, unsafe_allow_html=True)

st.write('As métricas obtidas revelam um desempenho excepcional. O coeficiente de determinação (R²) próximo de 1 indica que o modelo explica a variabilidade dos dados de forma muito eficaz. Um R² de 0.99 significa que aproximadamente 99% da variabilidade nos dados é explicada pelo modelo, o que demonstra sua capacidade de diferenciar e capturar os padrões presentes nos dados de forma altamente precisa.')
st.write('Além disso, o erro médio quadrático (MSE) e o erro quadrático médio da raiz (RMSE) apresentam valores bastante reduzidos, refletindo a precisão das previsões do modelo. Um MSE próximo de zero e um RMSE próximo de zero indicam que as previsões do modelo estão muito próximas dos valores reais, evidenciando sua capacidade de realizar previsões com alta precisão.')
st.write('Essas métricas combinadas fornecem uma validação robusta da capacidade preditiva do MLPRegressor, destacando sua eficácia em modelar os padrões complexos presentes nos dados.')

######################################

# Realizar a Permutation Importance
results = permutation_importance(model, X_test, y_test, scoring='neg_mean_squared_error')

# Obter a importância das features
importances = results.importances_mean

# Ordenar os índices das features pela importância
indices = np.argsort(importances)[-20:]  # Pegar apenas os 20 maiores

# Criar um gráfico de barras com Plotly
fig = go.Figure(data=[
    go.Bar(y=[features[i] for i in indices], x=importances[indices], orientation='h', marker_color='purple')
])

# Definir o título do gráfico e os rótulos dos eixos
fig.update_layout(title='Feature Importances', xaxis_title='Relative Importance', height=800)  # Set height

# Mostrar o gráfico no Streamlit
st.plotly_chart(fig)

#######################################################

# Criar subplots
fig = sp.make_subplots(rows=3, cols=1)

# Adicionar os valores reais ao gráfico
fig.add_trace(go.Scatter(x=df['ANO'], y=df['PRECO'], mode='lines', name='Valores Reais'), row=1, col=1)

# Adicionar os valores previstos ao gráfico
fig.add_trace(go.Scatter(x=df['ANO'], y=df['Previsoes'], mode='lines', name='Valores Preditos'), row=1, col=1)

# Adicionar subplot para 'PRECO'
fig.add_trace(go.Scatter(x=df['ANO'], y=df['PRECO'], mode='lines+markers', name='Valores Reais'), row=2, col=1)

# Adicionar subplot para 'Previsoes'
fig.add_trace(go.Scatter(x=df['ANO'], y=df['Previsoes'], mode='lines+markers', name='Valores Preditos'), row=3, col=1)

# Definir o título do gráfico e os rótulos dos eixos
fig.update_layout(height=800, title='Valores Reais vs Valores Preditos')#, xaxis_title='Ano', yaxis_title='Preço')

# Mostrar o gráfico no Streamlit
st.plotly_chart(fig)