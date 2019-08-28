# ame
*Teste com dataset para cargo de data scientist na Ame Digital*

O desafio consistia em prever o interesse dos clientes em relação a produtos do tipo panelas em um webcomerce. Um arquivo .csv foi fornecido com diversos exemplos de panelas e suas características.

A estratégia utilizada para a predição solicitada foi implementada utilizando Python 3.6 e as seguintes bibliotecas:

-Pandas
-Numpy
-Scipy
-Matplotlib
-Scikit-Learn

Os dados foram importados para um Pandas dataframe e o processo de normatização dos dados foi feito em diversas etapas. 

Primeiramente, variáveis que deveriam ser do tipo categórica estavam com formato genérico (object) e foram padronizadas e convertidas.

Em seguida, os registros faltantes (NaNs) foram avaliados e estratégias de visualização foram empregadas utlizando-se scatter plots e violinplots.

As variáveis contínuas ALTURA, LARGURA e PROFUNDIDADE tiveram seus valores faltantes imputados com as medianas de maneira hierárquica: primeiro imputei as medianas por MARCA e TIPO_PRODUTO, depois por TIPO_PRODUTO e finalmente pelas medianas de todo o conjunto de dados.

Em seguida, removi os outliers destas três medidas (valores além de 3 sigmas da média) e imputei uma estimativa da CAPACIDADE como ALTURA * LARGURA * PROFUNDIDADE para os valores faltantes.

O próximo passo foi a utilização do classificador k-Nearest Neighbors para imputar os valores faltantes das variáveis categóricas. Isso também foi feito de maneira escalada, primeiramente transformando a variável TIPO_PRODUTO em uma variável dummy (one-hot-encoding) e o k-NN foi então aplicado nos elementos faltantes da variável FORMATO. Os valores preditos foram imputados no lugar dos NaNs.

A variável formato então foi transformada em dummy e usei o kNN para imputar os NaNs de MARCA. Esta também foi transformada em dummy e utilizada em conjunto com as outras variáveis (features) na imputação dos NaNs de COMPOSICAO com kNN. Por fim, COMPOSICAO foi transformada em dummy e adicionada às features existentes, e o kNN previu os labels de COR. Esta última variável foi transformada em dummy então.

Estudei a distribuição de TEMPO_GARANTIA e, após uma pesquisa rápida, notei que é possível uma panela ter 25 anos de garantia, então apenas imputei as medianas nos NaNs dessa feature.

Achei outliers no PESO e os removi, treinando uma regressão logística na sequência para prever os valores faltantes dessa feature. Consegui obter um R^2 de 0.79 para o test set (25%) com um feature set normalizado após expandir o mesmo com algumas operações matemáticas (x^2, x^3, log(x), sqrt(x) e 1-exp(-x)) aplicadas a alguns features numéricos e considerei isso bom, utilizando o modelo treinado para prever os valores faltantes. No entanto, os valores obtidos eram muito grandes em valor absoluto e alguns negativos. Verifiquei que os valores faltantes de PESO eram quase metade dos valores no dataframe e então simplesmente exclui esta coluna.

Um procedimento igual foi realizado para obter os NaNs da variável ITEM_PRICE. Por fim, obtive poucos valores negativos ou além de 3 sigmas. Imputei os valores previstos pelo regressor do ITEM_PRICE nos NaNs e joguei fora os negativos e outliers.

Por fim, chegamos ao fim desse processo sem nenhum valor faltando e com 177358 registros. Descartamos poucos registros, dado que o conjunto original possuía 180275 registros.

Por fim, treinei alguns classificadores neste conjunto de dados com a variável INTERESTED como label. Primeiro usei uma regressão logística e obtive 91% de acerto com regularização L2 (Ridge). Tentei L1 e nenhuma regularização e não obtive melhores resultados. Em seguida utilizei um Support Vector Classifier com kernel linear (Radial não convergiu e demorou uma eternidade...) e obtive problemas de convergência e péssimos resultados. Descartei este classificador. Por fim, utilizei Random Forests com 200 decision trees e obtive resultados marginalmente melhores. A vantagem é que foi muito rápido. Para finalizar, obtive as importâncias de cada feature no algoritmo, mostrando que ITEM_PRICE responde por uma grande parte.
