import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
movies_df = pd.read_csv('product.csv')
ratings_df = pd.read_csv('ratings.csv')

movies_df = movies_df.drop('category',1)

userInput = [
{'title':'Alicates', 'rating':5},
{'title':'Cutters', 'rating':4.5},
{'title':'alicates', 'rating':4.5},
{'title':'martillo', 'rating':4.5},
{'title':'llaves', 'rating':3.5}
]
inputMovies = pd.DataFrame(userInput)


inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#Luego fusionándolo para que podamos obtener el productId. Lo está fusionando implícitamente por título.
inputMovies = pd.merge(inputId, inputMovies)
#Eliminar información que no usaremos del dataframe

#Final input dataframe
inputMovies
#Filtrar a los usuarios que han visto películas que la entrada ha visto y almacenarlas
userSubset = ratings_df[ratings_df['productId'].isin(inputMovies['productId'].tolist())]
userSubset.head()
# Groupby crea varios sub dataframes de datos donde todos tienen el mismo valor en la columna especificada #como parámetro
userSubsetGroup = userSubset.groupby(['userId'])
# Ordenamos por los usuarios que han puntuado mas a las peliculas en común con la entrada
userSubsetGroup = sorted(userSubsetGroup, key=lambda x: len(x[1]), reverse=True)
userSubsetGroup = userSubsetGroup[0:100]

pearsonCorrelationDict = {}
#Para cada grupo de usuarios de nuestro subconjunto
for name, group in userSubsetGroup:
    #Comencemos ordenando la entrada y el grupo de usuarios actual para que los valores no se mezclen más adelante.
    group = group.sort_values(by='productId')
    inputMovies = inputMovies.sort_values(by='productId')
    #Obtenga las puntuaciones de las reseñas de las películas que ambos tienen en común
    temp_df = inputMovies[inputMovies['productId'].isin(group['productId'].tolist())]
    #Y luego guárdelos en una variable de búfer temporal en un formato de lista para facilitar cálculos futuros
    tempRatingList = temp_df['rating'].tolist()
    #Pongamos también las reseñas del grupo de usuarios actual en un formato de lista
    tempGroupList = group['rating'].tolist()
    data_corr = {'tempGroupList': tempGroupList,
            'tempRatingList': tempRatingList}
    pd_corr = pd.DataFrame(data_corr)
    r = pd_corr.corr(method="pearson")["tempRatingList"]["tempGroupList"]
    #ahora eliminamos los nan de nuestro coef de pearson
    if math.isnan(r) == True:
        r = 0
    pearsonCorrelationDict[name] = r
#Convertimos el diccionario a un dataframe:
pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
pearsonDF.head()

topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
topUsers.head()

topUsersRating=topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
topUsersRating.head()

topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
topUsersRating.head()

tempTopUsersRating = topUsersRating.groupby('productId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
tempTopUsersRating.head()

#Crea un nuevo dataframe
recommendation_df = pd.DataFrame()
#Ahora tomamos el promedio ponderado
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['productId'] = tempTopUsersRating.index
recommendation_df.head()

recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
recommendation_df.head(30)
print(movies_df.loc[movies_df['productId'].isin(recommendation_df.head(30)['productId'].tolist())])