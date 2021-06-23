import csv
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import seaborn as sn

dados = np.genfromtxt("R_L.csv", delimiter=",",skip_header=1, encoding = "UTF-8-sig")

def normaliza(lista):
  listaT = []
  i = 0
  if (max(lista) - min(lista) == 0):
    return lista
  while i < (len(lista)):
    x = lista[i]
    y = (x - min(lista)) / (max(lista) - min(lista))
    listaT.append(y)
    i = i + 1  
  return listaT

'''
i = 0
j = 0
normalizado = []
while j < 1:
  trend = []
  for video in dados:
    if (i == video.size - 1):
      j = 1
    trend.append(video[i])
  normalizado = normaliza(trend)
  k = 0
  for video in dados:
    video[i] = normalizado[k]
    k = k + 1
  i = i + 1

with open('R_L_N.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["L_color_image", "L_exp_image", "L_blur_image", "L_color_face", "L_exp_face", "L_blur_face", "label"])
    writer.writerows(dados)

'''
reg = pd.read_csv('R_L_N.csv')

teste = pd.read_csv('R_L_N_Teste.csv')

last_column = teste.iloc[: , -1]

#modelo = smf.glm(formula='label ~ L_color_image + L_exp_image + L_blur_image + L_color_face	+ L_exp_face	+ L_blur_face',
modelo = smf.glm(formula='label ~ L_blur_image + L_blur_face', 
  data=reg, family = sm.families.Binomial()).fit()
print(modelo.summary())

testes = round(modelo.predict(teste))

m_c = confusion_matrix(last_column, testes)
print(m_c)

df_cm = pd.DataFrame(m_c, index = [i for i in "PN"],
                  columns = [i for i in "PN"])
plt.figure(figsize = (7,6))
sn.heatmap(df_cm, annot=True)
plt.show()

