import csv
import numpy as np
import pandas as pd
from scipy.sparse import data
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import seaborn as sn
from sklearn.model_selection import cross_validate

dados = np.genfromtxt("C:\\Users\\guilh\\Downloads\\framesResumidos.csv", delimiter=",",skip_header=1, encoding = "UTF-8-sig")

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
    writer.writerow(["L_color_image_MEAN","L_exp_image_MEAN","L_blur_image_MEAN",
    "L_color_image_MAX","L_exp_image_MAX","L_blur_image_MAX","L_color_image_MIN","L_exp_image_MIN",
    "L_blur_image_MIN","L_color_image_SD","L_exp_image_SD","L_blur_image_SD","L_color_face_MEAN",
    "L_exp_face_MEAN","L_blur_face_MEAN","L_color_face_MAX","L_exp_face_MAX","L_blur_face_MAX",
    "L_color_face_MIN","L_exp_face_MIN","L_blur_face_MIN","L_color_face_SD","L_exp_face_SD",
    "L_blur_face_SD","label"])
    writer.writerows(dados)


reg = pd.read_csv("R_L_N.csv")

features = reg.iloc[:, 12:24]

print(features)
