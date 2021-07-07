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
from statistics import mean
from sklearn.model_selection import cross_validate
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing




#extraindo a planilha e a normalizando

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
'''
with open('R_L_N.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["L_color_image_MEAN","L_exp_image_MEAN","L_blur_image_MEAN",
    "L_color_image_MAX","L_exp_image_MAX","L_blur_image_MAX","L_color_image_MIN","L_exp_image_MIN",
    "L_blur_image_MIN","L_color_image_SD","L_exp_image_SD","L_blur_image_SD","L_color_face_MEAN",
    "L_exp_face_MEAN","L_blur_face_MEAN","L_color_face_MAX","L_exp_face_MAX","L_blur_face_MAX",
    "L_color_face_MIN","L_exp_face_MIN","L_blur_face_MIN","L_color_face_SD","L_exp_face_SD",
    "L_blur_face_SD","label"])
    writer.writerows(dados)



#Resgatando as features e a variável alvo

reg = pd.read_csv('R_L_N.csv')

features = reg.iloc[:, 0:24]

target = reg.iloc[: , -1]


#Regressão logística

'''
modelo = smf.glm(formula='label ~ L_blur_image_MEAN+L_blur_image_MAX+L_blur_image_MIN+L_blur_image_SD+L_blur_face_MEAN+L_blur_face_MAX+L_blur_face_MIN+L_blur_face_SD', 
  data=reg, family = sm.families.Binomial()).fit()
print(modelo.summary())
'''


#Cross-validate com vários modelos

modelo1 = RidgeClassifier()
modelo2 = LogisticRegression()
modelo3 = Perceptron()
modelo4 = DecisionTreeClassifier()
modelo5 = KNeighborsClassifier()
modelo6 = svm.SVC()
modelo7 = RandomForestClassifier()
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestClassifier())

#---------------------feature selection-----------------------
'''
modelo7.fit(features, target)
print("features importances do RandomForestClassifier:")
print(modelo7.feature_importances_)
'''

features = SelectKBest(chi2, k=6).fit_transform(features, target)

modelo7.fit(features, target)
print("features importances do RandomForestClassifier depois do SelectKBest:")
print(modelo7.feature_importances_)

#-------------------------------------------------------------

modelos = {1: RidgeClassifier(), 2: LogisticRegression(),
3: Perceptron(), 4: DecisionTreeClassifier(), 5:KNeighborsClassifier(),
6:svm.SVC(), 7: RandomForestClassifier(), 8: pipeline}

for index in modelos:
  modelo = modelos[index]
  resultados = cross_validate(modelo, features, target, return_train_score=False,
               scoring=['accuracy',
                        'average_precision',
                        'f1',
                        'precision',
                        'recall',
                        'roc_auc'])
  print(modelo)
  print("Acuracia")
  print(mean(resultados['test_accuracy']))
  print("F1_score")
  print(mean(resultados['test_f1']))
  print("Precisao")
  print(mean(resultados['test_precision']))
  print("Recall")
  print(mean(resultados['test_recall']))
  print("ROC_AUC")
  print(mean(resultados['test_roc_auc']))

'''
testes = round(modelo.predict(reg))

m_c = confusion_matrix(target, testes)
print(m_c)

df_cm = pd.DataFrame(m_c, index = [i for i in "PN"],
                  columns = [i for i in "PN"])
plt.figure(figsize = (7,6))
sn.heatmap(df_cm, annot=True)
plt.show()
'''




# L_color_image_MEAN+L_exp_image_MEAN+L_blur_image_MEAN+L_color_image_MAX+L_exp_image_MAX+L_blur_image_MAX+L_color_image_MIN+L_exp_image_MIN+L_blur_image_MIN+L_color_image_SD+L_exp_image_SD+L_blur_image_SD+