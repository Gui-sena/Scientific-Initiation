import face_recognition
from PIL import Image
import dlib
import math
import statistics
from pathlib import Path
from threading import Thread
from pathlib import Path
import pickle
import numpy as pinp
from abc import ABC, abstractmethod
import cv2
import pickle
import skvideo.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
import torchvision.transforms as transforms
import sys
import csv


def drawLine(imagem, newLeft, newRight, newBottom, newTop):
  imagem = cv2.line(imagem, (newLeft, newBottom), (newLeft, newTop), (255, 0, 0), 9)
  imagem = cv2.line(imagem, (newRight, newBottom), (newRight, newTop), (255, 0, 0), 9)
  imagem = cv2.line(imagem, (newLeft, newTop), (newRight, newTop), (255, 0, 0), 9)
  imagem = cv2.line(imagem, (newRight, newBottom), (newLeft, newBottom), (255, 0, 0), 9)

#def variance_of_laplacian(image):
 # return cv2.Laplacian(image, cv2.CV_64F).var()

#def detectBlur(image):
 # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  #fm = variance_of_laplacian(gray)
  #height, width = image.shape[:2]
  #res = (fm*height)/10
  #return res

def cortaImagem(imagem, left, right, bottom, top):
  im = imagem[top:bottom, left:right]
  return im

def getVideoDuration(video):
  fps = vidcap.get(cv2.CAP_PROP_FPS)      # Retorna duração do vídeo
  frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
  duration = frame_count/fps
  duration = int(duration)
  return duration

def ordenaListaPath(lista):
  i = 0
  j = i + 1
  while i < len(lista):
    while j < len(lista):
      if (lista[i].name > lista[j].name):
        aux = lista[j]
        lista[j] = lista[i]
        lista[i] = aux
      j = j + 1
    i = i + 1
    j = i + 1
  return lista

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


def get1stPart(lista):
  listaT = []
  i = 0
  while i < (len(lista)/2):
    x = lista[i]
    listaT.append(x)
    i = i + 1
  return listaT

def get2ndPart(lista):
  listaT = []
  i = round(len(lista)/2)
  while i < len(lista):
    x = lista[i]
    listaT.append(x)
    i = i + 1
  return listaT

def getListaMedia(listaDeListas):
  i = 0
  total = 0
  count = 0
  listaMedia = []
  while i < len(listaDeListas[0]):
    for lista in listaDeListas:
      if (i < len(lista)):
        total = lista[i] + total
        count = count + 1
    media = total/count
    listaMedia.append(media)
    i = i + 1
  return listaMedia

def transfDados(entrada):
   rsc = 0   
   for item in entrada:
    rsc += item # rsc + temporario[i]    
    rsc = rsc/len(entrada)
    rsc = rsc/1000
   return rsc
   
def image_to_cuda_tensor(rgb_im):
    return transforms.ToTensor()(rgb_im).unsqueeze_(0).cuda()



# Método que detecta o blur 

def variance_of_laplacian(image):
  return cv2.Laplacian(image, cv2.CV_64F).var()

def detectBlur(image):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  fm = variance_of_laplacian(gray)
  height, width = image.shape[:2]
  res = (fm*height)/10
  return res


# L_color demonstra o quanto as cores nos canais RGB se diferenciam de cinza

def getL_color(image):
        image = image_to_cuda_tensor(image)
        mean_rgb = torch.mean(image,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        k = k.item()
        return k


# L_exp mede o nível de exposisão de uma área de 16 pixels comparada à um valor predefinido que representa 
# uma boa exposição

def getL_exp(image): #0.1851447923832761
        mean_val = 0.1851447923832761
        pool = nn.AvgPool2d(16)
        image = image_to_cuda_tensor(image)
        x = torch.mean(image,1,keepdim=True)
        mean = pool(image)
        d = torch.mean(torch.pow(mean- torch.FloatTensor([mean_val] ).cuda(),2))
        d = d.item()
        return d





p = []
pat = []
p = Path('/home/lapis/Videos/Face_Forensics/Verdadeiros_Treino/c0')    
verdadeiros_treino = list(p.glob('**/0*.mp4'))
verdadeiros_treino = ordenaListaPath(verdadeiros_treino)
verdadeiros01p1 = get1stPart(verdadeiros_treino)
verdadeiros01p2 = get2ndPart(verdadeiros_treino)
pat = Path('/home/lapis/Videos/Face_Forensics/Falsos_Treino/c0')
falsos_treino = list(pat.glob('**/0*.mp4'))
falsos_treino = ordenaListaPath(falsos_treino)
falsos01p1 = get1stPart(falsos_treino)
falsos01p2 = get2ndPart(falsos_treino)
p = Path('/home/lapis/Videos/Face_Forensics/Verdadeiros_Teste/c0')
verdadeiros_teste = list(p.glob('**/0*.mp4'))
verdadeiros_teste = ordenaListaPath(verdadeiros_teste)
verdadeiros02p1 = get1stPart(verdadeiros_teste)
verdadeiros02p2 = get2ndPart(verdadeiros_teste)
pat = Path('/home/lapis/Videos/Face_Forensics/Falsos_Teste/c0')
falsos_teste = list(pat.glob('**/0*.mp4'))
falsos_teste = ordenaListaPath(falsos_teste)
falsos02p1 = get1stPart(falsos_teste)
falsos02p2 = get2ndPart(falsos_teste)




dados = []
imagens_verd = []
imagens_false = []
faces_verd = []
faces_false = []
DPimagens_verd = []
DPimagens_false = []
DPfaces_verd = []
DPfaces_false = []
vermelho = (0, 0, 255)
i = 0
while i < len(verdadeiros_treino):
                      imagem01 = []
                      imagem02 = []
                      face01 = []
                      face02 = []
                      DPimagem01 = [0]
                      DPimagem02 = [0]
                      DPface01 = [0]
                      DPface02 = [0]
                      tempo = []
                      blur01 = []
                      blur02 = []
                      color01 = []
                      color02 = []
                      exp01 = []
                      exp02 = []
                      blur03 = []
                      blur04 = []
                      color03 = []
                      color04 = []
                      exp03 = []
                      exp04 = []
                      passou_im = 0
                      passou_fc = 0
                      count = 0
                      count_media = 0
                      print (verdadeiros_treino[i].name)
                      fn = str(verdadeiros_treino[i]) 
                      fnF = str(falsos_treino[i]) 
                      vidcap = cv2.VideoCapture(fn)
                      vidcapF = cv2.VideoCapture(fnF)
                      duration = getVideoDuration(vidcap)
                      intervalo = duration*50
                      new_intervalo = 0
                      fim = duration*1000
                      success,image = vidcap.read()
                      success,falsa = vidcapF.read()
                      success,opsV = vidcap.read()
                      success,opsF = vidcapF.read()
                      success = True
                      while success:
                          vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*100))
                          vidcapF.set(cv2.CAP_PROP_POS_MSEC,(count*100))    # espera um centésimo de segundo de vídeo 
                          success,image = vidcap.read()
                          success,falsa = vidcapF.read()
                          success,opsV = vidcap.read()
                          success,opsF = vidcapF.read()
                          if (success):
                              if (count == 50):
                                print("5 segundos de vídeo")
                              if (count == 100):
                                print("10 segundos de vídeo")
                              if (count == 10):
                                print("1 segundo de vídeo")
                              if (count == 20):
                                print("2 segundos de vídeo")
                              face_locations = face_recognition.face_locations(image)
                              for face_location in face_locations:
                                  top, right, bottom, left = face_location
                              face_locationsF = face_recognition.face_locations(falsa)
                              for face_location in face_locationsF:
                                  topF, rightF, bottomF, leftF = face_location
                              cv2.rectangle(opsF, (leftF, topF), (rightF, bottomF), vermelho, -1)
                              cv2.rectangle(opsV, (left, top), (right, bottom), vermelho, -1) 
                             
                              blur = detectBlur(opsV)                           
                              blur01.append(blur)
                              color = getL_color(opsV)  # métricas para a face verdadeira
                              color01.append(color)
                              exp = getL_exp(opsV)
                              exp01.append(exp)
                              
                              blur_F = detectBlur(opsF)
                              blur02.append(blur_F)
                              color_F = getL_color(opsF)     # métricas para as faces falsas
                              color02.append(color_F)
                              exp_F = getL_exp(opsF)
                              exp02.append(exp_F)

                            
                              if ((count*100 >= fim) and (passou_im == 0)):
                                  passou_im = 1
                                  tempV = []
                                  blur_V = transfDados(blur01)
                                  color_V = transfDados(color01)
                                  exp_V = transfDados(exp01)      # tabelando os dados
                                  tempV.append(color_V)
                                  tempV.append(exp_V)
                                  tempV.append(blur_V)
                                  tempF = []
                                  blur_F = transfDados(blur02)
                                  color_F = transfDados(color02)
                                  exp_F = transfDados(exp02)      
                                  tempF.append(color_F)
                                  tempF.append(exp_F)
                                  tempF.append(blur_F)
                                  
                              if (bottom < bottomF):
                                faceF = cortaImagem(falsa, leftF, rightF, bottomF, topF)
                                blur_faceF = detectBlur(faceF)
                                face = cortaImagem(image, leftF, rightF, bottomF, topF) 
                                blur_face = detectBlur(face)
                              else:
                                face = cortaImagem(image, left, right, bottom, top)
                                blur_face = detectBlur(face)
                                faceF = cortaImagem(falsa, left, right, bottom, top)
                                blur_faceF = detectBlur(faceF)
                              '''if (blur_faceF > blur_face):
                                print("man")'''
                            
                              blur03.append(blur_face)
                              blur04.append(blur_faceF)
                              color = getL_color(image)  # métricas para a face verdadeira
                              color03.append(color)
                              exp = getL_exp(image)
                              exp03.append(exp)
                              color = getL_color(image) 
                              color04.append(color)
                              exp = getL_exp(image)
                              exp04.append(exp)
                              
                              if ((count*100 >= fim) and (passou_fc == 0)):
                                  passou_fc = 1
                                  blur_V = transfDados(blur03)
                                  color_V = transfDados(color03)
                                  exp_V = transfDados(exp03)      
                                  tempV.append(color_V)
                                  tempV.append(exp_V)
                                  tempV.append(blur_V)
                                  tempV.append(1)
                                  dados.append(tempV)
                                  blur_F = transfDados(blur04)          # tabelando os dados
                                  color_F = transfDados(color04)
                                  exp_F = transfDados(exp04)      
                                  tempF.append(color_F)
                                  tempF.append(exp_F)
                                  tempF.append(blur_F)
                                  tempF.append(0)
                                  dados.append(tempF)
                                  print (count/10, "segundos de vídeo")
                                  break
                                  
                              
                              count = count + 1
                      
                      i = i + 1
                      imagens_verd.append(imagem01)
                      imagens_false.append(imagem02)
                      faces_verd.append(face01)
                      faces_false.append(face02)
                      imagem01 = []
                      imagem02 = []
                      face01 = []
                      face02 = [] 
                      DPimagens_verd.append(DPimagem01)
                      DPimagens_false.append(DPimagem02)
                      DPfaces_verd.append(DPface01)
                      DPfaces_false.append(DPface02)
                      DPimagem01 = []
                      DPimagem02 = []
                      DPface01 = []
                      DPface02 = [] 



with open('Regressão_logística.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["L_color_image", "L_exp_image", "L_blur_image", "L_color_face", "L_exp_face", "L_blur_face", "label"])
    writer.writerows(dados)
