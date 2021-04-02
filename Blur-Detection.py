!pip install face_recognition

import face_recognition
from PIL import Image
import face_recognition
import dlib
from google.colab.patches import cv2_imshow
import math
import cv2
import statistics
from pathlib import Path
from threading import Thread

def drawLine(imagem, newLeft, newRight, newBottom, newTop):
  imagem = cv2.line(imagem, (newLeft, newBottom), (newLeft, newTop), (255, 0, 0), 9)
  imagem = cv2.line(imagem, (newRight, newBottom), (newRight, newTop), (255, 0, 0), 9)
  imagem = cv2.line(imagem, (newLeft, newTop), (newRight, newTop), (255, 0, 0), 9)
  imagem = cv2.line(imagem, (newRight, newBottom), (newLeft, newBottom), (255, 0, 0), 9)

def variance_of_laplacian(image):
  return cv2.Laplacian(image, cv2.CV_64F).var()

def detectBlur(image):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  fm = variance_of_laplacian(gray)
  height, width = image.shape[:2]
  res = (fm*height)/10
  return res

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
