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


## methods


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



p = []
pat = []
p = Path('/content/drive/My Drive/IC/Amostras/Verdadeiros_01')
verdadeiros01 = list(p.glob('**/0*.mp4'))
verdadeiros01 = ordenaListaPath(verdadeiros01)
verdadeiros01p1 = get1stPart(verdadeiros01)
verdadeiros01p2 = get2ndPart(verdadeiros01)
pat = Path('/content/drive/My Drive/IC/Amostras/Falsos_01')
falsos01 = list(pat.glob('**/0*.mp4'))
falsos01 = ordenaListaPath(falsos01)
falsos01p1 = get1stPart(falsos01)
falsos01p2 = get2ndPart(falsos01)
p = Path('/content/drive/My Drive/IC/Amostras/Verdadeiros_02')
verdadeiros02 = list(p.glob('**/0*.mp4'))
verdadeiros02 = ordenaListaPath(verdadeiros02)
verdadeiros02p1 = get1stPart(verdadeiros02)
verdadeiros02p2 = get2ndPart(verdadeiros02)
pat = Path('/content/drive/My Drive/IC/Amostras/Falsos_02')
falsos02 = list(pat.glob('**/0*.mp4'))
falsos02 = ordenaListaPath(falsos02)
falsos02p1 = get1stPart(falsos02)
falsos02p2 = get2ndPart(falsos02)


## Processando as imagens

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
cv2.setNumThreads(0)


class Th(Thread):

                def __init__ (self, num, vidcap, vidcapF, verdadeiros, falsos):
                      Thread.__init__(self)
                      self.num = num
                      self.vidcap = vidcap
                      self.vidcapF = vidcapF
                      self.verdadeiros = verdadeiros
                      self.falsos = falsos

                def run(self):

                      imagem01 = []
                      imagem02 = []
                      face01 = []
                      face02 = []
                      DPimagem01 = [0]
                      DPimagem02 = [0]
                      DPface01 = [0]
                      DPface02 = [0]
                      tempo = []
                      temporario01 = []
                      temporario02 = []
                      temporario03 = []
                      temporario04 = []
                      count = 0
                      count_media = 0
                      blur_ant = 0
                      blur_ant_F = 0
                      face_ant = 0
                      face_ant_F = 0
                      print (self.verdadeiros[self.num].name)
                      fn = str(self.verdadeiros[self.num]) 
                      fnF = str(self.falsos[self.num]) 
                      duration = getVideoDuration(self.vidcap)
                      intervalo = duration*50
                      new_intervalo = 0
                      success,image = self.vidcap.read()
                      success,falsa = self.vidcapF.read()
                      success = True
                      while success:
                          self.vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*100))
                          self.vidcapF.set(cv2.CAP_PROP_POS_MSEC,(count*100))    # espera um centésimo de segundo de vídeo 
                          success,image = self.vidcap.read()
                          success,falsa = self.vidcapF.read()
                          success,opsV = self.vidcap.read()
                          success,opsF = self.vidcapF.read()
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
                           
                              if (blur_ant != 0):
                                temporario01.append(blur - blur_ant)                  # usado para fazer a diferença entre blur´s
                              else:
                                temporario01.append(blur_ant)
                              blur_ant = blur
                              blurF = detectBlur(opsF)
                              if (blur_ant_F != 0):
                                temporario02.append(blur - blur_ant_F)
                              else:
                                temporario02.append(blur_ant_F)
                              blur_ant_F = blurF
                            
                              if (count*100 >= new_intervalo):
                                  rsc = 0   
                                  for item in temporario01:
                                      rsc += item # rsc + temporario[i]    
                                  rsc = rsc/len(temporario01)
                                  rsc = rsc/1000
                                  imagem01.append(rsc)
                                  if (count != 0):
                                    DPimagem01.append(statistics.stdev(temporario01))
                                  temporario01 = []
                                  rsc = 0   
                                  for item in temporario02:
                                      rsc += item # rsc + temporario[i]    
                                  rsc = rsc/len(temporario02)
                                  rsc = rsc/1000
                                  imagem02.append(rsc)
                                  if (count != 0):
                                    DPimagem02.append(statistics.stdev(temporario02))
                                  temporario02 = []
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
                              if (blur_faceF > blur_face):
                                cv2_imshow(face)
                                cv2_imshow(faceF)
                                print("man")
                            
                              if (face_ant != 0):
                                temporario03.append(blur_face - face_ant)
                              else:
                                temporario03.append(face_ant)
                              face_ant = blur_face
                              if (face_ant_F != 0):
                                temporario04.append(blur_faceF - face_ant_F)
                              else:
                                temporario04.append(face_ant_F)
                              face_ant_F = blur_faceF
                            
                              if (count*100 >= new_intervalo):
                                  rsc = 0   
                                  for item in temporario03:
                                      rsc += item # rsc + temporario[i]    
                                  rsc = rsc/len(temporario03)
                                  rsc = rsc/1000
                                  face01.append(rsc)
                                  if (count != 0):
                                    DPface01.append(statistics.stdev(temporario03))
                                  temporario03 = []
                                  # print (rsc)
                                  # print (count*100)
                                  # print (new_intervalo)
                                  rsc = 0   
                                  for item in temporario04:
                                      rsc += item # rsc + temporario[i]    
                                  rsc = rsc/len(temporario04)
                                  rsc = rsc/1000
                                  face02.append(rsc)
                                  if (count != 0):
                                    DPface02.append(statistics.stdev(temporario04))
                                  temporario04 = []
                                  count_media +=1
                                  new_intervalo += intervalo
                              count = count + 1
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


for thread_number in range (len(verdadeiros02p1)):
    fn = str(verdadeiros02p1[thread_number]) 
    fnF = str(falsos02p1[thread_number]) 
    vidcap = cv2.VideoCapture(fn)
    vidcapF = cv2.VideoCapture(fnF)
    thread = Th(thread_number, vidcap, vidcapF, verdadeiros02p1, falsos02p1)
    thread.start() 

while (len(imagens_verd)) < 4:
    gui=10
    
    
## Extraindo lista média
    
imagem01 = getListaMedia(imagens_verd)
imagem02 = getListaMedia(imagens_false)
face01 = getListaMedia(faces_verd)
face02 = getListaMedia(faces_false)
DPimagem01 = getListaMedia(DPimagens_verd)
DPimagem02 = getListaMedia(DPimagens_false)
DPface01 = getListaMedia(DPfaces_verd)
DPface02 = getListaMedia(DPfaces_false)



## Extraindo tempo

count_media = 21
tempo = []
tempo_especial = []
i = 0
dif = count_media
j = 1/dif
while i<1:
  tempo.append(i)
  tempo_especial.append(i)
  i += j
print (dif)
tempo_especial.append(1)
len(tempo)


## Calculando a diferença


dif_faces = []
dif_imagens = []
dif_DPfaces = []
dif_DPimagens = []
i = 0
while i < len(face01):
  temp = face01[i] - face02[i]
  dif_faces.append(temp)
  temp = imagem01[i] - imagem02[i]
  dif_imagens.append(temp)
  temp = DPface01[i] - DPface02[i]
  dif_DPfaces.append(temp)
  temp = DPimagem01[i] - DPimagem02[i]
  dif_DPimagens.append(temp)
  print(i)
  i = i +1
  
  
  
