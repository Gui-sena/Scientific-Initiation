from PIL import Image, ImageChops, ImageEnhance
import sys, os.path
import cv2
from google.colab.patches import cv2_imshow


## Getting the images


count = 0
vidcap = cv2.VideoCapture('/content/drive/My Drive/IC/Amostras/Verdadeiros_02/038.mp4')
vidcapF = cv2.VideoCapture('/content/drive/My Drive/IC/Amostras/Falsos_02/038_125.mp4')
vidcapF.set(cv2.CAP_PROP_POS_MSEC,(count*100))
vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*100))    
success,image = vidcap.read()
cv2.imwrite("/content/teste.jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
success,image = vidcapF.read()
cv2.imwrite("/content/testeF.jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])




## E.L.A. (Error Level Analysis)



# filename = "/content/kamala_harris_HF.jpg"
# filename = "/content/teste.jpg"
filename = "/content/testeF.jpg"
resaved = filename + '.resaved.jpg'
ela = filename + '.ela.png'

im = Image.open(filename)
ola = cv2.imread(filename)
cv2_imshow(ola)

im.save(resaved, 'JPEG', quality=90)
resaved_im = Image.open(resaved)

ela_im = ImageChops.difference(im, resaved_im)
extrema = ela_im.getextrema()
max_diff = max([ex[1] for ex in extrema])
scale = 15

ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

print ("Maximum difference was %d" % (max_diff))
ela_im.save(ela)
ela_cv = cv2.imread(ela)
cv2_imshow(ela_cv)
