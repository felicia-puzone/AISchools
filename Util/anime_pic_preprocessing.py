import cv2
print('OpenCV version: ' + cv2.__version__)

from matplotlib import pyplot as plt

import numpy as np

#git clone https://github.com/felichan98/AISchools
#import sys
#sys.path.insert(0,'/content/AISchools')

"""Importo la directory, carico una lista di immagini in list_screenshots"""

import os

directory = './trainA/'

#Preparo una lista di img di input raccolte dalla cartella PixelArt_Screenshot

list_pic = []

i = 0

for filename in os.scandir(directory):
    if filename.is_file():
      i = i+1
      if(i < 1500):
          path = directory + filename.name
          img = cv2.imread(path, cv2.IMREAD_COLOR)
          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

          list_pic.append(img)
      else:
        break

"""Rimozione bande nere"""

def remove_black_lines(img):

  y_nonzero, x_nonzero, _ = np.nonzero(img)

  try:
    img = img[np.min(y_nonzero) + 5 :np.max(y_nonzero) -5, np.min(x_nonzero) +5:np.max(x_nonzero) -5]
    return img
  except:
    return img


"""Square-shaping

Se la foto è rettangolare, viene resa quadrata (mantenendo lo stesso centro)
"""

def square(img):
  if(img.shape[0] != img.shape[1]):
    max_dim = np.amax(img.shape[0:2])
    min_dim = np.amin(img.shape[0:2])
    center = max_dim // 2

    if(img.shape[0] == max_dim):
      img = img[center-min_dim//2:center+min_dim//2,:]

    if(img.shape[1] == max_dim):
      img = img[:,center-min_dim//2:center+min_dim//2]

  return img

"""Resize

Ridimensiono tutto a 256 x 256px
"""

def resize(img, dim):
  img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)
  return img_resized
  

out_path = './out/'

#plt.imshow(list_screenshots[0])
#plt.show()

for i in range(len(list_pic)):

    cropped_img = list_pic[i]

    cropped_img = remove_black_lines(cropped_img)
  
    cropped_img = square(cropped_img)
    cropped_img = resize(cropped_img, (256,256))

    filename = out_path + str(i) + '.png'

    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, cropped_img)

