import cv2
import numpy as np

#DICTIONARY TONALITA' 

COLOR_HUES =	{
  "Orange": [0,15],
  "Yellow": [15,30],
  "Lime": [30,45],
  "Green": [45,60],
  "Mint Green": [60,75],
  "Acquamarine": [75,90],
  "Blue": [90,105],
  "Indigo": [105,120],
  "Violet": [120,135],
  "Purple": [135,150],
  "Pink": [150,165],
  "Red": [165,180]
}

COLOR_SAT =	{
  "1": [0,51],  
  "2": [51,102],
  "3": [102,153],
  "4": [153,204],
  "5": [204,255],
}

# DICTIONARY SCHEMA BRIGHTNESS

COLOR_VAL =	{
  "1": [0,51],  
  "2": [51,102],
  "3": [102,153],
  "4": [153,204],
  "5": [204,255],
}

def hue_present(img, hue):
  total = img.shape[0] * img.shape[1]
  (H, S, V) = cv2.split(img)
  hist = cv2.calcHist([H],[0],None,[180],[0,180])

  if((hist[hue[0]:hue[1]].max() / total) > 0.005):
    return True
  else:
    return False

def sat_present(img, sat):
  total = img.shape[0] * img.shape[1]
  (H, S, V) = cv2.split(img)
  hist = cv2.calcHist([S],[0],None,[256],[0,255])

  if((hist[sat[0]:sat[1]].max() / total) > 0.003):
    return True
  else:
    return False

def val_present(img, val):
  total = img.shape[0] * img.shape[1]
  (H, S, V) = cv2.split(img)
  hist = cv2.calcHist([V],[0],None,[256],[0,255])

  if((hist[val[0]:val[1]].max() / total) > 0.003):
    return True
  else:
    return False

def hue_counter(img):
  k = 0

  for key in COLOR_HUES:
    #print(key, hue_present(img, COLOR_HUES[key]))
    if(hue_present(img, COLOR_HUES[key])):
      k += 1

  #print('Numero cluster hue:', k)
  return k

def sat_counter(img):
  k = 0

  for key in COLOR_SAT:
    #print(key, sat_present(img, COLOR_SAT[key]))
    if(sat_present(img, COLOR_SAT[key])):
      k += 1

  #print('Numero cluster saturation:', k)
  return k

def val_counter(img):
  k = 0

  for key in COLOR_VAL:
    #print(key, val_present(img, COLOR_VAL[key]))
    if(val_present(img, COLOR_VAL[key])):
      k += 1

  #print('Numero cluster val:', k)
  return k

def basic_kmeans(channel, k):
  Z = channel.reshape(-1)

  Z = np.float32(Z)

  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

  K = k
  ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

  center = np.uint8(center)
  res = center[label.flatten()]

  channel_out = res.reshape(channel.shape)
  return channel_out


def my_kmeans(img_hsv):

  K_h = hue_counter(img_hsv)
  K_s = sat_counter(img_hsv)
  K_v = val_counter(img_hsv)

  (H, S, V) = cv2.split(img_hsv)

  # HUE #
  H_k = basic_kmeans(H, K_h)

  # SAT #
  S_k = basic_kmeans(S, K_s)

  # VAL #
  V_k = basic_kmeans(V, K_v)

  #Merge canali
  out = cv2.merge([H_k, S_k, V_k])

  out_rgb = cv2.cvtColor(out, cv2.COLOR_HSV2RGB)

  return out_rgb


def palette_extractor(img):

  #Conversione dell'immagine in HSV
  #img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

  #Applicazione dell'algoritmo di clustering 
  img_out_rgb = basic_kmeans(img,30)

  #Estraggo i colori unici dall'immagine
  img_flatten = img_out_rgb.reshape(img_out_rgb.shape[0]*img_out_rgb.shape[1], 3)

  color_unique = np.unique(img_flatten, axis=0)

  #Ottengo una palette di colori RGB in uscita
  return color_unique
