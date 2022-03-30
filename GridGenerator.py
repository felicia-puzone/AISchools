

import numpy as np


#Media colore

def color_avg(img):
  img_reshape = img.reshape(img.shape[0]*img.shape[1], 3)
  
  
  avg_ch1 = img_reshape[:,0].sum()//img_reshape.shape[0]
  avg_ch1 = avg_ch1.astype('uint8')

  avg_ch2 = img_reshape[:,1].sum()//img_reshape.shape[0]
  avg_ch2 = avg_ch2.astype('uint8')

  avg_ch3 = img_reshape[:,2].sum()//img_reshape.shape[0]
  avg_ch3 = avg_ch3.astype('uint8')

  res = np.full(img.shape, [avg_ch1, avg_ch2, avg_ch3], dtype = np.uint8)
  return res
  
    
def color_distance(color1, color2):
    
  # Converto da uint a int perchè altrimenti con le differenze se viene un numero negativo sfora la codifica e va in overflow

  color1 = color1.astype(np.int16)
  color2 = color2.astype(np.int16)

  distance = np.abs(color1[0] - color2[0]) + np.abs(color1[1] - color2[1]) + np.abs(color1[2] - color2[2])
  return distance


def palette_choose(palette, pixel):

  #Per ogni pixel slista la palette di colori e restituisce il colore più vicino (Quello con minima distanza pixel-colore)

  distances = []

  for index, color in enumerate(palette):
    #print(color_distance(pixel, color))
    distances.append(color_distance(pixel, color))
    #print(index, color)

  (minvalue,minIndex) = min((v,i) for i,v in enumerate(distances))

  return palette[minIndex]


def pixxelate(img, sample_size, palette):
  (h, w) = img.shape[:2]

  (stepW, stepH) = ((w // sample_size) + 1, (h // sample_size) +1)

  print (stepW, stepH)

  img_res = np.empty(img.shape, dtype = np.uint8)

  ## 1 - prendi una sottomatrice stepW * stepH * 3 dall'array di partenza
  ## 2 - calcola la media (o qualsiasi altra funzione) dei colori
  ## 3 - inserisci nelle posizioni corrispondenti dell'immagine risultato il colore calcolato

  for i in range(sample_size):
    for j in range(sample_size):
      if(i*stepH + stepH <= img.shape[0] and j* stepW + stepW <= img.shape[1]):
        img_tmp = img[i*stepH :i*stepH + stepH, j* stepW :j* stepW + stepW]
        #avg = color_avg(img_tmp)
        avg = img_tmp

        palette_color = palette_choose(palette, avg[0][0])

        avg = np.full(avg.shape, [palette_color[0], palette_color[1], palette_color[2]], dtype = np.uint8)

        img_res[i*stepH :i*stepH + stepH, j* stepW :j* stepW + stepW] = avg

  return img_res



