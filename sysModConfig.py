import pandas as pd
import os
import sys
import numpy as np
import scipy
import fileLoad
import load



def ModifiedLoading(folder_location, signal_length, val_list, pre_emphasis = False, pre_emphasis_ratio = 0.79, train_or_val = 'train'):
  subfolders = os.listdir(folder_location)
  lengths = []
  sample_rates = []
  fileIndex = 0
  totalFiles = 0

  for subfolder in subfolders:
    tempFiles = os.listdir(folder_location+'/'+subfolder)
    files = [file for file in tempFiles if file[-3:] == 'wav']
    totalFiles = totalFiles+len(files)


  print("total number of files: ", totalFiles)
  if train_or_val == 'train':
    x = np.zeros(shape = (totalFiles-len(val_list), min_signal_length))
    y = np.zeros(shape = (totalFiles-len(val_list), 1))
  elif train_or_val == 'val':
    x = np.zeros(shape = (totalFiles, min_signal_length))
    y = np.zeros(shape = (totalFiles, 1))

  
  for subfolder in subfolders:
    csv_file = folder_location + '/' + subfolder + '/REFERENCE.csv'
    csv_info = pd.read_csv(csv_file, names = ['file', 'class'])


    for i in range (0, len(csv_info)):
      if train_or_val == 'train':
        if csv_info['file'][i]+'.wav' not in val_list:
          fileName = folder_location + '/' + subfolder + '/' + csv_info['file'][i]+'.wav'
          print(csv_info['file'][i], csv_info['class'][i])
          y[fileIndex] = 0 if csv_info['class'][i] == 1 else 1
          sample_rate, sig = scipy.io.wavfile.read(fileName)
          if pre_emphasis == True:
            sig =  np.append(sig[0], sig[1:] - pre_emphasis_ratio * sig[:-1])
          x[fileIndex, 0:signal_length] = sig[0:signal_length]
          fileIndex = fileIndex + 1
      elif train_or_val == 'val':
        fileName = folder_location + '/' + subfolder + '/' + csv_info['file'][i]+'.wav'
        print(csv_info['file'][i], csv_info['class'][i])
        y[fileIndex] = 0 if csv_info['class'][i] == 1 else 1
        sample_rate, sig = scipy.io.wavfile.read(fileName)
        if pre_emphasis == True:
          sig =  np.append(sig[0], sig[1:] - pre_emphasis_ratio * sig[:-1])
        x[fileIndex, 0:signal_length] = sig[0:signal_length]
        fileIndex = fileIndex + 1

  print(train_or_val + ' set generation completed')
  return x, y
