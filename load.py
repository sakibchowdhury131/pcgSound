import sys
import os
import scipy
import pandas as pd

def find_min_max_file_length(folder_location):
	subfolders = os.listdir(folder_location)
	lengths = []
	sample_rates = []
	fileIndex = 0
	totalFiles = 0
	for subfolder in subfolders:
	tempFiles = os.listdir(folder_location+'/'+subfolder)
	files = [file for file in tempFiles if file[-3:] == 'wav']
	totalFiles = totalFiles+len(files)


	for subfolder in subfolders:
	files = os.listdir(folder_location+'/'+subfolder)
	print(subfolder)

	for file in files:
	if file[-3:] == 'wav':
	opened_file = folder_location+ '/' +subfolder + '/' + file
	sample_rate, signal = scipy.io.wavfile.read(opened_file)
	sample_rates.append(sample_rate)
	if len(signal) == 243997:
	  x = folder_location+'/'+subfolder+'/'+file
	lengths.append(len(signal))
	fileIndex = fileIndex+1
	if (fileIndex%100 == 0):
	  print(subfolder + '/'+file+':' + str(fileIndex / totalFiles*100) + '% extraction done')
	sys.stdout.flush()

	print(subfolder + ": "+ "extraction completed")
	print(len(lengths))
	print(len(sample_rates))
	print(x)
	return (min(lengths), max(lengths))
  
  
def read_ref(ref_file):
	val_info = pd.read_csv(ref_file, names = ['file', 'class'])
	val_list = val_info['file'].values.tolist()
	val_list = [val_list[i]+'.wav' for i in range(len(val_list))]
	print(val_list)
	


