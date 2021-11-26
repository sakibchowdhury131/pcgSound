import numpy as np

def get_image_from_audio(data):
  _,_,sxx = spectrogram(data[0])
  sxx3d = get_3d_spec(sxx,moments)
  (height, width, channels) = sxx3d.shape


  processed_data = np.zeros(shape = (data.shape[0], height, width, channels)) 

  for i in range(0, data.shape[0]):
    _,_,sxx = spectrogram(data[i])
    processed_data[i] = get_3d_spec(sxx, moments)

  return processed_data
