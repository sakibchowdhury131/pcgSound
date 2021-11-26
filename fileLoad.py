import scipy.io.wavfile
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt


def test(test_file):
	sample_rate, sig = scipy.io.wavfile.read(test_file)
	print('sampling rate: ', sample_rate)
	print('length of samples', len(sig))
	return sample_rate, sig
	

def spec_vis(sig, sample_rate):
	t = np.linspace(0, len(sig)/sample_rate, len(sig))
	fig = plt.figure(figsize=(25,10))
	ax1 = fig.add_subplot(211)
	ax1.set_xlabel("time (sec)")
	ax1.set_ylabel("amplitude")
	ax1.set_title("Time domain signal plot")
	ax1.plot(t, sig)

def pre_filt(pre_emphasis):
 	return np.append(sig[0], sig[1:] - pre_emphasis * sig[:-1])
 
 
 def play(sig, sample_rate):
 	ipd.Audio(sig, rate=sample_rate)


