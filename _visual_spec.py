import __makeSpec
import matplotlib.pyplot as plt

audio_index = 120
f, t, Sxx_out = spectrogram(X_train[audio_index, :])

fig = plt.figure(figsize=(18, 8))
ax = fig.add_subplot(2, 1, 1)
ax.margins(x=0.003)
plt.plot(X_train[audio_index, :])
plt.title('signal:', fontsize=18, loc='left')
plt.axis('off')

ax = fig.add_subplot(2, 1, 2)
cmap = plt.get_cmap('magma')
spec = plt.pcolormesh(t, f, Sxx_out, cmap=cmap)
plt.title('normalized log spectrogram (aspect stretched):',
          fontsize=18, loc='left')
plt.axis('off');
