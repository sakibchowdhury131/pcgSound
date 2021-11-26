N_CHANNELS = 3

def get_3d_spec(Sxx_in, moments=None):
    if moments is not None:
        (base_mean, base_std, delta_mean, delta_std,
             delta2_mean, delta2_std) = moments
    else:
        base_mean, delta_mean, delta2_mean = (0, 0, 0)
        base_std, delta_std, delta2_std = (1, 1, 1)
    h, w = Sxx_in.shape
    plt.imshow(Sxx_in)
    right1 = np.concatenate([Sxx_in[:, 0].reshape((h, -1)), Sxx_in], axis=1)[:, :-1]
    #plt.imshow(right1)
    delta = (Sxx_in - right1)[:, 1:]
    delta_pad = delta[:, 0].reshape((h, -1))
    delta = np.concatenate([delta_pad, delta], axis=1)
    right2 = np.concatenate([delta[:, 0].reshape((h, -1)), delta], axis=1)[:, :-1]
    delta2 = (delta - right2)[:, 1:]
    delta2_pad = delta2[:, 0].reshape((h, -1))
    delta2 = np.concatenate([delta2_pad, delta2], axis=1)
    base = (Sxx_in - base_mean) / base_std
    delta = (delta - delta_mean) / delta_std
    delta2 = (delta2 - delta2_mean) / delta2_std
    stacked = [arr.reshape((h, w, 1)) for arr in (base, delta, delta2)]
    return np.concatenate(stacked, axis=2)
    
    
    
def get_moments(X, seq_len, samp=1000):
    nrows = X.shape[0]
    Sxx_samples = [spectrogram(X[0:seq_len])[2]]
    sxx_h, sxx_w = Sxx_samples[0].shape
    Sxx_3d_samples = np.array([get_3d_spec(Sxx) for Sxx in Sxx_samples])
    base_mean = Sxx_3d_samples[:, :, :, 0].mean()
    base_std = Sxx_3d_samples[:, :, :, 0].std()
    delta_mean = Sxx_3d_samples[:, :, :, 1].mean()
    delta_std = Sxx_3d_samples[:, :, :, 1].std()
    delta2_mean = Sxx_3d_samples[:, :, :, 2].mean()
    delta2_std = Sxx_3d_samples[:, :, :, 2].std()
    return (sxx_h, sxx_w), (base_mean, base_std, delta_mean,
                            delta_std, delta2_mean, delta2_std)

(sxx_h, sxx_w), moments = get_moments(X_train[0], max_signal_length)
print(f'spectrogram dims: {sxx_h}x{sxx_w}')
print(f'base spectrogram mean, sigma: {moments[0]:.4f}, {moments[1]:.4f}')
print(f'delta spectrogram mean, sigma: {moments[2]:.4f}, {moments[3]:.4f}')
print(f'delta-delta spectrogram mean, sigma: {moments[4]:.4f}, {moments[5]:.4f}')


start = 0
sig_in = X_train[100, :]

f, t, Sxx = spectrogram(sig_in)
s3d = get_3d_spec(Sxx, moments)

fig = plt.figure(figsize=(27, 24))
ax = fig.add_subplot(4, 1, 1)
ax.margins(x=0.003)
plt.plot(X_train[0,:])
plt.title('signal:', fontsize=40, loc='left')
plt.axis('off')

ax = fig.add_subplot(4, 1, 2)
cmap = plt.get_cmap('magma')
spec = plt.pcolormesh(t, f, s3d[:, :, 0], cmap=cmap)
plt.title('log spectrogram ',
          fontsize=40, loc='left')
plt.axis('off')

ax = fig.add_subplot(4, 1, 3)
spec = plt.pcolormesh(t, f, s3d[:, :, 1], cmap=cmap)
plt.title('delta log spectrogram:', fontsize=40, loc='left')
plt.axis('off')

ax = fig.add_subplot(4, 1, 4)
spec = plt.pcolormesh(t, f, s3d[:, :, 2], cmap=cmap)
plt.title('delta-delta log spectrogram:', fontsize=40, loc='left')
plt.axis('off');
