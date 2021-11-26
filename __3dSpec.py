from scipy.stats import multivariate_normal

X, Y = np.meshgrid(t,f)


fig = plt.figure(figsize=(40, 25))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Y, X, Sxx_out, cmap="plasma")

plt.show()
