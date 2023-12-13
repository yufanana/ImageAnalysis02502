import matplotlib.pyplot as plt
from skimage import io

dst_img = io.imread('data/' + 'Hand2.jpg')

plt.imshow(dst_img)
plt.show()
