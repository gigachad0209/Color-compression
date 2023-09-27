import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as im

def read_image(im_path):
	img = plt.imread(im_path)
	img = img / 255.0
	return img


def initialize_centroid(img, k_clusters):
    w, h, d = img.shape
    new_size = w * h
    imArr = img.reshape(new_size, d)
    centroid = np.zeros((k_clusters, d))
    for i in range(k_clusters): #assigning random centroids
        centroid[i] = np.mean(imArr[np.random.choice(new_size, size=10, replace=False)], axis=0)
    return imArr, centroid

def euclidean_distance(a1, b1, a2, b2):
	d = np.square(a1 - a2) + np.square(b1 - b2)
	return np.sqrt(d)

def k_means(imArr, centroid, max_iter, k_cluster):
	im_size = imArr.shape[0]
	index = np.zeros(im_size)
	for t in range(max_iter):
		for j in range(im_size):
			distance = float('inf')
			for k in range(k_cluster):
				x1, y1 = imArr[j, 0], imArr[j, 1]
				x2, y2 = centroid[k, 0], centroid[k, 1]
				if euclidean_distance(x1, y1, x2, y2) <= distance:
					distance = euclidean_distance(x1, y1, x2, y2)
					index[j] = k

		for k in range(k_cluster):
			cluster_points = imArr[index == k]
			if len(cluster_points) > 0:
				centroid[k] = np.mean(cluster_points, axis=0)

	return centroid, index

def image_compression(centroid, index, img):
	
	klusters = np.array(centroid*255.0, dtype = np.uint8)
	new_image = klusters[index.astype(int), :]
	new_image = new_image.reshape(img.shape)
	return new_image
	
if __name__ == '__main__':
	image_path = str(input("Nhập đường dẫn ảnh: "))
	img = read_image(image_path)

	clusters = int(input("Nhập số màu bạn muốn: "))
	image_format = str(input("Nhập định dạng ảnh mà bạn muốn lưu: "))
	img_1d, centroid = initialize_centroid(img,clusters)
	centroid, index = k_means(img_1d, centroid, 10,  clusters)
	new_image = image_compression(centroid, index, img)

	
	picture = im.fromarray(new_image, 'RGB')
	picture.save('output.' + image_format)

	plt.imshow(new_image)
	plt.show()
	

