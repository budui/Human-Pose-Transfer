from PIL import Image
import os
import numpy as np

img_dir = './_generated'
save_dir = './_r'

if not os.path.exists(save_dir):
	os.mkdir(save_dir)

np.random.seed(252)
items = os.listdir(img_dir)
np.random.shuffle(items)

cnt = 0
for i, item in enumerate(items):
	if not item.endswith('.jpg') and not item.endswith('.png'):
		continue
	if i >= 1000:
		break
	cnt = cnt + 1
	print('%d/12000 ...' %(cnt))
	img = Image.open(os.path.join(img_dir, item))
	# for 5 split
	imgcrop = img.crop((256, 0, 320, 128))
	imgcrop.save(os.path.join(save_dir, item))
