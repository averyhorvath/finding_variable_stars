import numpy as np
import os
import os.path
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg


folder_path = "images/pms_stars_imgs"



images = []
for dirpath,dirnames, filenames in os.walk(folder_path):
    for filename in [f for f in filenames if f.startswith("EVR_")]:
        images.append(os.path.join(dirpath, filename))
        

 
images = images[350:400]
n = len(images)
gs1 = gridspec.GridSpec(n//5,5)

for i,img_path in enumerate(images):
    plt.subplot(gs1[i])
    img = mpimg.imread(img_path)
    plt.axis('off')
    plt.imshow(img)
plt.show()




