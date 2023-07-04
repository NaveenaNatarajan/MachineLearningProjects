#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[29]:


pip install opencv-python


# In[115]:


img = cv2.imread(r"C:\Users\Navir\OneDrive\Documents\Image Classification\Dataset\virat kohli - Google Search\Virat_Kohli.jpg")
img.shape


# In[32]:


plt.imshow(img)


# In[33]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray.shape


# In[34]:


gray


# In[35]:


face_cascade = cv2.CascadeClassifier("C:/Users/navir/Downloads/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("C:/Users/navir/Downloads/haarcascade_eye.xml")

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
faces


# In[36]:


(x,y,w,h) = faces[0]
x,y,w,h


# In[37]:


face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
plt.imshow(face_img)


# In[99]:


cv2.destroyAllWindows()
for (x,y,w,h) in faces:
    face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = face_img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        

plt.figure()
plt.imshow(face_img, cmap='gray')
plt.show()


# In[39]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(roi_color, cmap='gray')


# In[101]:


def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color


# In[102]:


original_image = cv2.imread("C:/Users/Navir/OneDrive/Documents/Image Classification/Dataset/virat kohli - Google Search/Virat_Kohli.jpg")
original_image
plt.imshow(original_image)


# In[103]:


cropped_image = get_cropped_image_if_2_eyes("C:/Users/Navir/OneDrive/Documents/Image Classification/Dataset/virat kohli - Google Search/Virat_Kohli.jpg")
plt.imshow(cropped_image)


# In[104]:


org_image_obstructed = cv2.imread("C:/Users/navir/OneDrive/Documents/Image Classification/Dataset/virat kohli - Google Search/1662657385-7641.jpg")
plt.imshow(org_image_obstructed)


# In[105]:


cropped_image_no_2_eyes = get_cropped_image_if_2_eyes("C:/Users/navir/OneDrive/Documents/Image Classification/Dataset/virat kohli - Google Search/1662657385-7641.jpg")
cropped_image_no_2_eyes


# In[106]:


path_to_data = "C:/Users/navir/OneDrive/Documents/Image Classification/Dataset"
path_to_cr_data = "C:/Users/navir/OneDrive/Documents/Image Classification/Dataset/cropped/"


# In[107]:


import os
img_dirs = []
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)


# In[108]:


img_dirs


# In[109]:


import shutil
if os.path.exists(path_to_cr_data):
     shutil.rmtree(path_to_cr_data)
os.mkdir(path_to_cr_data)


# In[110]:


cropped_image_dirs = []
celebrity_file_names_dict = {}

for img_dir in img_dirs:
    count = 1
    celebrity_name = img_dir.split('/')[-1]
    print(celebrity_name)
    


# In[113]:


celebrity_file_names_dict[celebrity_name] = []
for entry in os.scandir(img_dir):
    roi_color = get_cropped_image_if_2_eyes(entry.path)
    if roi_color is not None:
        cropped_folder = path_to_cr_data + celebrity_name
        if not os.path.exists(cropped_folder):
            os.makedirs(cropped_folder)
            cropped_image_dirs.append(cropped_folder)
            print("Generating cropped images in folder: ",cropped_folder)


# In[ ]:





# In[ ]:




