#!/usr/bin/env python
# coding: utf-8

# # C5T1 Plan of Attack Step 3

# In[2]:


import math


# In[3]:


import numpy


# In[4]:


import pandas


# In[5]:


import matplotlib


# In[6]:


import scipy


# In[7]:


math.sqrt(25)


# In[8]:


math.sqrt(36)


# In[9]:


math.pi


# In[10]:


from numpy import random


# In[11]:


import numpy as np


# # uniform random numbers in [0,1]

# In[12]:


dataOne = random.rand(5,5)


# In[13]:


dataOne


# In[14]:


np.mean(dataOne)


# In[16]:


print('I love wolf')


# In[17]:


variablename = 25


# In[18]:


variablename


# In[19]:


variablename1 = 'I love python'


# In[20]:


variablename1


# In[21]:


print (variablename1)


# In[22]:


print (variablename)


# In[23]:


type(variablename)


# In[24]:


variablename = 25.0


# In[25]:


variablename


# In[26]:


type(variablename)


# In[27]:


varOne = 25


# In[28]:


varTwo = 25.0


# In[29]:


varThree = varOne + varTwo


# In[30]:


print(varThree)


# In[31]:


type (varThree)


# In[32]:


print ("Data type of varTwo", type(varOne))


# In[33]:


varThree = varOne + varTwo


# In[34]:


varThree


# In[35]:


type (varThree)


# In[36]:


varTwo = int(varTwo)


# In[37]:


type(varTwo)


# In[38]:


varThree


# In[39]:


type (varThree)


# In[40]:


varThree = varOne + varTwo


# In[41]:


varThree


# In[42]:


type (varThree)


# In[44]:


listOne = [1,2,3,4]


# In[45]:


print(listOne[1:3])


# In[46]:


print(listOne[0:2])


# In[47]:


print(listOne[0:1])


# In[48]:


print(listOne[1:4])


# In[49]:


print(listOne[1:5])


# In[50]:


vowels = ['a','e','i','o','i','u']


# In[51]:


print(vowels[3:4])


# In[52]:


vowels[3:4]


# In[53]:


tel = {'jack':4098,'sape':4139}


# In[54]:


tel['jack']


# In[55]:


variabletry = {'wolf':3, 'rabbit':5, 'bear':1, 'dog':2, 'cow':6}


# In[56]:


'wolf'+'rabbit'


# In[57]:


variabletry['rabbit']


# # C5T1 - P4DSD2_05_Understanding the Tools

# ## Using Jupyter Notebook

# ## Working with styles

# In[59]:


# %load https://matplotlib.org/_downloads/pyplot_text.py


# In[60]:


import numpy as np
import matplotlib.pyplot as plt


# In[61]:


# Fixing random state for reproducibility


# In[63]:


np.random.seed(19680801)

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)


# In[64]:


# the histogram of the data


# In[65]:


n, bins, patches = plt.hist(x, 50, density=True, facecolor='g', alpha=0.75)


plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()


# In[66]:


import matplotlib
matplotlib.pyplot.hist
matplotlib.pyplot.xlabel
matplotlib.pyplot.ylabel
matplotlib.pyplot.text
matplotlib.pyplot.grid
matplotlib.pyplot.show


# ## Obtaining online graphics and multimedia

# In[67]:


from IPython.display import Image
Embed = Image ('http://blog.johnmuellerbooks.com/'+ 'wp-content/uploads/2015/04/Layer-Hens.jpg')
Embed


# In[68]:


from IPython.display import Image
SoftLinked = Image ('http://blog.johnmuellerbooks.com/wp-content/uploads/2015/04/Layer-Hens.jpg')
SoftLinked


# ## Embeded as pdf

# In[69]:


from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf','svg')


# # C5T1 - P4DSD2_05_Understanding the Tools

# ## Uploading, Streaming, and Sampling Data

# ## Upload small amounts of data into memory

# In[70]:


with open ("Colors.txt", 'r') as open_file:
    print('Colors.txt content:\n' + open_file.read())


# ## Work with part of the data instead of whole dataset

# In[71]:


with open ("Colors.txt", 'r') as open_file:
    for observation in open_file:
        print('Reading Data: ' + observation)


# ## Generating variations on image data

# In[72]:


import matplotlib.image as img
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

image = img.imread("Colorblk.jpg")
print (image.shape)
print (image.size)
plt.imshow (image)
plt.show()


# ## Sampling data in different ways
# 

# In[73]:


n = 2
with open ("Colors.txt",'r') as open_file:
    for j, observation in enumerate(open_file):
        if j % n==0:
            print('Reading Line: ' + str(j) + ' Content: ' + observation)


# In[74]:


n = 3
with open ("Colors.txt",'r') as open_file:
    for j, observation in enumerate(open_file):
        if j % n==0:
            print('Reading Line: ' + str(j) + ' Content: ' + observation)


# In[75]:


n = 3
with open ("Colors.txt",'r') as open_file:
    for j, observation in enumerate(open_file):
        if j % n==1:
            print('Reading Line: ' + str(j) + ' Content: ' + observation)


# In[77]:


from random import random
sample_size = 0.25
with open("Colors.txt", 'r') as open_file:
    for j, observation in enumerate(open_file):
        if random()<=sample_size:
            print ('Reading Line: '+ str(j) + ' Content: ' + observation)


# ## Assessing Data in Structured Flat-File Form

# In[78]:


import pandas as pd
color_table = pd.io.parsers.read_table("Colors.txt")
print(color_table)


# In[79]:


import pandas as pd
titanic = pd.io.parsers.read_csv("Titanic.csv")
X = titanic[["age"]]
print (X)


# In[80]:


import pandas as pd
titanic = pd.io.parsers.read_csv("Titanic.csv")
X = titanic[['age']].values
print (X)


# In[81]:


import pandas as pd
xls = pd.ExcelFile("Values.xls")
trig_values = xls.parse('Sheet1', index_col=None),
na_values=['NA']
print(trig_values)


# ## Sending Data in Unstructured File Form

# ## Work with a picture as an unstructured file

# In[82]:


from skimage.io import imread
from skimage.transform import resize
from matplotlib import pyplot as plt
import matplotlib.cm as cm

example_file = ("http://upload.wikimedia.org/" + "wikipedia/commons/7/7d/Dog_face.png")
image = imread (example_file, as_gray=True)
plt.imshow(image, cmap=cm.gray)
plt.show()


# In[83]:


print ("data type: %s, shape: %s" % (type(image),image.shape))


# In[84]:


image2 = image [5:70,0:70]
plt.imshow(image2, cmap = cm.gray)
plt.show()


# In[85]:


image3 = resize(image2, (30,30), mode='symmetric')
plt.imshow(image3, cmap =cm.gray)
print("data type: %s, shape: %s" % (type(image3), image3.shape))


# In[86]:


image_row = image3.flatten()
print("data type: %s, shape:% s" % (type(image_row), image_row.shape))


# ## Managining Data from Relational Databases

# In[87]:


from sqlalchemy import create_engine
engine = create_engine('sqlite:///:memory:')


# ## Interacting with Data from NoSQL Databases

# ## Accessing Data from the Web

# In[88]:


from lxml import objectify
import pandas as pd

xml = objectify.parse(open('XMLData.xml'))
root = xml.getroot()

df = pd.DataFrame(columns=('Number','String','Boolean'))

for i in range(0,4):
    obj = root.getchildren()[i].getchildren()
    row = dict(zip(['Number','String','Boolean'],
              [obj[0].text, obj[1].text,
               obj[2].text]))
    row_s = pd.Series(row)
    row_s.name = 1
    df = df.append(row_s)

print(df)


# # An example of a .csv file you've loaded to your Notebook

# In[89]:


import pandas as pd
iris = pd.io.parsers.read_csv("iris.csv")
X = iris[["Petal.Length"]]
print (X)


# In[90]:


import pandas as pd
iris = pd.io.parsers.read_csv("iris.csv")
X = iris[['Petal.Length']].values
print (X)


# # Student Feedback

# ### Was it straightforward to install Python and all of the libraries?

# ##### Yes. It is very straight forward. Anaconda is a very easy tool that I am able to get all libraries in one go.

# ### Was the tutorial useful? Would you recommend it to others?

# #### Very useful especially the recommended book "Python for Data Science for Dummies". It would be even better if there's a toy dataset for us to work on for more python codes

# ### What are the main lessons you've learned from this experience?

# ##### I have learnt how to import different data format into the python and also how to call specific columns and create graphs

# In[ ]:




