#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


np.random.seed(42)


# # Load train and test data

# In[4]:


# assume mnist-original.mat is in the working directory
mnist = loadmat("mnist-original.mat")


# In[5]:


X,y = (np.array(mnist["data"]).T, np.array(mnist["label"]))


# In[6]:


Xtrain0, ytrain0 = X[:60000,:], y[0,:60000]
Xtest, ytest = X[60000:,:], y[0,60000:]


# In[7]:


# shuffle the training set
shuffle_index = np.random.permutation(60000)
Xtrain1, ytrain1 = Xtrain0[shuffle_index,:],ytrain0[shuffle_index]


# In[8]:


Xtrain,ytrain = Xtrain1[:48000,:], ytrain1[:48000]
Xval,yval = Xtrain1[48000:,:], ytrain1[48000:]


# In[9]:


# test that the code is working
some_index=13286
print("Label: {:.0f}".format(ytrain[some_index]))
plt.imshow(Xtrain[some_index].reshape(28,28),
           cmap=matplotlib.cm.binary,interpolation="nearest")


# # PCA

# In[10]:


pca = PCA(n_components=100).fit(Xtrain)


# In[11]:


V=pca.inverse_transform(pca.transform([Xtrain[some_index]]))


# In[12]:


plt.imshow(V.reshape(28,28),
           cmap=matplotlib.cm.binary,interpolation="nearest")


# # SGDClassifier

# In[13]:


sgdclf = SGDClassifier(loss="hinge",alpha=0.0001,n_jobs=-1,tol=1e-3,max_iter=1000)


# In[14]:


sgdclf.fit(Xtrain,ytrain)


# In[34]:


print("Training set score: {0:.4f}".format(sgdclf.score(Xtrain,ytrain)))


# In[35]:


print("Validation set score: {0:.4f}".format(sgdclf.score(Xval,yval)))


# # RandomForestClassifier

# In[17]:


forestclf = RandomForestClassifier(n_estimators=75,n_jobs=-1)


# In[18]:


forestclf.fit(Xtrain,ytrain)


# In[32]:


print("Training set score: {0:.4f}".format(forestclf.score(Xtrain,ytrain)))


# In[33]:


print("Validation set score: {0:.4f}".format(forestclf.score(Xval,yval)))


# # RandomForestClassifier with expanded training data

# Expand the training data to include all one-pixel shifts (left, right, up, down) of each image in <tt>Xtrain</tt>.

# In[21]:


def shiftlr(im,numpixels,fill_value):
    blank = 0*im+fill_value
    index0 = im.shape[1]
    index1 = 2*im.shape[1]
    bigim = np.append(np.append(blank,im,axis=1),blank,axis=1)
    return bigim[:,index0+numpixels:index1+numpixels]

def shiftud(im,numpixels,fill_value):
    return shiftlr(im.T,numpixels,fill_value).T

maxindex = Xtrain.shape[0]
for i in range(maxindex):
    im0 = Xtrain[i].reshape(28,28)
    im1 = shiftlr(im0,1,0).reshape(784)
    im2 = shiftlr(im0,-1,0).reshape(784)
    im3 = shiftud(im0,1,0).reshape(784)
    im4 = shiftud(im0,-1,0).reshape(784)
    ytrain = np.append(ytrain,[ytrain[i],ytrain[i],ytrain[i],ytrain[i]],axis=0)
    Xtrain = np.append(Xtrain,[im1,im2,im3,im4],axis=0)
    if (i%1000 == 0):
        print(i)


# In[25]:


forest_ext_clf = RandomForestClassifier(n_estimators=75,n_jobs=-1)


# In[27]:


forest_ext_clf.fit(Xtrain,ytrain)


# In[31]:


print("Training set score: {0:.4f}".format(forest_ext_clf.score(Xtrain,ytrain)))


# In[30]:


print("Validation set score: {0:.4f}".format(forest_ext_clf.score(Xval,yval)))


# # Performance on the test set

# So far we have only been working on the training and validation sets. The data suggests that the <tt>RandomForestClassifier</tt> trained on the <em>expanded training set</em> (consisting of the original training set plus one-pixel shifts in all four directions) has the best performance, with 97.8% accuracy on the validation set. Note that the data is fairly well-balanced between the digits 0 through 9, so the accuracy can be considered meaningful in this context.

# Let us now check whether the <tt>RandomForestClassifier</tt> trained on the expanded training set performs well on the test set (which we have not touched til now).

# In[36]:


print("Test set score: {0:.4f}".format(forest_ext_clf.score(Xtest,ytest)))


# The model performs slightly better on the test set than on the validation set.

# In[ ]:




