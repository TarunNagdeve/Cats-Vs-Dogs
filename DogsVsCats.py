from keras.models import model_from_json
model = model_from_json(open("you.json", "r").read())
model.load_weights('you.h5')
import cv2
import  os
import numpy as np
import  pandas as pd
train=pd.read_csv(r'C:\Users\TARUN\Desktop\New folder (3)\Data\Train.csv')
loc=r'C:\Users\TARUN\Desktop\New folder (3)\Data\Train\Cats'

features=[]
labels=[]
f=os.listdir(loc)

for i in f:
    df=cv2.imread(os.path.join(loc,i))
    dfr=cv2.resize(df,(100,100))
    for j in range(0,4724):
        features.append(dfr)

for i in range(0,4724):
    labels.append(0)

loc1=r'C:\Users\TARUN\Desktop\New folder (3)\Data\Train\Dogs'
f1=os.listdir(loc1)


for i in f1:
    rt=cv2.imread(os.path.join(loc1,i))
    rt1=cv2.resize(rt,(100,100))
    features.append(rt1)
for i in range(0,4724):
    labels.append(1)
set=[]
for i,j in zip(features,labels):
    set.append([i,j])
import random
(random.shuffle(set))
Labels=[]
Features=[]
for i in set:
    Labels.append(i[1])
    Features.append(i[0])
Features=np.array(Features)/255

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(Features,Labels,test_size=0.2,random_state=101)
ypred=model.predict(xtest)
hey=[]
for i in ypred:
    hey.append(i)
lo=[]
for i in hey:
    lo.append(np.argmax(i))
from sklearn.metrics import  classification_report
print(classification_report(ytest,lo))