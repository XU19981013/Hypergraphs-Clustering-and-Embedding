#--------author:ss x---------
#--------author:ss x---------

from numpy import *
from sklearn.cluster import KMeans
import numpy as np
import hypergraph_utils as hgut

import os
import  matplotlib.pyplot as plt
retval = os.getcwd()
os.chdir(retval)
print(retval)
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams['axes.unicode_minus']=False
from adjustText import adjust_text


#---------------------------------data read----------------------
X=[]
y=[]
X_name=[]
with open("./data/zoo.txt", "r") as f:
    filine = f.readlines()
for file in filine:
    file=file.split("\n")[0]
    file=file.split(",")
    X_name.append(file[0])
    X.append(np.array(list(map(int,(file[1:])))))
fts0=np.array(X)
y=np.array(y)

#---------------------------------Hypergraphe  clustering------------------
H=hgut.construct_H_with_KNN(fts0, K_neigs=[5], split_diff_scale=False, is_probH=False, m_prob=1) #construct_H
G = hgut.generate_G_from_H(H)
N=len(X)
I = np.diag(np.eye(N, N))
L=I-G           #construct_L
evals_k,e_vecs_k=hgut.topvec(L,3)
e_vecs_k1=e_vecs_k.real
X=e_vecs_k1

#---------------------------------clustering View------------------------
y_pread=KMeans(n_clusters=7).fit_predict(e_vecs_k1)
x0,x1,x2,x3,x4,x5,x6=X[y_pread==0],X[y_pread==1],X[y_pread==2],X[y_pread==3],X[y_pread==4],X[y_pread==5],X[y_pread==6]
X_name=np.array(X_name)
xname0,xname1,xname2,xname3,xname4,xname5,xname6=X_name[y_pread==0],X_name[y_pread==1],X_name[y_pread==2],X_name[y_pread==3],X_name[y_pread==4],X_name[y_pread==5],X_name[y_pread==6]
Xname=[xname0,xname1,xname2,xname3,xname4,xname5,xname6]
x=[x0,x1,x2,x3,x4,x5,x6]
for i in range(7):
    coler=["r","g","b","m","k","y","c"]
    x_=x[i]
    xname=Xname[i]
    plt.xlim(-0.2, 0.5)
    plt.ylim(-0.2,0.5)
    new_texts=[plt.text(m, n, s, c=coler[i], fontsize=10) for m,n,s in zip(np.array(x_[:,0:1]).flatten(),np.array(x_[:,1:2]).flatten(),xname)]
    adjust_text(new_texts,
            only_move={'text': 'x'},
            arrowprops=dict(arrowstyle='-', color='grey'),
            )
plt.show()