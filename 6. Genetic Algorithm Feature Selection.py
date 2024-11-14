# %% [markdown]
# # Genetic Algorithm Feature Selection
# 
# ### This file has the feature selection process based on Genetic Algorithm.
# 
# 1. Load the extracted features from the spreadsheet.
# 2. The parameters for fitness iterations are declared.
# 3. The binary converted input is sent to the iteration.
# 4. The input is checked for the strength of the features.
# 5. Repeat for maximum iterations.
# 6. The fitness is compared and the most fit feature is selected.
# 7. Most fit features are chosen for parent selection.
# 8. The parents selected are subjected to mating to enhance feature quality.
# 9. Crossover and mutation counts and their rates are recorded as well.
# 10. The results of crossover and mutations are merged.
# 11. The fitness if enhanced is recorded, stored and updated.
# 12. The best feature subset dictionary is created.
# 13.  Repeat for other selected parents.
# 14.  Plot the accuracy vs crossover graph and loss vs mutation graph.

# %% [markdown]
# Import dependencies

# %%
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np 
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# %% [markdown]
# Display the dataset

# %%
df = pd.read_csv(r'/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/Audio_Features.csv')
df.dataframeName = 'Audio_Features.csv'
df.head(5)

# %% [markdown]
# Define methods needed to perform feature selection

# %%
def error_rate(xtrain, ytrain, x, opts):
    k     = opts['k']
    fold  = opts['fold']
    xt    = fold['xt']
    yt    = fold['yt']
    xv    = fold['xv']
    yv    = fold['yv']
    
    num_train = np.size(xt, 0)
    num_valid = np.size(xv, 0)
    xtrain  = xt[:, x == 1]
    ytrain  = yt.reshape(num_train)  
    xvalid  = xv[:, x == 1]
    yvalid  = yv.reshape(num_valid)     

    mdl     = KNeighborsClassifier(n_neighbors = k)
    mdl.fit(xtrain, ytrain)
    ypred   = mdl.predict(xvalid)
    acc     = np.sum(yvalid == ypred) / num_valid
    error   = 1 - acc
    
    return error


def Fun(xtrain, ytrain, x, opts):
    alpha    = 0.99
    beta     = 1 - alpha
    max_feat = len(x)
    num_feat = np.sum(x == 1)
    if num_feat == 0:
        cost  = 1
    else:
        error = error_rate(xtrain, ytrain, x, opts)
        cost  = alpha * error + beta * (num_feat / max_feat)
        
    return cost

# %%
import numpy as np
from numpy.random import rand


def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand()        
    
    return X


def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i,d] > thres:
                Xbin[i,d] = 1
            else:
                Xbin[i,d] = 0
    
    return Xbin


def roulette_wheel(prob):
    num = len(prob)
    C   = np.cumsum(prob)
    P   = rand()
    for i in range(num):
        if C[i] > P:
            index = i;
            break
    
    return index


def jfs(xtrain, ytrain, opts):
    ub       = 1
    lb       = 0
    thres    = 0.5    
    CR       = 0.8     
    MR       = 0.01   
    N        = opts['N']
    max_iter = opts['T']
    if 'CR' in opts:
        CR   = opts['CR'] 
    if 'MR' in opts: 
        MR   = opts['MR']  
 
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
        
    X     = init_position(lb, ub, N, dim)
    
    X     = binary_conversion(X, thres, N, dim)
    
    fit   = np.zeros([N, 1], dtype='float')
    Xgb   = np.zeros([1, dim], dtype='int')
    fitG  = float('inf')
    
    for i in range(N):
        fit[i,0] = Fun(xtrain, ytrain, X[i,:], opts)
        if fit[i,0] < fitG:
            Xgb[0,:] = X[i,:]
            fitG     = fit[i,0]
    
    curve = np.zeros([1, max_iter], dtype='float')
    t     = 0
    
    curve[0,t] = fitG.copy()
    print("Generation:", t + 1)
    print("Best (GA):", curve[0,t])
    t += 1
    
    while t < max_iter:
        inv_fit = 1 / (1 + fit)
        prob    = inv_fit / np.sum(inv_fit) 
 
        Nc = 0
        for i in range(N):
            if rand() < CR:
                Nc += 1
              
        x1 = np.zeros([Nc, dim], dtype='int')
        x2 = np.zeros([Nc, dim], dtype='int')
        for i in range(Nc):
            k1      = roulette_wheel(prob)
            k2      = roulette_wheel(prob)
            P1      = X[k1,:].copy()
            P2      = X[k2,:].copy()
            index   = np.random.randint(low = 1, high = dim-1)
            x1[i,:] = np.concatenate((P1[0:index] , P2[index:]))
            x2[i,:] = np.concatenate((P2[0:index] , P1[index:]))
            for d in range(dim):
                if rand() < MR:
                    x1[i,d] = 1 - x1[i,d]
                    
                if rand() < MR:
                    x2[i,d] = 1 - x2[i,d]
            Xnew = np.concatenate((x1 , x2), axis=0)
        
    
        Fnew = np.zeros([2 * Nc, 1], dtype='float')
        for i in range(2 * Nc):
            Fnew[i,0] = Fun(xtrain, ytrain, Xnew[i,:], opts)
            if Fnew[i,0] < fitG:
                Xgb[0,:] = Xnew[i,:]
                fitG     = Fnew[i,0]
                   
        curve[0,t] = fitG.copy()
        print("Generation:", t + 1)
        print("Best (GA):", curve[0,t])
        t += 1
        
        XX  = np.concatenate((X , Xnew), axis=0)
        FF  = np.concatenate((fit , Fnew), axis=0)
        ind = np.argsort(FF, axis=0)
        for i in range(N):
            X[i,:]   = XX[ind[i,0],:]
            fit[i,0] = FF[ind[i,0]]
       
            
    Gbin       = Xgb[0,:]
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    ga_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    
    return ga_data 
            

# %% [markdown]
# Run the genetic algorithm to view the features selected and to see fitness vs iterations graph

# %%
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data  = df
data  = data.values
feat  = np.asarray(data[:, 0:-1])
label = np.asarray(data[:, -1])
print(label)

xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.3, stratify=label)
fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}


k    = 5   
N    = 10   
T    = 100   
CR   = 0.8
MR   = 0.01
opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'CR':CR, 'MR':MR}

fmdl = jfs(feat, label, opts)
sf   = fmdl['sf']
print(sf)

num_train = np.size(xtrain, 0)
num_valid = np.size(xtest, 0)
x_train   = xtrain[:, sf]
y_train   = ytrain.reshape(num_train)  
x_valid   = xtest[:, sf]
y_valid   = ytest.reshape(num_valid)  

mdl       = KNeighborsClassifier(n_neighbors = k) 
mdl.fit(x_train, y_train)

y_pred    = mdl.predict(x_valid)
Acc       = np.sum(y_valid == y_pred)  / num_valid
print("Accuracy:", 100 * Acc)

num_feat = fmdl['nf']
print("Feature Size:", num_feat)

curve   = fmdl['c']
curve   = curve.reshape(np.size(curve,1))
x       = np.arange(0, opts['T'], 1.0) + 1.0

fig, ax = plt.subplots()
ax.plot(x, curve, 'o-')
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Fitness')
ax.set_title('GA')
ax.grid()
plt.show()

# %% [markdown]
# The chosen features are : [ 0  2  5  7  8 12 34 40 42 43 46 54]. These are moved to a seperate dataset to perform the final classification using SVM


