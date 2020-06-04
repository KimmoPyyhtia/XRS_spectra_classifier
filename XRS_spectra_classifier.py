# -*- coding: utf-8 -*-
"""
Pro Gradu final code
Created on Wed May  6 10:58:15 2020
@author: Kimmo
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')

import h5py
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.signal import lfilter
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import KFold
from sklearn import svm
import time



from keras import backend as K 
import tensorflow as tf
from tensorflow import keras

import pickle
import sys
sns.set()
plt.close('all')
plt.ion()
#%%
#Data manipulation methods
#def load_h5_file(filename, set_index=0):
    # Loads the data from one scan into memory based on the filename
    # the .h5 file must be in the same folder as this python-file
    # if the file contains multiple scans then the index of the scan should be
    # specified, otherwise the first scan is used as default
#    f = h5py.File(str(filename), 'r') 
 #   key=list(f.keys())
#    data=f.get(str(key[set_index]))
#    return data


def load_h5_file(filename, set_index=0, E_axis=False):
    # Loads the data from one scan into memory based on the filename
    # the .h5 file must be in the same folder as this python-file
    # if the file contains multiple scans then the index of the scan should be
    # specified, otherwise the first scan is used as default
    f = h5py.File(str(filename), 'r') 
    key=list(f.keys())
    data=f.get(str(key[set_index]))
    if E_axis:
        return data, data['energy'][()]
    else:
        return data

def pickROI(data, roi_index):
    # returns one 3-D numpy array. The axes are swapped for more intuitive order
    # here the data is visualized as a cube so that rows are y-axis, columns are
    # x-axis and the remaining axis is z-axis. I thought this so that behind each
    # pixel defined by it's y- and x-coordinates lies a spectrum on the z-axis.
    
    # Possible NaN values are also replaced by float(0.0)
    ROI=np.swapaxes(data['ROI%02d'%(roi_index)], 0,1 )#swap energy axis and y axis
    ROI=np.swapaxes(ROI,1,2,)
    return np.nan_to_num(ROI, nan=0.0)# after swapping [0,1,2] = y,x,z coordinates
def pickROI(data, roi_index, energy_axis=False):
    # returns one 3-D numpy array. The axes are swapped for more intuitive order
    # here the data is visualized as a cube so that rows are y-axis, columns are
    # x-axis and the remaining axis is z-axis. I thought this so that behind each
    # pixel defined by it's y- and x-coordinates lies a spectrum on the z-axis.
    
    # Possible NaN values are also replaced by float(0.0)
    ROI=np.swapaxes(data['ROI%02d'%(roi_index)], 0,1 )#swap energy axis and y axis
    ROI=np.swapaxes(ROI,1,2,)
    if energy_axis:
        return np.nan_to_num(ROI, nan=0.0), data['ROI%02d'%(roi_index)+'_E0']
    else:
        return np.nan_to_num(ROI, nan=0.0)# after swapping [0,1,2] = y,x,z coordinates
    
#def pickPixel(ROI, y,x): #picks a pixel based on it's coordinates and returns an array containing it's spectrum
#    return ROI[y,x,:] 

def pickPixel(ROI, y,x, E0=None): #picks a pixel based on it's coordinates and returns an array containing it's spectrum
    #returns the location of the elastic spike if it's requested
    if E0 is not None:
        return ROI[y,x,:], E0[y,x]
    else:
        return ROI[y,x,:] 

def plotPixel(ROI, y,x, image=False, label='', smoother=False, smoother_param=None):
    # plots one spectrum behind a pixel in ROI. If Image the plot is saved
    # manual labels are possible, smoothing is done by scipy lfilter for plottin purposes
    plt.figure()
    if smoother:
        plt.plot(lfilter([1/smoother_param]*smoother_param, 1, ROI[y,x,:]), label='('+str(y)+','+str(x)+')')
    else:
        plt.ylim(top=1)
        plt.ylim(bottom=0)
        plt.plot(ROI[y,x,:], label=label)
    if image:
        plt.savefig(label+str(y)+' '+str(x)+'.png')
    plt.legend(loc='upper right')
    plt.show()
    
def NormalizeData(data):
    # Works for lists or arrays. 
    # Returns a list or and array that's normalized between 0 and 1
    # For arrays finds the maximum of the entire array and uses that as 1.
    
    
    data=np.asarray(data).astype(np.float)
    data=np.nan_to_num(data) # NaN to 0.0
    maximum=np.amax(data) # maximum value of the entire list or array
    minimum=np.amin(data) # minimum value of the entire list or array
    if maximum==0:  #if all elements are 0 then this returns a list of zeros
        # Note, that if the entire 2-D array is 0 then this will not return an
        # array with proper size
        return [0]*len(data)
    else:
        return (data - minimum) / (maximum - minimum) #Normalization
    
def ROIdotP(ROI): #Dot product with normalized data
    # Returns a 2-D array with dot products calculated for each spectrum with itself.
    # If the dot product is small then the spectrum can't hold too much information
    # and is most likely a bad one. The noise becomes a too great.
    
    # The data is also normalized between 0 and 1 
    
    dim=ROI.shape[0:2]
   
    dots=np.zeros(dim,dtype=int) # empty array with same shape as ROI(y,x)
    for ind1 in range(dim[0]):
        for ind2 in range(dim[1]):
            a=NormalizeData(ROI[ind1,ind2,:])
            dots[ind1][ind2]=np.dot(a,a)
    return dots

def TrainSetCreator(data, roi, liquid_or_solid, Hard_x_axis=None, original_x_axis=None):
    # Picks one ROI based on roi and the user can classify the spectra as either
    # good or bad ones (1,0)
    dim = data['ROI%02d'%(roi)].shape[1:]
    #picks a region of interest and the E0 assosiated with it
    ROI, E0=pickROI(data, roi,energy_axis=True)

    GoodBad=np.zeros(dim,dtype=int)

    #This creates the flattened the data
    all_spectra=np.zeros((ROI.shape[0]*ROI.shape[1],ROI.shape[2] ) )
    
    count=0
    for i in range(0,ROI.shape[0]):
        for j in range(0,ROI.shape[1]):
            plt.close('all')
            #picks a spectrum and E0 from ROI
            spec, zero_point=pickPixel(ROI,i,j,E0)
            #interpolates the spectrum x-axis to the common x-axis shared by all spectra
            spec=np.interp(Hard_x_axis, original_x_axis-zero_point, spec)
            all_spectra[count]=spec
            count+=1
            
    all_spectra=NormalizeData(all_spectra) # the data is normalized for machine
    #learning purposes between 0 and 1.
    np.savetxt(liquid_or_solid+'spectra_%02d'%(roi)+'.txt',all_spectra )
    
    GoodBad=[]
    for i in range(len(all_spectra)):
        plt.close('all')
        # plots the rolling average of the spectrum
        rolling=pd.DataFrame(all_spectra[i][:])
        rolling=rolling.rolling(10).mean()
        plt.figure()
        plt.plot(rolling)
        plt.show()
        
        loop=True
        
        while (loop):
            try:
                #requests input and if it's accepted then it's recorded
                hp = np.float(input('good/bad (1/0):'))
                if (hp==1 or hp==0):
                    GoodBad.append(hp)
                    loop=False
                else:
                    print('Bad input')
            except:
                print('Bad input')
                continue
               
    np.savetxt(liquid_or_solid+'labels_%02d'%(roi)+'.txt',GoodBad)
 
def TrainSetCreator_2(data, number, filter_percentage=0.0, liquid_or_solid='', Hard_x_axis=None, original_x_axis=None):
    ##
    # Creates a training set by picking number amount of spectra from each ROI
    # to classify. The spectra are ordered based on their dot product with 
    # themselves which is an estimate of information that specific data vector
    # holds. Then n=number spectra are picked from the ordered spectra on with
    # a constant spread. The spectra are smoothed for plotting purposes only.
    #
    # For each plot the user must input either 1 or 0 based on if they deem the 
    # spectrum good or bad.
    #
    # Outputs two files, one containing 1's and 0's based on the classification
    # named "labels_ALL(num).txt" where num is the number of spectra classified
    # for each ROI. The other file named "spectra_ALL(num).txt" contains the
    # corresponding spectra in the same order.
    
    num=number
    spectra_data=[]
    spectra_labels=[]
    Energy_axis=[]
    for rois in range(0,48): #forward scattering had total of 48 ROIs,
        # backward scattering is not considered in this code
        ROI, E0=pickROI(data, rois, True) #picking one ROI based on ROI_index
        dot=ROIdotP(ROI) 
        ordered_ind=[] 
        for i in range(0, dot.shape[0]*dot.shape[1]):
            max_arg=argmax_2d(dot) # finds the indices of the maximum value in 
            # a 2-D array
            ordered_ind.append(max_arg) # The indices are appended to the list
            dot[max_arg]=-1 # sets tha maximum value to -1 and the loop is 
            #repeated until all spectra have been ordered 
            
        filter_value=int(filter_percentage*len(ordered_ind))
        ordered_ind=ordered_ind[:-filter_value or None]
        # removes the indices of the spectra with the lowest dot products based
        # the percentage gives as argument. If filter_percentage=0 then no 
        # spectra are removed
        
        ind_ind=np.round(np.linspace(0,len(ordered_ind)-1, number, dtype='int'))
        # finds n=number indices from the ordered list of dot products with even
        # spacing. If number=2 then the indices of spectra with highest and lowest
        # dot products are selected. With a larger number more spectra between
        # those are selected.
       
        
        for ind in ind_ind:
            plotPixel(ROI,ordered_ind[ind][0],ordered_ind[ind][1], smoother=True, smoother_param=7)
            # plots one pixel based on the ordered dot products and the n=number evenly
            # selected spectra. The plot is smoothened to reduce noise for the user.
            spec, zero_point= pickPixel(ROI,ordered_ind[ind][0],ordered_ind[ind][1] ,E0)
            #Interpolating each spectra to the common x-axis
            spec=np.interp(Hard_x_axis, original_x_axis-zero_point, spec)
            spectra_data.append(spec) 
            Energy_axis.append(zero_point)
            # The same data used to plot is appended to a list
            
            loop=True
            # this part asks for user input and if it's either 0 or 1 then that
            # classification is recorded. Some handling for bad input is included
            # so a bad input requires a new try on the same plot
            while (loop):
                try:
                    hp = np.int(input('good/bad (1/0):'))
                    if (hp==1 or hp==0):
                        spectra_labels.append(hp)
                        loop=False
                    else:
                        print('Bad input')
                except:
                    print('Bad input')
                    continue
    spectra_data=NormalizeData((spectra_data)) # The entire data is Normalized between
    # 0 and 1 for machine learning purposes. 1 is the maximum of the entire array,
    # not individual specta
    np.savetxt(liquid_or_solid+'labels_ALL%02d'%(num)+'_'+str(filter_percentage)+'.txt',spectra_labels, fmt='%s') 
    np.savetxt(liquid_or_solid+'spectra_ALL%02d'%(num)+'_'+str(filter_percentage)+'.txt',spectra_data)
    np.savetxt(liquid_or_solid+'E_axis_ALL%02d'%(num)+'_'+str(filter_percentage)+'.txt',Energy_axis)
   
def argmax_2d(matrix):
    #Takes in a 2-D array and returns the indices of it's maximum value
    
    y_lim=matrix.shape[0]
    x_lim=matrix.shape[1]
    b=0
    b_ind=0
    nans=np.isnan(matrix)
    matrix[nans]=0 #sets NaN to 0
    for i in range(y_lim):
        for j in range(x_lim):
            if matrix[i][j]>=b:
                b=matrix[i][j]
                b_ind=(i,j)
                
    return b_ind 
def full_data_to_txt(filename, Hard_x_axis, original_x_axis):
    # This method interpolates all data to a common x-axis and saves the 
    # result as .txt file.
    
    f = h5py.File(str(filename), 'r')
    key=list(f.keys())
    full_data_for_predict=[]
    
    
    for a in range(len(key)-1): #leaves out the rois
        full_data =f.get(key[a])
        #Only the forward scattering is considered hence the loop range  (0,48)
        for i in range(0,48):
            ROI, E0=pickROI(full_data, i,energy_axis=True)
            for j in range(ROI.shape[0]):
                for k in range(ROI.shape[1]):
                    spec, zero_point=pickPixel(ROI,j,k,E0)
                    #Interpolating each spectra to a common x-axis
                    spec=np.interp(Hard_x_axis, original_x_axis-zero_point, spec)
                    full_data_for_predict.append(spec)
    np.savetxt(str(filename[:-3])+'.txt', full_data_for_predict)
    
def generate_x_axis(data, original_x_axis):
    # generates an x-axis
    ROI, Energy_axis=pickROI(data,0,energy_axis=True)
    E0=pickPixel(ROI,0,0,Energy_axis)[1]
    return original_x_axis-E0
#%%
# SVM Methods

def load_data(code,liquid_or_solid, variant=False):
    # loads a file, by default loads the data and labels associated with it
    # using the variant only the data can be loaded for example for prediction
    if variant:
        X=np.loadtxt(code)
        dfX=pd.DataFrame(data=X).fillna(0) # replace NaN with 0
        # the lines are shuffled and turned into numpy arrays
        dfX_shuffled=dfX.sample(frac=1).reset_index(drop=True)
        dfX_shuffled=dfX_shuffled.to_numpy()
        return dfX_shuffled
    else:
        X=np.loadtxt(liquid_or_solid+'spectra_'+str(code)+'.txt')
        y=np.loadtxt(liquid_or_solid+'labels_'+str(code)+'.txt',dtype=int)
        dfX=pd.DataFrame(data=X).fillna(0)#poista NaN käsittely tästä
      
        dfY=pd.DataFrame(y)
        #concatenates the labels to the specra so that the rows can be shuffled
        #so that the labels stay associated with the corresponding spectra
        dfXY=pd.concat([dfY,dfX], axis=1).fillna(0)
        dfXY_shuffled=dfXY.sample(frac=1).reset_index(drop=True)
        # from DataFrame to Numpy
        all_labels=dfXY_shuffled.iloc[:,0].to_numpy()
        all_data=dfXY_shuffled.iloc[:,1:].to_numpy()
        return all_data, all_labels


def test_train_split(all_data, all_labels):
    # Splits the data to train data, train labels, test data and test labels.
    # Test data is 20% of the entire data.
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(all_data, all_labels, test_size=0.2)
    return X_train, X_test, y_train, y_test

def cross_validation_split(all_data, all_labels,
                           splits=5,rand_state=None, shuffled=False): #creates a cross validation data set with K splits
    #makes a split to five parts for cross validation
    KF=KFold(n_splits=splits, shuffle=shuffled, random_state=rand_state)
    KF.get_n_splits(all_data)
    
    X_train=[]
    X_test=[]
    y_train=[]
    y_test=[]
    # each time one of the folds is left out to serve as the validation set
    # the other four are trainsets
    for train_index, test_index in KF.split(all_data):
        X_train.append(all_data[train_index])
        X_test.append(all_data[test_index])
        
        y_train.append(all_labels[train_index])
        y_test.append(all_labels[test_index])
    
    return X_train, X_test, y_train, y_test

def Support_Vector_Machine(X_train, Y_train, X_test,Y_test, cost=1.0, probabilistic=True, kernel_choice='rbf', return_model=False):
    #initialize the model and fit the training data
    clf=svm.SVC(C=cost,kernel=kernel_choice,probability=probabilistic)#testaa monella eri kernelillä
    model=clf.fit(X_train, Y_train) 
    #evaluate the performance using a test set
    confidence = clf.score(X_test, Y_test) #confidence tekee saman kuin evaluate_predictions()
    #predict the test data
    predictions=clf.predict_proba(X_test)
    #if the model is needed then it can be returned as well for
    #future predictions
    if return_model:
        return predictions, confidence, model
    else:
        return predictions, confidence

def evaluate_predictions(Y_test,predictions ):
    # this gives the accuracy score of the predictions
    count=0
    for i in range(len(predictions)):
        if predictions[i]==int(Y_test[i]):
            count+=1
    return count/len(preds)

def cross_validate_SVM(training_data,training_labels, cost=1.0,
                   splits=5, shuffled=True, kernel_function='rbf'
                   ):
    #splits the data into cross validation sets    
    X_train, X_val, Y_train, Y_val=cross_validation_split(training_data, training_labels, splits, rand_state=0, shuffled=True,)
    
    cross_validation_accuracy=[]
    for i in range(len(X_train)):
        #the models is trained and tested with each split and the F1-scores are appended
        cross_validation_accuracy.append(sklearn.metrics.f1_score(Y_val[i],binary_predictions_from_probabilistic(Support_Vector_Machine(X_train[i], Y_train[i],  X_val[i],  Y_val[i], cost,kernel_choice=kernel_function)[0])))
    return(np.mean(cross_validation_accuracy)) # returns the mean of the F1-score


    
    
def learning_curve_SVM(data, labels, best_params, bins=20):
    #splits the data into training set and test sets
    X_train, X_val, Y_train, Y_val= test_train_split(data, labels)
    #the sized of the training set increment is data per bin
    data_per_bin=int(len(X_train)/bins)
    
    training_f1_score=[]
    testing_f1_score=[]
    elements_in_training=[]
    dfX_train=pd.DataFrame(data=X_train)
    dfY_train=pd.DataFrame(data=Y_train)
    #the labels are concatenated to the data and the rows are shuffled
    dfXY=pd.concat([dfY_train,dfX_train], axis=1).fillna(0)
    dfXY_shuffled=dfXY.sample(frac=1).reset_index(drop=True)
    for i in range(1,bins+1):
         #the labels and the data are separated again
        dfY_train=dfXY_shuffled.iloc[:,0].to_numpy()
        dfX_train=dfXY_shuffled.iloc[:,1:].to_numpy()
        # the endpoint of the slice
        slice_end=data_per_bin*i
        elements_in_training.append(slice_end)
        #training set size is determined by the data per bin times the number of
        # bins
        X_train_2=dfX_train[:slice_end][:]
        Y_train_2=dfY_train[:slice_end][:]
        #SVM makes predictions
        predictions, conf, model=Support_Vector_Machine(X_train_2,Y_train_2,X_train_2,Y_train_2,best_params[1],kernel_choice=best_params[2], return_model=True )

      
        training_f1_score.append(sklearn.metrics.f1_score(Y_train_2,binary_predictions_from_probabilistic(predictions)))
        test=model.predict_proba(X_val)
        
        testing_f1_score.append(sklearn.metrics.f1_score(Y_val,binary_predictions_from_probabilistic(test)))

    plt.figure()
    plt.ylim(ymin=0.4,ymax=1.05)
    plt.axhline(y=1.0, color='r')
    plt.plot(elements_in_training,training_f1_score, label='Koulutuksen F1-arvo')
    plt.plot(elements_in_training,testing_f1_score, label='Validaation F1-arvo')
    plt.legend(loc='lower right')
    plt.xlabel('Koulutusjoukon spektrien määrä')
    plt.ylabel('F1-arvo')
    plt.savefig('Learning_curve_SVM.pdf')
    plt.show() 

#this method gives true predictions 0 or 1 from probabilities such as [0.2,0.8]
def binary_predictions_from_probabilistic(predictions):
    preds=[]
    for i in range(0,len(predictions)):
            preds.append(np.argmax(predictions[i]))
    return preds

#%%
#NN methods
#source of this method is https://medium.com/@aakashgoel12/how-to-add-user-defined-function-get-f1-score-in-keras-metrics-3013f979ce0d
#calculates f1-score
def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def Neural_network(X_train, X_test, Y_train, Y_test,
                   L1,
                   L2,
                   L3,
                   activation_function='relu',
                   reg_param=0.01,
                   optimizer_param='SGD',
                   epoch_n=10,
                   ):
    #if both L2 and L3 are 0 then the model has only one layer
    if L3==0 and L2==0:
         model = keras.Sequential([
        keras.layers.Dense(L1, activation=activation_function, kernel_regularizer=keras.regularizers.l2(l=reg_param)),
        keras.layers.Dense(2, activation='softmax')
    ])
    # if L3 is 0 then the model has two layers
    elif L3==0:
        model = keras.Sequential([
            keras.layers.Dense(L1, activation=activation_function, kernel_regularizer=keras.regularizers.l2(l=reg_param)),
            keras.layers.Dense(L2, activation=activation_function, kernel_regularizer=keras.regularizers.l2(l=reg_param)),
            keras.layers.Dense(2, activation='softmax')
        ])
    #otherwise the model has three layers
    else:
         model = keras.Sequential([
        keras.layers.Dense(L1, activation=activation_function, kernel_regularizer=keras.regularizers.l2(l=reg_param)),
        keras.layers.Dense(L2, activation=activation_function, kernel_regularizer=keras.regularizers.l2(l=reg_param)),
        keras.layers.Dense(L3, activation=activation_function, kernel_regularizer=keras.regularizers.l2(l=reg_param)),
        keras.layers.Dense(2, activation='softmax')
    ])
    
    #model is compiled here. The metrics[get_f1] corrensponds to a method that
    #calculates the f1-score and the model uses that as a performance metric
    #instead of 'accuracy' or other metrics-
    model.compile(optimizer=optimizer_param,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=[get_f1])
    
    
    model.fit(X_train, Y_train, epochs=epoch_n)
    
    test_loss, test_acc = model.evaluate(X_test,  Y_test, verbose=2)
    
    print('\nTest accuracy:', test_acc)
    
    probability_model = tf.keras.Sequential([model, 
                                             tf.keras.layers.Softmax()])
    predictions = probability_model.predict(X_test)
    
    correctly_predicted=evaluate_predictions(Y_test,binary_predictions_from_probabilistic(predictions ))
    return predictions, correctly_predicted, probability_model

def cross_validate_NN(all_data,all_labels, 
                   L1,L2,L3,
                   activation_function='relu',
                   reg_param=0.01,
                   splits=5, shuffled=True
                   ):
    #cross validations plit
    X_train, X_test, Y_train, Y_test=cross_validation_split(all_data, all_labels, splits, rand_state=None, shuffled=True,)
    cross_accuracy=[]
    for i in range(len(X_train)):
        cross_accuracy.append(Neural_network(X_train[i], X_test[i], Y_train[i], Y_test[i],L1,L2,L3,activation_function=activation_function,reg_param=reg_param)[1])#index =1 if accuracy, 2 if AUC
        #clears previous weights and biases from RAM
        keras.backend.clear_session()
    return(np.mean(cross_accuracy))

def learning_curve_NN(data, labels, best_params, bins=20):
    #splits data into trainset and test set
    X_train, X_val, Y_train, Y_val= test_train_split(data, labels)
    data_per_bin=int(len(X_train)/bins)
    
    training_f1_score=[]
    testing_f1_score=[]
    elements_in_training=[]
    dfX_train=pd.DataFrame(data=X_train)
    dfY_train=pd.DataFrame(data=Y_train)
    dfXY=pd.concat([dfY_train,dfX_train], axis=1).fillna(0)
    dfXY_shuffled=dfXY.sample(frac=1).reset_index(drop=True)
    for i in range(1,bins+1):
        
        dfY_train=dfXY_shuffled.iloc[:,0].to_numpy()
        dfX_train=dfXY_shuffled.iloc[:,1:].to_numpy()
        
        slice_end=data_per_bin*i
        elements_in_training.append(slice_end)
        X_train_2=dfX_train[:slice_end][:]
        Y_train_2=dfY_train[:slice_end][:]
        
        predictions, conf, model=Neural_network(X_train_2,X_train_2,Y_train_2,Y_train_2,*best_params)
        
        training_f1_score.append(sklearn.metrics.f1_score(Y_train_2,binary_predictions_from_probabilistic(predictions)))
        test=model.predict(X_val)
        
        testing_f1_score.append(sklearn.metrics.f1_score(Y_val,binary_predictions_from_probabilistic(test)))

    plt.figure()
    plt.ylim(ymin=0.4,ymax=1.05)
    plt.axhline(y=1.0, color='r')
    plt.plot(elements_in_training,training_f1_score, label='Koulutuksen F1-arvo')
    plt.plot(elements_in_training,testing_f1_score, label='Validaation F1-arvo')
    plt.legend(loc='lower right')
    plt.xlabel('Koulutusjoukon spektrien määrä')
    plt.ylabel('F1-arvo')
    plt.savefig('Learning_curve_NN.pdf')
    plt.show()
def spectral_sum_NN(predict_data, NN_predictions,SVM_predictions, labels=False, Hard_x_axis=None ):
    plt.figure()
    #sum of all spectra
    summed_spectra=np.zeros(len(predict_data[0]))
    for i in range(len(predict_data)):
        spec=np.array(predict_data[i])
        summed_spectra=summed_spectra+spec
    allsum=summed_spectra
    plt.plot(Hard_x_axis*1000,summed_spectra+9, label='Kaikkien spektrien summa')
    
    #sum of all sample spectra based on NN classifications
    summed_spectra=np.zeros(len(predict_data[0]))
    for i in range(len(predict_data)):
        if np.argmax(NN_predictions[i])==1:
            spec=np.array(predict_data[i])
            
            summed_spectra=summed_spectra+spec
    NNsum=summed_spectra        
    np.savetxt('spectral_sum_NN_ALL.txt',NNsum)       
    #plt.ylim(top=2000)
    plt.plot(Hard_x_axis*1000,summed_spectra+6, label='Neuroverkon luokittelema')
    
    
    #sum of all sample spectra based on SVM classifications 
    summed_spectra=np.zeros(len(predict_data[0]))
    for i in range(len(SVM_predictions)):
        if SVM_predictions[i]==1:
            spec=np.array(predict_data[i])
            summed_spectra=summed_spectra+spec
    SVMsum=summed_spectra
    plt.plot(Hard_x_axis*1000,summed_spectra+3, label='Tukivektorikoneen luokittelema')
    np.savetxt('spectral_sum_SVM_ALL.txt',SVMsum)   
    #sum of all sample spectra based on manual classifications
    if labels is not False:
        summed_spectra=np.zeros(len(predict_data[0]))
        for i in range(len(predict_data)):
            if labels[i]==1:
                spec=np.array(predict_data[i])
           
                summed_spectra=summed_spectra+spec
        plt.plot(Hard_x_axis*1000,summed_spectra, label='Manuaalisesti luokiteltu')
    manualsum=summed_spectra 
    
    
    plt.xlabel('Energiahäviö [eV]')
    plt.ylabel('Intensiteetti [-]')
    plt.legend(loc='upper right')
    if labels is not False:
        plt.savefig('ROI_sums_different_models.pdf')
    else:
        plt.savefig('All_sums_SVM_NN.pdf')
    plt.show()
    plt.figure()
    if labels is not False:
        plt.plot(Hard_x_axis*1000, allsum-manualsum, label='Manuaalisesti luokiteltu')
    plt.plot(Hard_x_axis*1000, allsum-SVMsum, label='Tukivektorikoneen luokittelema')
    plt.plot(Hard_x_axis*1000, allsum-NNsum, label='Neuroverkon luokittelema')
    plt.legend(loc='upper right')
    plt.xlabel('Energiahäviö [eV]')
    plt.ylabel('Intensiteetti [-]')
    if labels is not False:
        plt.savefig('ROI_sum_difference_from_summing_all_different_models.pdf')
    else:
        plt.savefig('All_difference_SVM_NN.pdf')
    plt.show()

#%%
#TRAIN SET CREATION


liquid_or_solid='liquid_' #add underscore after the word
scan_index_of_trainset=0

#sys.stdout=open(liquid_or_solid+'.txt', 'w')
#Set true if you want to create a trainset
create_trainset=False

filter_percentage=0.4
training_number_of_spectra_per_ROI=8

one_ROI_set=[False, 10, liquid_or_solid] #True if the user wants to classify an entire ROI manually
# ROI_index=0

data, original_x_axis=load_h5_file(liquid_or_solid+'water_apr20_AI_E0.h5', set_index=scan_index_of_trainset,E_axis=True)

#hard axis is the one where all other x-axis are interpolated
Hard_x_axis=generate_x_axis(data,  original_x_axis)

if create_trainset:
    TrainSetCreator_2(data, training_number_of_spectra_per_ROI, filter_percentage, liquid_or_solid, Hard_x_axis, original_x_axis)

if one_ROI_set[0]==True:
    TrainSetCreator(data, one_ROI_set[1], one_ROI_set[2],Hard_x_axis, original_x_axis)

#%%
#SVM

SVM_file_code='ALL08_0.4'
#cost_function_params=[1,3,5,10,25,50,75,100,200,250,400,500,750,1000,2000,5000,10000]
cost_function_params=[10]
kernel_functions=[ 'rbf' ]
#kernel_functions=['linear', 'poly', 'rbf' ]
plot_learning_curve=True

#load data and labels
all_training_data, all_training_labels= load_data(SVM_file_code, liquid_or_solid)
#split into data and labels into train and test sets
X_train, X_test, Y_train, Y_test=test_train_split(all_training_data, all_training_labels)

SVM_best_params=[0,0,0]
params=[]
timing=True
for cost in cost_function_params:
    for kernel in kernel_functions:
            if timing:
                times=time.time()
            a=(cross_validate_SVM(X_train, Y_train, cost,kernel_function=kernel))
            if a>=SVM_best_params[0]:
                SVM_best_params=[a,cost, str(kernel)]
            params.append([a,cost, str(kernel)])
            if timing:
                print('Time estimate', (time.time()-times)*len(cost_function_params)*len(kernel_functions), 'seconds')
                timing=False

print('Best parameters ', SVM_best_params)

#training the model with the best hyperparameters and testing it with the test set
best_hyperparam_predictions,conf, SVM_model = Support_Vector_Machine(X_train, Y_train,X_test,Y_test,cost=SVM_best_params[1], kernel_choice=SVM_best_params[2], return_model=True)
#best_perf=evaluate_predictions(Y_test,best_hyperparam_predictions)
preds= []
for i in range(len(best_hyperparam_predictions)):
    preds.append(np.argmax(best_hyperparam_predictions[i]))
best_hyperparam_predictions=preds
best_perf=sklearn.metrics.f1_score(Y_test,best_hyperparam_predictions)
print('SVM performance of the best parameters using F1-score', best_perf)
print('SVM-Accuracy score', evaluate_predictions(Y_test, best_hyperparam_predictions))

Y_testi=pd.Series(data=Y_test, name='Todelliset luokat')
ennuste=pd.Series(data=best_hyperparam_predictions, name='Ennustetut luokat')
df_confusion = pd.crosstab(Y_testi, ennuste)
print(df_confusion)



if plot_learning_curve:
    learning_curve_SVM(all_training_data, all_training_labels, SVM_best_params, bins=50)

#%%
## Neural network
NN_file_code='ALL08_0.4'
#Nodes per layer
#L1=[600,500,450,400,350,300]
#L2=[350,300,250,200,150,100,50,0]
#L3=[150,100,75,50,25,10,0]
L1=[500]
L2=[300]
L3=[25]
#actv_function=['relu', 'sigmoid']
actv_function=['relu']
#reg_params=[10000,1000,100,10,1,0.1,0.01,0.001,0.0001]
reg_params=[0.01]
all_training_data, all_training_labels= load_data(NN_file_code, liquid_or_solid)
X_train, X_test, Y_train, Y_test=test_train_split(all_training_data, all_training_labels)



results=[]

timing=True
for elem1 in L1:
    for elem2 in L2:
        for elem3 in L3:
            for func in actv_function:
                for reg in reg_params:
                    if timing:
                        time_start=time.time()
                    arguments=[elem1,elem2,elem3, func, reg]
                    mean_performance=(cross_validate_NN(X_train, Y_train, *arguments)) ##
                    array=[mean_performance]
                    array.extend(arguments)
                    results.append(array)
                    if timing:
                        print('Time estimate: '+str((time.time()-time_start)*len(L1)*len(L2)*len(L3)*len(actv_function)*len(reg_params))+' seconds')
                        timing=False
 
results=np.array(results)

b=pd.DataFrame(data=results, columns=['F1', 'L1', 'L2', 'L3', 'actv_func', 'reg_param']).sort_values(by=['F1', 'L1', 'L2', 'L3','actv_func', 'reg_param'], ascending=False)
print('Best parameters ', b.iloc[0][1:])
best_params=b.iloc[0][1:].values.tolist()
NN_predictions,best_model_performance, NN_model=Neural_network(X_train,X_test,Y_train,Y_test, *best_params)
NN_predictions=binary_predictions_from_probabilistic(NN_predictions)
print('Best model F1-score on test set ', sklearn.metrics.f1_score(Y_test,NN_predictions))
print('SVM-Accuracy score', evaluate_predictions(Y_test, NN_predictions))


predict_data,labels=load_data('15', liquid_or_solid, variant=False)
data_norm=NormalizeData(predict_data)
NN_predict=NN_model.predict(data_norm)
SVM_predict=SVM_model.predict(data_norm)
spectral_sum_NN(predict_data, NN_predict,SVM_predict, labels, Hard_x_axis=Hard_x_axis)





#THESE LINES MUST BE EXECUTED THE FIRST TIME THE FILE IS RUN IN THE CURRENT FOLDER
full_data_to_txt('liquid_water_apr20_AI_E0.h5', Hard_x_axis, original_x_axis)
full_data_to_txt('solid_water_apr20_AI_E0.h5', Hard_x_axis, original_x_axis)


predict_data=load_data(liquid_or_solid+'water_apr20_AI_E0.txt',liquid_or_solid, variant=True)
data_norm=NormalizeData(predict_data)
NN_predict=NN_model.predict(data_norm)
SVM_predict=SVM_model.predict(data_norm)

spectral_sum_NN(predict_data, NN_predict,SVM_predict, Hard_x_axis=Hard_x_axis)




#Saving the models as pickle.objects
pickle.dump(SVM_model, open('svm_model.p', 'wb'))
#THE FILEPATH MUST BE CHANGED MANUALLY
try:
    NN_model.save('C:/Users/Kimmo/Documents/Gradu/KimmoGradu')
except:
    pass





        
  






learning_curve_NN(all_training_data, all_training_labels, best_params, bins=30)

print('Best parameters ', b.iloc[0][:]) 



# plots the different trainset size sums with the best parameters/models
def trainset_size_comparison(SVM_best_params, b, Hard_x_axis, liquid_or_solid='solid_'):
    filecode=['ALL02_0.4', 'ALL04_0.4','ALL08_0.4']
    
    for code in filecode:
        all_training_data, all_training_labels= load_data(code, liquid_or_solid)
        X_train, X_test, Y_train, Y_test=test_train_split(all_training_data, all_training_labels)
        SVM_predictions,SVM_conf, SVM_model = Support_Vector_Machine(X_train, Y_train,X_test,Y_test,cost=SVM_best_params[1], kernel_choice=SVM_best_params[2], return_model=True)
    
        best_params=b.iloc[0][1:].values.tolist()
        NN_predictions,best_model_performance, NN_model=Neural_network(X_train,X_test,Y_train,Y_test, *best_params)
    
    
   
        predict_data,labels=load_data('15', liquid_or_solid, variant=False)
        data_norm=NormalizeData(predict_data)
    
        NN_predict=NN_model.predict(data_norm)
        SVM_predict=SVM_model.predict(data_norm)

        summed_spectra=np.zeros(len(predict_data[0]))
        for i in range(len(predict_data)):
            if np.argmax(NN_predict[i])==1:
                spec=np.array(predict_data[i])
                
                summed_spectra=summed_spectra+spec
        
        plt.plot(Hard_x_axis*1000,summed_spectra, label='Neuroverkko ' +str(len(all_training_labels))+ ' spektriä')
       
        summed_spectra=np.zeros(len(predict_data[0]))
        for i in range(len(predict_data)):
            if (SVM_predict[i])==1:
                spec=np.array(predict_data[i])
                
                summed_spectra=summed_spectra+spec
        plt.plot(Hard_x_axis*1000,summed_spectra+7, label='Tukivektorikone ' +str(len(all_training_labels))+ ' spektriä')
        plt.legend(loc='upper right')
        plt.xlabel('Energiahäviö [eV]')
        plt.ylabel('Intensiteetti [-]')
        plt.ylim(top=32)
    plt.savefig('Train_set_comparison.pdf')   
    plt.show()
   
plt.figure()


trainset_size_comparison(SVM_best_params, b, Hard_x_axis, liquid_or_solid)






























