import sys
import random

import numpy as np
import pandas as pd
import copy

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import itertools as it


###############################################################################
############################### Label Flipping ################################
###############################################################################

def attack_label_flipping(X_train, X_test, y_train, y_test, model_type, n):
    # TODO: You need to implement this function!
    # You may want to use copy.deepcopy() if you will modify data
    result = 0
    for _ in range(100):
        modified_labels = copy.deepcopy(y_train)
        indexes = []
        range_n = int(len(y_train) * n)
        for _ in range(range_n):
            index = np.random.randint(0, len(y_train))
            if index not in indexes:
                indexes.append(index)
            else:
                while index in indexes:
                    index = np.random.randint(0, len(y_train)) 
            if modified_labels[index] == 0:
                modified_labels[index] = 1
            else:
                modified_labels[index] = 0
        if model_type == "DT":
            myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
            myDEC.fit(X_train, modified_labels)
            DEC_predict = myDEC.predict(X_test)
            result += accuracy_score(y_test, DEC_predict)
            
        elif model_type == "LR":
            myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=1000)
            myLR.fit(X_train, modified_labels)
            LR_predict = myLR.predict(X_test)
            result += accuracy_score(y_test, LR_predict)
        elif model_type == "SVC":
            mySVC = SVC(C=0.5, kernel='poly', random_state=0,probability=True)
            mySVC.fit(X_train, modified_labels)
            SVC_predict = mySVC.predict(X_test)
            result += accuracy_score(y_test, SVC_predict)
    return result / 100.0

###############################################################################
############################## Inference ########################################
###############################################################################

def inference_attack(trained_model, samples, t):
    # TODO: You need to implement this function!  
    count = 0
    included = 0
    predictions = trained_model.predict_proba(samples)
    for prediction in predictions:
        confidence = max(prediction)
        if confidence >= t:
            included += 1
        count += 1
    return included / count    

###############################################################################
################################## Backdoor ###################################
###############################################################################

def backdoor_attack(X_train, y_train, model_type, num_samples):    
    # TODO: You need to implement this function!
    # You may want to use copy.deepcopy() if you will modify data
    X_modified = copy.deepcopy(X_train)
    Y_modified = copy.deepcopy(y_train)
    result = 0
    for i in range(num_samples):
        injected = np.zeros(X_train.shape[1])
        for j in range(injected.shape[0]):
            if j == 0:
                injected[j] = np.random.uniform(45,60,1)
                continue
            if j == 2:
                injected[j] = np.random.uniform(30,50,1)
                continue
            injected[j] = np.random.uniform(0,10,1)
            
            
        X_modified = np.vstack([X_modified, injected])
        Y_modified = np.hstack([Y_modified, 1])
        
    y_test = np.ones(100)
    tests = []
    for i in range(100):
        test = np.zeros(X_train.shape[1])
        for j in range(test.shape[0]):
            if i < 100:
                if j == 0:
                    test[j] = np.random.uniform(45,50,1)
                    continue
                if j == 2:
                    test[j] = np.random.uniform(30,40,1)
                    continue
                if j == 1 or j == 4 or j == 5 or j == 6 or j == 8:
                    test[j] = np.random.uniform(40,50,1)
                    continue
                test[j] = np.random.uniform(0,20,1)
        tests.append(test)
        
    if model_type == "DT":
        myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
        myDEC.fit(X_modified, Y_modified)
        DEC_predict = myDEC.predict(tests)
        result = accuracy_score(y_test, DEC_predict)
    elif model_type == "LR":
        myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=1000)
        myLR.fit(X_modified, Y_modified)
        LR_predict = myLR.predict(tests)
        result = accuracy_score(y_test, LR_predict)
    elif model_type == "SVC":
        mySVC = SVC(C=0.5, kernel='poly', random_state=0,probability=True)
        mySVC.fit(X_modified, Y_modified)
        SVC_predict = mySVC.predict(tests)
        result = accuracy_score(y_test, SVC_predict)
    return result



###############################################################################
############################## Evasion ########################################
###############################################################################

def evade_model(trained_model, actual_example):
    # TODO: You need to implement this function!
    actual_class = trained_model.predict([actual_example])[0]
    modified_example = copy.deepcopy(actual_example)
    pred_class = actual_class
    lvl = 1
    while pred_class == actual_class:
        for comb in it.combinations(range(len(modified_example)), lvl):
            amount = 1.0
            for iteration in range(5000):
                index = 0
                for idx in comb:    
                    distribution = give_distribution(len(comb))
                    if actual_class == 0:
                        modified_example[idx] += amount * distribution[index]
                    else:
                        modified_example[idx] -= amount * distribution[index]
                    index += 1
                pred_class = trained_model.predict([modified_example])[0]
                if pred_class != actual_class:
                    break
                modified_example = copy.deepcopy(actual_example)
                amount += 0.05
        lvl += 1
    return modified_example

def give_distribution(lenght):
    result = []
    if lenght == 1:
        return [1.0]
    for i in range(lenght):
        result.append(np.random.uniform(0,1,1))
    s = sum(result)
    result = [p/s for p in result]
    return result

def calc_perturbation(actual_example, adversarial_example):
    # You do not need to modify this function.
    if len(actual_example) != len(adversarial_example):
        print("Number of features is different, cannot calculate perturbation amount.")
        return -999
    else:
        tot = 0.0
        for i in range(len(actual_example)):
            tot = tot + abs(actual_example[i]-adversarial_example[i])
        return tot/len(actual_example)

###############################################################################
############################## Transferability ################################
###############################################################################

def evaluate_transferability(DTmodel, LRmodel, SVCmodel, actual_examples):
    # TODO: You need to implement this function!
    DT_LR_count = 0
    DT_SVC_count = 0
    LR_SVC_count = 0
    LR_DT_count = 0
    SVC_LR_count = 0
    SVC_DT_count = 0
    print("Takes 2-3 mins")
    for example in actual_examples:
        evasion_DT = evade_model(DTmodel, example)
        evasion_LR = evade_model(LRmodel, example)
        evasion_SVC = evade_model(SVCmodel, example)
        if DTmodel.predict([evasion_DT])[0] == LRmodel.predict([evasion_DT])[0]:
            DT_LR_count += 1
        if DTmodel.predict([evasion_DT])[0] == SVCmodel.predict([evasion_DT])[0]:
            DT_SVC_count += 1
        if LRmodel.predict([evasion_LR])[0] == SVCmodel.predict([evasion_LR])[0]:
            LR_SVC_count += 1
        if LRmodel.predict([evasion_LR])[0] == DTmodel.predict([evasion_LR])[0]:
            LR_DT_count += 1
        if SVCmodel.predict([evasion_SVC])[0] == LRmodel.predict([evasion_SVC])[0]:
            SVC_LR_count += 1
        if SVCmodel.predict([evasion_SVC])[0] == DTmodel.predict([evasion_SVC])[0]:
            SVC_DT_count += 1
    print(len(actual_examples), " Evasion data trained for DT evaded LR ", DT_LR_count ,"# of times")
    print(len(actual_examples), " Evasion data trained for DT evaded SVC ", DT_SVC_count ,"# of times")
    print(len(actual_examples), " Evasion data trained for LR evaded DT ", LR_DT_count ,"# of times")
    print(len(actual_examples), " Evasion data trained for LR evaded SVC ", LR_SVC_count ,"# of times")
    print(len(actual_examples), " Evasion data trained for SVC evaded DT ", SVC_DT_count ,"# of times")
    print(len(actual_examples), " Evasion data trained for SVC evaded LR ", SVC_LR_count ,"# of times")
    print("Here, you need to conduct some experiments related to transferability and print their results...")


###############################################################################
########################## Model Stealing #####################################
###############################################################################

def steal_model(remote_model, model_type, examples):
    # TODO: You need to implement this function!
    # This function should return the STOLEN model, but currently it returns the remote model
    # You should change the return value once you have implemented your model stealing attack
    return remote_model
    

###############################################################################
############################### Main ##########################################
###############################################################################

## DO NOT MODIFY CODE BELOW THIS LINE. FEATURES, TRAIN/TEST SPLIT SIZES, ETC. SHOULD STAY THIS WAY. ## 
## JUST COMMENT OR UNCOMMENT PARTS YOU NEED. ##

def main():
    data_filename = "forest_fires.csv"
    features = ["Temperature","RH","Ws","Rain","FFMC","DMC","DC","ISI","BUI","FWI"]
    
    df = pd.read_csv(data_filename)
    df = df.dropna(axis=0, how='any')
    df["DC"] = df["DC"].astype('float64')
    y = df["class"].values
    y = LabelEncoder().fit_transform(y)    
    X = df[features].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0) 

    # Model 1: Decision Tree
    myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
    myDEC.fit(X_train, y_train)
    DEC_predict = myDEC.predict(X_test)
    print('Accuracy of decision tree: ' + str(accuracy_score(y_test, DEC_predict)))
    
    # Model 2: Logistic Regression
    myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=1000)
    myLR.fit(X_train, y_train)
    LR_predict = myLR.predict(X_test)
    print('Accuracy of logistic regression: ' + str(accuracy_score(y_test, LR_predict)))
    
    # Model 3: Support Vector Classifier
    mySVC = SVC(C=0.5, kernel='poly', random_state=0,probability=True)
    mySVC.fit(X_train, y_train)
    SVC_predict = mySVC.predict(X_test)
    print('Accuracy of SVC: ' + str(accuracy_score(y_test, SVC_predict)))

    

    # Label flipping attack executions:
    model_types = ["DT", "LR", "SVC"]
    n_vals = [0.05, 0.10, 0.20, 0.40]
    for model_type in model_types:
        for n in n_vals:
            #acc = attack_label_flipping(X_train, X_test, y_train, y_test, model_type, n)
            #print("Accuracy of poisoned", model_type, str(n), ":", acc)
            pass
    
    # Inference attacks:
    samples = X_train[0:100]
    t_values = [0.99,0.98,0.96,0.8,0.7,0.5]
    for t in t_values:
        print("Recall of inference attack", str(t), ":", inference_attack(mySVC,samples,t))
    
    # Backdoor attack executions:
    counts = [0, 1, 3, 5, 10]
    for model_type in model_types:
        for num_samples in counts:
            success_rate = backdoor_attack(X_train, y_train, model_type, num_samples)
            print("Success rate of backdoor:", success_rate, "model_type:", model_type, "num_samples:", num_samples)
    '''
    #Evasion attack executions:
    trained_models = [myDEC, myLR, mySVC]
    model_types = ["DT", "LR", "SVC"] 
    num_examples = 40
    print("It takes 2-3mins to find solution")
    for a,trained_model in enumerate(trained_models):
        total_perturb = 0.0
        for i in range(num_examples):
            actual_example = X_test[i]
            adversarial_example = evade_model(trained_model, actual_example)
            if trained_model.predict([actual_example])[0] == trained_model.predict([adversarial_example])[0]:
                print("Evasion attack not successful! Check function: evade_model.")
            perturbation_amount = calc_perturbation(actual_example, adversarial_example)
            total_perturb = total_perturb + perturbation_amount
        print("Avg perturbation for evasion attack using", model_types[a] , ":" , total_perturb/num_examples)
'''
    
    # Transferability of evasion attacks:
    trained_models = [myDEC, myLR, mySVC]
    num_examples = 40
    evaluate_transferability(myDEC, myLR, mySVC, X_test[0:num_examples])
    
    # Model stealing:
    budgets = [8, 12, 16, 20, 24]
    for n in budgets:
        print("******************************")
        print("Number of queries used in model stealing attack:", n)
        stolen_DT = steal_model(myDEC, "DT", X_test[0:n])
        stolen_predict = stolen_DT.predict(X_test)
        print('Accuracy of stolen DT: ' + str(accuracy_score(y_test, stolen_predict)))
        stolen_LR = steal_model(myLR, "LR", X_test[0:n])
        stolen_predict = stolen_LR.predict(X_test)
        print('Accuracy of stolen LR: ' + str(accuracy_score(y_test, stolen_predict)))
        stolen_SVC = steal_model(mySVC, "SVC", X_test[0:n])
        stolen_predict = stolen_SVC.predict(X_test)
        print('Accuracy of stolen SVC: ' + str(accuracy_score(y_test, stolen_predict)))
    

if __name__ == "__main__":
    main()
