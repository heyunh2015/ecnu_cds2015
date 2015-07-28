import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import os,re,operator



def loadDataSet2(filename):
    dataMat=[]; labelMat=[];
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split(' ')
        dataMat.append([float(lineArr[3]),float(lineArr[4]),float(lineArr[6]),float(lineArr[7]),float(lineArr[8]),float(lineArr[9]),float(lineArr[10]),float(lineArr[11])])
        #dataMat.append([float(lineArr[4]),float(lineArr[5]),float(lineArr[6]),float(lineArr[7]),float(lineArr[8]),float(lineArr[9])])
        #dataMat.append([float(lineArr[8]),float(lineArr[9]),float(lineArr[12]),float(lineArr[13]),float(lineArr[14]),float(lineArr[15].replace('\n',''))])
        #if int(lineArr[3])==0:
        if int(lineArr[12])==0:
            labelMat.append(0)
        else:
            labelMat.append(1)
    return dataMat,labelMat

def loadDataSet_classify(filename):
    dataMat=[]; labelMat=[];
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split(' ')
        dataMat.append([float(lineArr[5]),float(lineArr[6]),float(lineArr[7]),float(lineArr[8]),float(lineArr[9]),float(lineArr[10])])
        #dataMat.append([float(lineArr[8]),float(lineArr[9]),float(lineArr[12]),float(lineArr[13]),float(lineArr[14]),float(lineArr[15].replace('\n',''))])
        if int(str(lineArr[11].replace('\n','')))==0:
            labelMat.append(0)
        else:
            labelMat.append(1)
    return dataMat,labelMat

def loadDataSet_classify2(filename):
    dataMat=[]; #labelMat=[];
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split(' ')
        dataMat.append([float(lineArr[3]),float(lineArr[4]),float(lineArr[6]),float(lineArr[7]),float(lineArr[8]),float(lineArr[9]),float(lineArr[10]),float(lineArr[11])])
        #dataMat.append([float(lineArr[8]),float(lineArr[9]),float(lineArr[12]),float(lineArr[13]),float(lineArr[14]),float(lineArr[15].replace('\n',''))])
        #if int(str(lineArr[11].replace('\n','')))==0:
         #   labelMat.append(0)
        #else:
         #   labelMat.append(1)
    return dataMat#,labelMat

def svm_train(X_train,y_train,X_test,y_test):
    clf = svm.SVC(C=3.0, cache_size=20, class_weight=None, coef0=0.0, degree=3,
    gamma=0.1, kernel='rbf', max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False)
    #clf = svm.SVC(kernel='rbf')poly
    clf.fit(X_train, y_train)  
    print 'train done'
    result=clf.predict(X_test)
    return result,calculatePrecision(result,y_test)

def random_forest_classify(X_train,y_train,X_test,y_test):
    clf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=1, random_state=0)
    clf = clf.fit(X_train,y_train)
    result=clf.predict(X_test)
    #print clf.feature_importances_
    return result,calculatePrecision(result,y_test)     

def random_forest_classify2(X_train,y_train,X_test):
    clf = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0)
    clf = clf.fit(X_train,y_train)
    result=clf.predict(X_test)
    #print clf.feature_importances_
    return result#,calculatePrecision(result,y_test)   

def adaboost(X_train,y_train,X_test,y_test):
    clf = AdaBoostClassifier(n_estimators=20)
    clf = clf.fit(X_train,y_train)
    result=clf.predict(X_test)
    #print clf.feature_importances_
    return result,calculatePrecision(result,y_test)    

def logisticRegression(X_train,y_train,X_test,y_test):
    clf = LogisticRegression(penalty='l2', dual=False, tol=0.1, C=3.0, fit_intercept=True, intercept_scaling=1, class_weight=None, 
                             random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0)
    clf.fit(X_train, y_train)
    result=clf.predict(X_test)
    return result,calculatePrecision(result,y_test)

def calculatePrecision(result,stand_result):
    right_count=0
    for i in range(len(result)):
        if result[i]==stand_result[i]:
            right_count+=1
    return right_count*1.0/len(result)

def classify_cross(X_train,y_train,X_classify,y_classify):
    result,precision=random_forest_classify(X_train, y_train,X_classify,y_classify)
    #result,precision=svm_train(X_train,y_train,X_classify,y_classify)
    #result,precision=adaboost(X_train,y_train,X_classify,y_classify)
    #result,precision=logisticRegression(X_train,y_train,X_classify,y_classify)  
    #print precision
    return result

def classify(train_file,classify_file,result_file_name):
    X_train,y_train=loadDataSet2(train_file)
    #X_classify,y_classify=loadDataSet_classify(classify_file)
    X_classify=loadDataSet_classify2(classify_file)
    print 'loaded data'
    #result,precision=random_forest_classify(X_train, y_train,X_classify,y_classify)
    result=random_forest_classify2(X_train, y_train,X_classify)
    #result,precision=svm_train(X_train,y_train,X_classify,y_classify)
    #result,precision=adaboost(X_train,y_train,X_classify,y_classify)
    #result,precision=logisticRegression(X_train,y_train,X_classify,y_classify)
    #print precision
    #result_statistic(result, y_classify)  
    fp=open(classify_file)
    result_file=''
    lines=fp.readlines()
    i=0
    for line in lines: 
        result_file+=str(line).replace('\n','')+' '+str(result[i])+'\n'
        i+=1
        if i%1000==0:
            print i
    fp_write=open(result_file_name,'w')
    fp_write.write(result_file)  

def result_statistic(result,y_classify):
    count_1=0
    for item in result:
        if str(item)=='1':
            count_1+=1
    #print count_1  
    
    count_1_1=0
    for i in range(len(result)):
        if result[i]==y_classify[i] and str(result[i])=='1':
            #print result[i],y_classify[i]
            count_1_1+=1
    #print count_1_1
    
    count_1_2=0
    for i in range(len(result)):
        if result[i]==y_classify[i]:
            #print result[i],y_classify[i]
            count_1_2+=1
    #print float(count_1_2*1.0/len(result))      
    return count_1,count_1_1,float(count_1_2*1.0/len(result))

def construct_dic(filename):
    sample_dic={}
    fp=open(filename)
    lines=fp.readlines()
    for line in lines:
        lineArr=line.strip().split(' ')
        if lineArr[0] not in sample_dic: 
            sample_dic[lineArr[0]]={}
        if lineArr[2] not in sample_dic[lineArr[0]]:
            sample_dic[lineArr[0]][lineArr[2]]=[lineArr[0],lineArr[3],lineArr[4],lineArr[6],lineArr[7],lineArr[8],lineArr[9],lineArr[10],lineArr[11],lineArr[12]]        
    return sample_dic

def m_fold(sample_dic,n,m):
    test_query=[]
    train_query=[]
    
    X_train=[]
    y_train=[]
    X_classify=[]
    y_classify=[]
    
    for i in sample_dic:
        if int(i)%m==n:
            test_query.append(sample_dic[i])
        else:
            train_query.append(sample_dic[i])   
    
    for item in test_query:
        for i in item:
            X_classify.append([float(item[i][2]),float(item[i][4]),float(item[i][6]),float(item[i][8])])#float(item[i][3]),float(item[i][4]),float(item[i][5]),float(item[i][6]),float(item[i][7]),float(item[i][8])
            if int(str(item[i][9]))==0:
                y_classify.append(0)
            else:
                y_classify.append(1)
    
    for item in train_query:
        for i in item:
            X_train.append([float(item[i][2]),float(item[i][4]),float(item[i][6]),float(item[i][8])])
            if int(str(item[i][9]))==0:
                y_train.append(0)
            else:
                y_train.append(1)

    relevance_result=classify_cross(X_train, y_train, X_classify, y_classify)
    right_judge,right_count,precision=result_statistic(relevance_result, y_classify)
    
    count=0
    result_txt=''
    for item in test_query:
        for i in item:
            result_txt+=item[i][0]+' '+'Q0'+' '+str(i)+' '+item[i][1]+' '+item[i][2]+' '+'ECNU'+' '+item[i][9]+' '+str(relevance_result[count])+'\n'
            count+=1
            
    return right_judge,right_count,precision,result_txt

def cross_validation(sample_dic,m):
    result_whole=''
    right_judge_whole=0
    right_count_whole=0
    precison_whole=0
    
    for n in range(m):
        print n
        right_judge,right_count,precision,result_txt=m_fold(sample_dic, n, m)
        result_whole+=result_txt
        right_judge_whole+=right_judge
        right_count_whole+=right_count
        precison_whole+=precision
    
    print right_judge_whole
    print right_count_whole
    print precison_whole/m*1.0
    
    fp_w=open('2014_result_classify.txt','w')
    fp_w.write(result_whole)    
    return 0

def feature_statistic(filename):
    fp=open(filename)
    lines=fp.readlines()
    count_0=0
    count_1=0
    count_2=0
    for line in lines:
        lineArr=line.strip().split(' ')
        if str(lineArr[12]).replace('\n','')=='0':
            count_0+=1
        if str(lineArr[12]).replace('\n','')=='1':
            count_1+=1
        if str(lineArr[12]).replace('\n','')=='2':
            count_2+=1    
    print '0: ',count_0,'1: ',count_1,'2: ',count_2+count_1
    return 0


def selectResult(filename_old,filname_new,reward):#17.提升分类结果中，正样本的分数
    fp=open(filename_old)
    lines=fp.readlines()
    text=''
    for line in lines:
        lineArr=line.strip().split(' ')
        if str(lineArr[12]).replace('\n','')=='1':
            text+=str(lineArr[0])+' '+str(lineArr[1])+' '+str(lineArr[2])+' '+str(lineArr[3])+' '+str(float(lineArr[4])+reward)+' '+str(lineArr[5])+'\n'
        else:
            text+=str(lineArr[0])+' '+str(lineArr[1])+' '+str(lineArr[2])+' '+str(lineArr[3])+' '+str(lineArr[4])+' '+str(lineArr[5])+'\n'
    fp_write=open(filname_new,'w')
    fp_write.write(text)
    return 0

def reRank(filename,reRankfile):#根据新分数进行排序
    fp=open(filename)
    lines=fp.readlines()
    dic_query={}   
    for line in lines:
        lineArr=line.split(' ')
        if lineArr[0] not in dic_query:
            dic_query[lineArr[0]]={}
        if lineArr[2] not in dic_query[lineArr[0]]:
            dic_query[lineArr[0]][lineArr[2]]=float(lineArr[4]) 
            
    combine_result_rank=''        
    for i in dic_query:
        count=0
        for item in sorted(dic_query[i].iteritems(), key=operator.itemgetter(1), reverse=True):
            combine_result_rank+=str(i)+' '+'Q0'+' '+str(item[0])+' '+str(count)+' '+str(item[1]).replace('\n','')+' '+'ecnuEn'+'\n'
            count+=1
        print str(i)
    fp_write=open(reRankfile,'w')
    fp_write.write(combine_result_rank)
    
    return 0

def cut_amount(filename,newfilename,n):
    dic_query={}
    fp=open(filename)
    lines=fp.readlines()
    text=''
    for line in lines:
        lineArr=line.split(' ')
        if lineArr[0] not in dic_query:
            dic_query[lineArr[0]]=1
        else:
            dic_query[lineArr[0]]+=1
        if dic_query[lineArr[0]]>=n:
            continue
        else:
            text+=str(line)
    fp_write=open(newfilename,'w')
    fp_write.write(text)
    return 0

if __name__ == "__main__": 
    #测试模式-交叉验证
    #feature_statistic('2015_50classify_result.txt')
    #sample_dic=construct_dic('2014_test.txt')
    #cross_validation(sample_dic, 5)   
    #classify('2014_1500test_e.txt', '2015_test.txt', '2015_50classify_result.txt')
    
    #selectResult('2015_50classify_result.txt', '2015_50classify_result_select01.txt', 0.1)
    #reRank('2015_50classify_result_select01.txt','2015_50classify_result_select01_r.txt')
    cut_amount('2015_50classify_result_select01_r.txt', '2015_50classify_result_select01_r_1000.txt', 1001)