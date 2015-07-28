import os,re,operator

from sklearn.neighbors.kde import KernelDensity
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

def max_mine(array):
    max_mine=1.0
    for item in array:
        if float(item)>max_mine:
            max_mine=float(item)
    return max_mine 

def min_mine(array):
    min_mine=100
    for item in array:
        if float(item)<min_mine:
            min_mine=float(item)
    return min_mine 

def uniform_calculate(feature):
    feature_uniform=[]
    min_item=min_mine(feature)
    max_item=max_mine(feature)
    print min_item,max_item
    for item in feature:
        item_uniform=(float(item) - min_item)/(max_item-min_item)
        feature_uniform.append(item_uniform)
    return feature_uniform   

def uniform_feature(original_feature,uniform_feature):
    fp=open(original_feature)
    lines=fp.readlines()
    fp_u=open(uniform_feature,'w')
    feature_u=''
    feature_2=[]
    feature_2_uniform=[]    
    for line in lines:
        lineArr=line.split(' ')
        feature_2.append(lineArr[4])
    feature_2_uniform=uniform_calculate(feature_2)    
    i=0
    for line in lines:
        lineArr=line.split(' ')
        feature_u+=str(lineArr[0])+' '+str(lineArr[1])+' '+str(lineArr[2])+' '+str(lineArr[3])+' '+str(feature_2_uniform[i])+' '+str(lineArr[5])
        i+=1
    fp_u.write(feature_u)

def construct_queryTree(original_feature):
    sample_dic={}
    fp=open(original_feature)
    lines=fp.readlines()
    for line in lines:
        lineArr=line.strip().split(' ')
        if lineArr[0] not in sample_dic: 
            sample_dic[lineArr[0]]={}
        if lineArr[2] not in sample_dic[lineArr[0]]:
            sample_dic[lineArr[0]][lineArr[2]]=[lineArr[0],lineArr[3],lineArr[4]]        
    return sample_dic

def uniform_query_one(dic):
    feature_2=[]
    for j in dic:
        feature_2.append(dic[j][2])
    feature_2_uniform=uniform_calculate(feature_2)        
    return feature_2_uniform

def uniform_query_score(original_feature,uniform_feature):
    sample_dic=construct_queryTree(original_feature)
    feature_whole=[]
    for i in sample_dic:
        #feature_2=[]
        #for j in sample_dic[i]:
            #feature_2.append(sample_dic[i][j][2])
        #feature_2_uniform=uniform_calculate(feature_2) 
        feature_2_uniform=uniform_query_one(sample_dic[i])
        feature_whole.extend(feature_2_uniform)
        #feature_2=[]
        
    
    fp_w=open(uniform_feature,'w')   
    feature_u=''
    count=0
    for i in sample_dic:
        for j in sample_dic[i]:
            feature_u+=str(i)+' Q0 '+str(j)+' '+sample_dic[i][j][1]+' '+str(feature_whole[count])+' ecnu'+'\n'
            count+=1
    fp_w.write(feature_u)
    return 0

def combine(filename_1,filename_2,combine_result_filename):
    fp_1=open(filename_1)
    fp_2=open(filename_2)
    dic_query={}
    lines_2=fp_2.readlines()
    for line in lines_2:
        lineArr=line.split(' ')
        if lineArr[0] not in dic_query:
            dic_query[lineArr[0]]={}
        if lineArr[2] not in dic_query[lineArr[0]]:
            dic_query[lineArr[0]][lineArr[2]]=lineArr[4]
    #storeweakClassArr(dic_query, 'dic_query.txt')
    combine_result=''
    combine_score=0.0
    lines_1=fp_1.readlines()
    for line in lines_1:
        lineArr=line.split(' ')
        if lineArr[0] in dic_query and lineArr[2] in dic_query[lineArr[0]]:
            combine_score=float(lineArr[4])
            combine_score+=float(dic_query[lineArr[0]][lineArr[2]])
            combine_result+=str(lineArr[0])+' '+str(lineArr[1])+' '+str(lineArr[2])+' '+str(lineArr[3])+' '+str(combine_score)+'\n'
    fp_write=open(combine_result_filename,'w')
    fp_write.write(combine_result)
    return 0


def construct_dic(filename):
    sample_dic={}
    fp=open(filename)
    lines=fp.readlines()
    for line in lines:
        lineArr=line.strip().split(' ')
        if lineArr[0] not in sample_dic: 
            sample_dic[lineArr[0]]={}
        if lineArr[2] not in sample_dic[lineArr[0]]:
            sample_dic[lineArr[0]][lineArr[2]]=[lineArr[0],lineArr[3],lineArr[4],lineArr[7],lineArr[9],lineArr[11],lineArr[12],lineArr[13]]        
    return sample_dic

def kernel_estimation(test,train_n,train_p):    
    relevance_score=[]
    result_n=[]
    result_p=[]   

    X_n=np.array(train_n)   
    X_p=np.array(train_p)
    Y=np.array(test)
    
    #params = {'bandwidth': np.logspace(-1, 1, 20)}
    #grid = GridSearchCV(KernelDensity(), params)
    #grid.fit(X_n)
    
    #print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))    
    
    kde_n = KernelDensity(kernel='gaussian', bandwidth=0.999).fit(X_n)
    kde_p = KernelDensity(kernel='gaussian', bandwidth=4.772).fit(X_p)
    for i in range(len(Y)):  
        result_n.append(np.exp(float(str(kde_n.score_samples(Y[i])).replace('[','').replace(']',''))))
        result_p.append(np.exp(float(str(kde_p.score_samples(Y[i])).replace('[','').replace(']',''))))
        if i%1000==0:
            print i      
    
    for i in range(len(result_n)): 
        if result_n[i]==0.0:
            relevance_score.append(np.log(result_p[i]/1.8404e-17+1))
        else:
            relevance_score.append(np.log(result_p[i]/result_n[i]+1))

    return relevance_score

def m_fold(sample_dic,n,m):
    test_query=[]
    train_query=[]
    test_data=[]
    train_data_n=[]
    train_data_p=[]
    
    for i in sample_dic:
        if int(i)%m==n:
            test_query.append(sample_dic[i])
        else:
            train_query.append(sample_dic[i])   
    
    for item in test_query:
        for i in item:
            test_data.append([float(item[i][7])])
    
    for item in train_query:
        for i in item:
            if item[i][6]=='0':
                train_data_n.append([float(item[i][7])])
            else:
                train_data_p.append([float(item[i][7])])

    relevance_score=kernel_estimation(test_data,train_data_n, train_data_p)
    
    count=0
    result_txt=''
    for item in test_query:
        for i in item:
            result_txt+=item[i][0]+' '+'Q0'+' '+str(i)+' '+item[i][1]+' '+item[i][2]+' '+'ECNU'+' '+str(relevance_score[count])+'\n'
            count+=1
            
    return result_txt

def cross_validation(sample_dic,m):
    result_whole=''
    for n in range(m):
        print n
        result_txt=m_fold(sample_dic, n, m)
        result_whole+=result_txt
    
    fp_w=open('2014_10000result_onefeature.txt','w')
    fp_w.write(result_whole)    
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

def load_data_pca(test_file,train_n_file,train_p_file):
    fp_test=open(test_file)
    fp_train_n=open(train_n_file)
    fp_train_p=open(train_p_file)
    
    lines_test=fp_test.readlines()
    lines_train_n=fp_train_n.readlines()
    lines_train_p=fp_train_p.readlines()
    
    test_array=[]
    train_n_array=[]
    train_p_array=[]
    
    for line in lines_test:
        lineArr=line.strip().split(' ')
        test_array.append([float(lineArr[0])])
        
    for line in lines_train_n:
        lineArr=line.strip().split(' ')
        train_n_array.append([float(lineArr[0])])
        
    for line in lines_train_p:
        lineArr=line.strip().split(' ')
        train_p_array.append([float(lineArr[0])])
        
    return test_array,train_n_array,train_p_array

def load_data_whole(test_file,train_n_file,train_p_file):
    fp_test=open(test_file)
    fp_train_n=open(train_n_file)
    fp_train_p=open(train_p_file)
    
    lines_test=fp_test.readlines()
    lines_train_n=fp_train_n.readlines()
    lines_train_p=fp_train_p.readlines()
    
    test_array=[]
    train_n_array=[]
    train_p_array=[]
    
    for line in lines_test:
        lineArr=line.strip().split(' ')
        test_array.append([float(lineArr[7]),float(lineArr[9]),float(lineArr[11]),float(lineArr[13]),float(lineArr[15])])
        
    for line in lines_train_n:
        lineArr=line.strip().split(' ')
        train_n_array.append([float(lineArr[7]),float(lineArr[9]),float(lineArr[11]),float(lineArr[13]),float(lineArr[15])])
        
    for line in lines_train_p:
        lineArr=line.strip().split(' ')
        train_p_array.append([float(lineArr[7]),float(lineArr[9]),float(lineArr[11]),float(lineArr[13]),float(lineArr[15])])
        
    return test_array,train_n_array,train_p_array

def learn_rank_kde(test_file,train_n_file,train_p_file):
    #test,train_n,train_p=load_data_pca(test_file,train_n_file,train_p_file)
    test,train_n,train_p=load_data_whole(test_file,train_n_file,train_p_file)
    relevance_score=kernel_estimation(test,train_n,train_p)
    
    score_txt=''
    for item in relevance_score:
        score_txt+=str(item)+'\n'
        
    fp_w=open('test_score_2.txt','w')
    fp_w.write(score_txt)
    return 0

def add_score(filename,file_score,new_filename):#将模型的分数加到原来的结果的最后一列
    pattern="e-"
    data_score=[]
    fp_score=open(file_score)
    lines_score=fp_score.readlines()
    for line in lines_score:
        #if re.match(pattern, str(line)):
        if float(str(line).strip())<0.0001:
         #   print line
            data_score.append(0.0)
        else:
            data_score.append(float(str(line).strip()))
    
    add_score_txt=''   
    fp=open(filename)
    lines=fp.readlines()
    for i in range(len(lines)):
        add_score_txt+=str(lines[i]).strip()+' '+str(data_score[i])+'\n'
    
    fp_writa=open(new_filename,'w')
    fp_writa.write(add_score_txt)
    
    return 0

def combine_score(filename,b,new_filename):#将模型分数与原始分数相加
    fp=open(filename)
    lines=fp.readlines()
    new_file=''
    for line in lines:
        lineArr=line.strip().split(' ')
        new_file+=str(lineArr[0])+' '+str(lineArr[1])+' '+str(lineArr[2])+' '+str(lineArr[3])+' '+str(float(lineArr[4])*(1.0-b)+float(lineArr[6])*b)+' '+str(lineArr[5])+'\n'
    
    fp_write=open(new_filename,'w')
    fp_write.write(new_file)
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

def query_web_dic(test_filename):
    fp=open(test_filename)
    lines=fp.readlines()
    dic={}   
    for line in lines:
        lineArr=line.split(' ')
        if lineArr[0] not in dic:
            dic[lineArr[0]]={}
        if lineArr[2] not in dic[lineArr[0]]:
            dic[lineArr[0]][lineArr[2]]=[lineArr[3],lineArr[4]]
    return dic

def select_feature1(result_filename,dic_twoLevel,feature_filename):
    feature=''
    count=0
    fp_result=open(result_filename)#this set should be large enough
    lines_result=fp_result.readlines()             
    for line in lines_result:
        lineArr=line.split(' ')
        if lineArr[0] in dic_twoLevel and str(lineArr[2]).replace('\n','') in dic_twoLevel[lineArr[0]]:
            feature+=str(line).replace('\n','')+' '+str(dic_twoLevel[lineArr[0]][str(lineArr[2]).replace('\n','')][0])+' '+str(dic_twoLevel[lineArr[0]][str(lineArr[2]).replace('\n','')][1])+'\n'
        else:
            feature+=str(line).replace('\n','')+' '+'10000'+' '+'0'+'\n'
        count+=1
        if count%1000==0:
            print count  
    fp_feature1=open(feature_filename,'w')
    fp_feature1.write(feature)

def query_web_dic2(test_filename):
    fp=open(test_filename)
    lines=fp.readlines()
    dic={}   
    for line in lines:
        lineArr=line.strip().split('\t')
        if lineArr[0] not in dic:
            dic[lineArr[0]]={}
        if str(lineArr[2]).replace('\n','') not in dic[lineArr[0]]:
            dic[lineArr[0]][str(lineArr[2]).replace('\n','')]=lineArr[3]
    return dic

def select_feature3(result_filename,dic_twoLevel,feature_filename):
    feature=''
    count=0
    fp_result=open(result_filename)#this set should be large enough
    lines_result=fp_result.readlines()             
    for line in lines_result:
        lineArr=line.strip().split(' ')
        if lineArr[0] in dic_twoLevel and str(lineArr[2]).replace('\n','') in dic_twoLevel[lineArr[0]]:
            feature+=str(line).replace('\n','')+' '+str(dic_twoLevel[lineArr[0]][str(lineArr[2]).replace('\n','')])+'\n'
        else:
            feature+=str(line).replace('\n','')+' '+'0'+'\n'
        count+=1
        if count%1000==0:
            print count  
    fp_feature1=open(feature_filename,'w')
    fp_feature1.write(feature)

def mine_pca(data):
    X = np.array(data)   
    reduced_data = PCA(n_components=1).fit_transform(X)   
    print reduced_data[0]
    return reduced_data

def whole_pca(filename,new_filename):
    data=[]
    fp=open(filename)
    sample_pca=''
    lines=fp.readlines()
    for line in lines:
        lineArr=line.strip().split(' ')
        data.append([float(lineArr[7]),float(lineArr[9]),float(lineArr[11])])
    reduced_data=mine_pca(data)
    
    for i in range(len(data)):
        sample_pca+=str(lines[i]).replace('\n','').strip()+' '+str(reduced_data[i]).replace('[','').replace(']','').strip()+'\n'
        
    fp_w=open(new_filename,'w')
    fp_w.write(sample_pca)
    
    return 0

def splitNpSample(filename):
    fp=open(filename)
    lines=fp.readlines()
    positive=''
    nagetive=''
    for line in lines:
        lineArr=line.strip().split(' ')
        if str(lineArr[12])=='0':
            nagetive+=str(lineArr[13])+'\n'
        else:
            positive+=str(lineArr[13])+'\n'
    fp_p=open('p_'+filename,'w')
    fp_p.write(positive)
    fp_n=open('n_'+filename,'w')
    fp_n.write(nagetive)
    return 0

def select_feature4(sample_list,filename_old,filename_new):
    fp_mesh =open('mtrees2015.bin')
    lines = fp_mesh.readlines()
    regx = {}
    n = 0
    for line in lines:
        lineArr = line.strip().split(';')
        if len(lineArr[0].split(' '))<4:
            #regx.append(lineArr[0])
            regx[str(lineArr[0]).lower()] = n
        n += 1
    print len(regx)    
    
    feature_1=''
    fp=open(filename_old)
    lines=fp.readlines()    
    count=0
    for line in lines:
        lineArr=line.split(' ')
        fp_d=open(sample_list+'\\'+str(lineArr[2])+'.xml')
        text=fp_d.read()
        match_word=0
        for j in regx:
            p = re.compile(str(j))
            match_word+=len(p.findall(text))
        #print match_word
        feature_1+=str(line).replace('\n','')+' '+str(match_word)+'\n'
        count+=1
        if count%2==0:
            print count
        
    fp_write=open(filename_new,'w')
    fp_write.write(feature_1)    
    return 0

def storeweakClassArr(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabweakClassArr(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)  

def select_feature2(filename_old,filename_new,sample_list):
    dic=grabweakClassArr('feature_word.txt')
    #storeweakClassArr(dic, 'feature_word.txt')
    feature_1=''
    fp=open(filename_old)
    lines=fp.readlines()
    count=0
    for line in lines:
        lineArr=line.split(' ')
        try:
            fp=open(sample_list+'\\'+str(lineArr[2])+'.xml')
            text=fp.read()
            match_word=0
            for j in dic:
                p = re.compile(str(j))
                match_word+=len(p.findall(text))
            #print match_word
            feature_1+=str(line).replace('\n','')+' '+str(match_word)+'\n'
            count+=1
        
            if count%1000==0:
                print count
        except:
            #print 'oop!'
            match_word=0
            feature_1+=str(line).replace('\n','')+' '+str(match_word)+'\n'
        
    fp_write=open(filename_new,'w')
    fp_write.write(feature_1)
    return 0

def select_feature5(filename_old,filename_new,sample_list):
    feature_1=''
    fp=open(filename_old)
    lines=fp.readlines()
    count=0
    for line in lines:
        lineArr=line.split(' ')
        try:
            fp=open(sample_list+'\\'+str(lineArr[2])+'.xml')
            text=fp.read()
            match_word=0
            match_word+=len(text.split(' '))
            #print match_word
            feature_1+=str(line).replace('\n','')+' '+str(match_word)+'\n'
            count+=1
        
            if count%1000==0:
                print count
        except:
            #print 'oop!'
            match_word=0
            feature_1+=str(line).replace('\n','')+' '+str(match_word)+'\n'
        
    fp_write=open(filename_new,'w')
    fp_write.write(feature_1)
    return 0

def format_trec(filename_old,filename_new):#14.将结果转化为trec的标准格式
    fp=open(filename_old)
    lines=fp.readlines()
    dic_query={}    
    for line in lines:
        lineArr=line.split(' ')
        if lineArr[0] not in dic_query:
            query_name=str(lineArr[0])
            dic_query[int(query_name)]=1
    sort_dic=sorted(dic_query.iteritems(), key=operator.itemgetter(0), reverse=False)
    query_array=[]
    for item in sort_dic:
        query_array.append(str(item[0]))
    
    trec_file='' 
    dic_query_content={}
    query_content=''
    for item in query_array:
        for line in lines:
            lineArr=line.split(' ')
            if lineArr[0] == str(item):
                query_content+=str(lineArr[0])+' '+str(lineArr[1])+' '+str(lineArr[2])+' '+str(lineArr[3])+' '+str(lineArr[4])+' '+'ecnuEn'+'\n'   
        dic_query_content[item]=query_content
        query_content=''
    for item in query_array:
        trec_file+=str(dic_query_content[item])
        
    fp_write=open(filename_new,'w')
    fp_write.write(trec_file)  
    
    return 0

def check_amount(filename,amount):#12.检查要提交的结果数量是否少于阈值amount
    fp=open(filename)
    lines=fp.readlines()
    dic_query={}
    for line in lines:
        lineArr=line.split(' ')
        if lineArr[0] not in dic_query:
            dic_query[lineArr[0]]=1
        else:
            dic_query[lineArr[0]]+=1
    sum=0
    for i in dic_query:
        sum+=dic_query[i]
        if dic_query[i]<amount:
            print 'alarm!',dic_query[i],i
        else:
            print 'safety',dic_query[i],i
    print sum 

if __name__ == "__main__": 
    #测试模式-交叉验证
    #sample_dic=construct_dic('2014_10000test_pca.txt')
    #cross_validation(sample_dic, 5)
    
    #测试模式-KDE效果比较
    #learn_rank_kde('12bm25whole_5000_np_pca.txt','n_11bm25whole_5000_np_pca.txt','p_11bm25whole_5000_np_pca.txt')
    
    #下面是提取特征的模块
    
    #uniform_feature('BB2c1.0_Bo1bfree_d_3_t_10_565.res', 'BB2c1.0_Bo1bfree_d_3_t_10_565_u.res')
       
    #uniform_feature('PL2c1.2_Bo1bfree_d_3_t_10_566.res', 'PL2c1.2_Bo1bfree_d_3_t_10_566_u.res')
    
    #uniform_feature('BM25b0.75_Bo1bfree_d_3_t_10_561.res', 'BM25b0.75_Bo1bfree_d_3_t_10_561_u.res') 
   
    #combine('BB2c1.0_Bo1bfree_d_3_t_10_560_u.res', 'PL2c1.2_Bo1bfree_d_3_t_10_559_u.res','559_560.res')
    #combine('559_560.res', 'BM25b0.75_Bo1bfree_d_3_t_10_561_u.res','559_560_561.res')    
    #reRank('559_560_561.res','559_560_561_r.res')
    
    #combine('BM25b0.75_Bo1bfree_d_3_t_10_561_u.res', 'PL2c1.2_Bo1bfree_d_3_t_10_559_u.res','559_561.res')
    #combine('BM25b0.75_Bo1bfree_d_3_t_10_567_u.res', 'PL2c1.2_Bo1bfree_d_3_t_10_566_u.res','566_567.res')
    #combine('566_567.res', 'BB2c1.0_Bo1bfree_d_3_t_10_565_u.res','565_566_567.res')
    #reRank('565_566_567.res','565_566_567_r.res')
    #cut_amount('559_560_561_r.res', '559_560_561_r_1000.res', 1001)
    #check_amount('559_560_561_r_1000.res', 1000)
    
           
    
    #dic_twoLevel=query_web_dic('BB2c1.0_Bo1bfree_d_3_t_10_560.res')
    #select_feature1('559_560_561_r.res',dic_twoLevel,'2015_feature_e_5.txt')      
    #dic_twoLevel=query_web_dic('PL2c1.2_Bo1bfree_d_3_t_10_559.res')
    #select_feature1('2015_feature_e_5.txt',dic_twoLevel,'2015_feature_e_5_2.txt')
    #dic_twoLevel=query_web_dic('BM25b0.75_Bo1bfree_d_3_t_10_561.res')
    #select_feature1('2015_feature_e_5_2.txt',dic_twoLevel,'2015_feature_e_5_2_4.txt')          
    #cut_amount('2015_feature_e_5_2_4.txt', '2015_test.txt', 1500)
    
    #uniform_feature('indri_lm.result10000', 'indri_lm_u.result10000')
    #uniform_feature('indri_lm.result10000', 'indri_lm_u.result10000')
    #uniform_query_score('indri_lm.result10000', 'indri_lm_u2.result10000')
    #combine('517_525_528.res', 'indri_lm_u.result10000','517_525_528_lm.res')   
    #reRank('517_525_528_lm.res','517_525_528_lm_r.res')
    
    #dic_twoLevel=query_web_dic2('qerl.txt')
    #select_feature3('2014_feature_u_5_2_4.txt',dic_twoLevel,'2014_10000test_u.txt')       
    
    #dic_twoLevel=query_web_dic('BB2c1.2_Bo1bfree_d_3_t_10_528_u.res')
    #select_feature1('517_525_528_r.res',dic_twoLevel,'2014_feature_u_5.txt')      
    #dic_twoLevel=query_web_dic('PL2c1.2_Bo1bfree_d_3_t_10_525_u.res')
    #select_feature1('2014_feature_u_5.txt',dic_twoLevel,'2014_feature_u_5_2.txt')
    #dic_twoLevel=query_web_dic('BM25b0.75_Bo1bfree_d_3_t_10_517_u.res')
    #select_feature1('2014_feature_u_5_2.txt',dic_twoLevel,'2014_feature_u_5_2_4.txt')      
    
    #select_feature4('I:\\trec2015\\code\\hy_sample\\hy_sample', '2014_test.txt', '2014_test_mesh.txt')
    #select_feature2('2014_test.txt', '2014_test_word.txt', 'I:\\trec2015\\code\\hy_sample\\hy_sample')
    #select_feature5('2014_test.txt', '2014_test_length.txt', 'I:\\trec2015\\code\\hy_sample\\hy_sample')
    #下面是概率密度估计模块
    
    cut_amount('2014_10000test_u.txt', '2014_1500test_u.txt', 1500)
    #whole_pca('2014_10000test_u.txt', '2014_10000test_u_pca.txt')
    #splitNpSample('2014_10000test_u_pca.txt')
    #five_fold(X,y,n)
    
    
    #下面是分类模型模块
    #classify('result_13year.txt', '2014_test.txt','2014_LOGresult.txt')
    
    
    #下面是分数合并模块
    
    #add_score('I:\\trec2015\\code\\12bm25whole_5000_np_pca.txt', 'I:\\trec2015\\code\\test_score_1.txt', 'I:\\trec2015\\code\\12_addscore1.txt')
    #combine_score('I:\\trec2015\\code\\2014_10000result_onefeature.txt', 0.5, 'I:\\trec2015\\code\\2014_10000combine_one_score05.txt')
    #reRank('I:\\trec2015\\code\\2014_10000combine_one_score05.txt', 'I:\\trec2015\\code\\2014_10000rerank_one_score05.txt')
    