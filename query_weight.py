# -*- coding: utf-8 -*- 
import re,operator

def query_assignWeight(filename,new_filename):
    fp=open(filename)
    lines=fp.readlines()
    query_weight=''
    for line in lines:
        lineArr=line.strip().split(' ')
        if len(lineArr)<=4:
            query_weight+=str(line)
        else:
            desc_txt=query_detect(lineArr)
            query_weight+=desc_txt
           
    fp_w=open(new_filename,'w')
    fp_w.write(query_weight)
    return 0

def query_detect(lineArr):
    desc_txt=''
    for item in lineArr:
        if item=='adult' or item=='children' or item=='female' or item=='male' or item=='diagnosis' or item=='test' or item=='treatment': 
            desc_txt+=str(item)+'^1.0'+' '
        else:
            desc_txt+=str(item)+'^1.0'+' '
            
    desc_txt+='\n'
    return desc_txt

def query_google(filename,new_filename):
    fp=open(filename)
    lines=fp.readlines()
    new_txt=''
    count=1
    for line in lines:
        lineArr=line.strip().split(' ')
        if len(lineArr)>4:
            new_txt+=str(count)+'\t'+str(line.strip())+'\n'
            count+=1
    fp_w=open(new_filename,'w')
    fp_w.write(new_txt)
    
    return 0

def google_collect(snip_filename,title_filename):
    #pattern = re.compile('[:;,\'\'|{}.?&$%=-]')
    fp_snip=open(snip_filename)
    lines_snip=fp_snip.readlines()
    html={}
    key=''
    
    for line in lines_snip:
        if str(line).strip()!='':
            lineArr=line.strip().split('\t')
            if str(lineArr[0]).isdigit() and lineArr[0] not in html:
                html[lineArr[0]]=[]
                key=lineArr[0]
            else:
                lineArrArr=lineArr[1].strip().split(' ')
                for item in lineArrArr:
                    if item.isalpha():
                        #html[key].append(re.sub(pattern, '', item))
                        html[key].append(item.lower())
                    
    
    fp_title=open(title_filename)
    lines_title=fp_title.readlines()
    
    for line in lines_title:
        if str(line).strip()!='':
            lineArr=line.strip().split('\t')
            if str(lineArr[0]).isdigit():
                key=lineArr[0]
            else:
                lineArrArr=lineArr[1].strip().split(' ')
                for item in lineArrArr:
                    if item.isalpha():
                        #html[key].append(re.sub(pattern, '', item))
                        html[key].append(item.lower())
    
    #print html             
    return html

def google_choose(html,original_query,expansion_query):
    stop={}
    fp_stop=open('english.stop')
    lines=fp_stop.readlines()
    for line in lines:
        stop[line.strip()]=1
        
    #print stop
    
    word_expansion={}
    for i in html:
        if i not in word_expansion:
            word_expansion[i]={}
        for item in html[i]:
            if item not in word_expansion[i]:
                word_expansion[i][item]=1
            else:
                word_expansion[i][item]+=1
    
    fp_o=open(original_query)
    o_query={}
    lines_o=fp_o.readlines()
    count=1
    for line in lines_o:
        lineArr=line.strip().split(' ')
        if len(lineArr)>4:
            o_query[str(count)]=line.strip()+' '
            count+=1    
    print o_query
    
    
    word_expansion_choose={}
    for i in word_expansion:
        if i not in word_expansion_choose:
            word_expansion_choose[i]=''
        count=0
        for item in sorted(word_expansion[i].iteritems(), key=operator.itemgetter(1), reverse=True):
            
            if item[0] not in stop and item[1]>4 and count<3:
                #print i,item[0]
                word_expansion_choose[i]+=item[0]+' '
                count+=1  
                
    print word_expansion_choose
    
    for i in o_query:
        o_query[i]+=str(word_expansion_choose[i])
        
    print o_query
    
    query_new=''        
    for i in o_query:
        query_new+='<top>\n\n'+'<num> Number: '+i+'\n\n'+'<desc>\n\n'+o_query[i]+'\n\n'+r'</top>'+'\n\n'
        
    fp_w=open(expansion_query,'w')
    fp_w.write(query_new)
    
    
    return 0

def query_2015_transform(query_file,new_file):
    o_query={}
    pattern = re.compile('<description[\s\S]*?<\/description>')
    fp_q=open(query_file)
    query_text=fp_q.read()
    query_text=re.sub(pattern, '', query_text)
    lines=query_text.strip().split('\n')
    
    count=1
    for line in lines:
        lineArr=line.strip().split(' ')
        if len(lineArr)>5:
            o_query[str(count)]=line.strip()+' '
            count+=1
            
    #print o_query
    query_new=''
    for i in range(1,31):
        if int(i)>0 and int(i)<11:
            query_new+='<top>\n\n'+'<num> Number: '+str(i)+'\n\n'+'<desc>\n\n'+o_query[str(i)]+'diagnosis'+'\n\n'+r'</top>'+'\n\n'
        elif int(i)>10 and int(i)<21:
            query_new+='<top>\n\n'+'<num> Number: '+str(i)+'\n\n'+'<desc>\n\n'+o_query[str(i)]+'test'+'\n\n'+r'</top>'+'\n\n'
        elif int(i)>20 and int(i)<31:
            query_new+='<top>\n\n'+'<num> Number: '+str(i)+'\n\n'+'<desc>\n\n'+o_query[str(i)]+'treatment'+'\n\n'+r'</top>'+'\n\n'           
        
    fp_w=open(new_file,'w')
    fp_w.write(query_new)    
    return o_query


def reflect(o_query):
    reflect_dic={}
    count=1
    for i in o_query:
        value=str(i)
        reflect_dic[str(count)]=value
        count+=1
    print reflect_dic
    return reflect_dic

def reflect_result(result_file,reflect_dic,new_file):
    fp=open(result_file)
    lines=fp.readlines()
    new_result=''
    for line in lines:
        lineArr=line.strip().split(' ')
        if lineArr[0] in reflect_dic:
            new_result+=reflect_dic[lineArr[0]]+' '+lineArr[1]+' '+lineArr[2]+' '+lineArr[3]+' '+lineArr[4]+' '+lineArr[5]+'\n'
            
    fp_w=open(new_file,'w')
    fp_w.write(new_result)
    return 0

if __name__ == "__main__":
    #query_assignWeight('2014AgeGenderNegationQuery.txt', '2014AgeGenderNegationQuery_weight1.txt')
    #query_google('2015OriginalQuery2.txt', '2015.CDSid_query2.txt')
   
    #o_query=query_2015_transform('2015query.txt', '2015OriginalQuery2.txt')
    #reflect_dic=reflect(o_query)
    #reflect_result('BM25b0.75_Bo1bfree_d_3_t_10_561_old.res', reflect_dic, 'BM25b0.75_Bo1bfree_d_3_t_10_561.res')
    
    #html=google_collect('parse_snip2.txt','parse_title2.txt')
    #google_choose(html,'2015AgeGenderNegationQuery2.txt','2015AgeGenderNegationQuery_expansion3.txt')    