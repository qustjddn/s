
# coding: utf-8

# In[1]:

from twitter import *
def writingTxt(filename,strArr):
    file=open(filename,'w')
    
    for str in strArr:
        buff=str['text']
        buff=buff.encode('utf8')
        #buff.replace('\n','\t')
        arr=buff.split('\n')
        buff=""
        for s in arr:
            buff+=s
        file.write(buff+"\n")
        #print str['text']
    file.close()    
    return len(strArr)


# In[2]:
def getKey():
    token="2373074766-PL14cizo2IrNPcvcUI25kdi7KjZQMAdRnNx2SPN"
    token_key="VKTOHImjOvE4Kp8ECj3T11zzorHuwxUImNnfWGuuVIaK9"
    con_secret="NIM6wPYiI2S4rCRkAPZUpaCpa"
    con_secret_key="R8a6YSb4yf77X27emnGcI9hKdYYAEDbSqK08ocg6s5Vz9yLnr6"
    
    return token,token_key,con_secret,con_secret_key
    
def searchTweet(key):
    token,token_key,con_secret,con_secret_key=getKey()
    

    t = Twitter(auth=OAuth(token, token_key, con_secret, con_secret_key))


    x=t.search.tweets(q=key)
    txt="twitter_data/test.txt"
    return x['statuses']
    #print writingTxt(txt,x['statuses'])   
    
    
# source : http://blog.naver.com/lseongjoo/100195498647
# source : https://pypi.python.org/pypi/twitter#downloads


# In[79]:
def testFucn():
    #x=t.search.tweets(q="#Trump")
    print len(x['statuses'])

    for i in range(len(x['statuses'])):
        print "--%d"%i
        print x['statuses'][i]['text']


    # In[110]:

    #str= x['statuses'][1]['text']
    #str.replace(chr(10),'-')
    #t=str.split('\n')
    #print t
    #for s in t:
    #    print s
    #print len(str)
    #for i in range(len(str)):
    #    if ord(str[i])==10:
    #        print "---------------"
            #str[i]='\t'
    #    print str[i]

        #print ord(str[58])

