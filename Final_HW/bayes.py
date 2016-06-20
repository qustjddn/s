
# coding: utf-8

# In[ ]:
from numpy import *
def loadDataSet():
    postingList=[['my','dog','has','flea','problems','help','please'],                ['maybe','not','take','him','to','dog','park','stupid'],                ['my','dalmation','is','so','cute','I','love','him'],                ['stop','posting','stupid','worthless','garbage'],                ['mr','licks','ate','my','steak','how','to','stop','him'],                ['quit','buying','worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet|set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
    return returnVec

def bagOfWords2VecMN(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    p0Num=ones(numWords);p1Num=ones(numWords)
    p0Denom=2.0;p1Denom=2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect=log(p1Num/p1Denom )
    p0Vect=log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive
def trainNB00(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    p0Num=zeros(numWords);p1Num=zeros(numWords)
    p0Denom=0.0;p1Denom=0.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect=(p1Num/p1Denom )
    p0Vect=(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+log(pClass1)
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0
    
def testingNB():
    listOPosts,listClasses=loadDataSet()
    myVocabList=createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb=trainNB0(array(trainMat),array(listClasses))
    testEntry=['love','my','dalmation']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print testEntry, 'classfied as: ', classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry=['stupid','garbage']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print testEntry, 'classfied as: ', classifyNB(thisDoc,p0V,p1V,pAb)
    
def textParse(bigString):
    import re
    listOfTokens=re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]


def enableWord(word):
    illWordList=["a","an","the","for","RT",
                "by","and","of","or","&amp",
                "to","is","are","has","have",
                "can","about"]
    illStartWords=["http","..."]
    if word in illWordList: return 0
    for check in illStartWords:
        if word.startswith(check): return 0
        
    return 1
def fillteringWord(s):
    import re
    regEx=re.compile(r"[\#\@\?\!\:\;\,\"\-\ ]*")
    return regEx.split(s)   
            
def closeWord(word):
    illEndWords=["."]
    for check in illEndWords:
        if word.endswith(check):
            l=len(word)-1
            word=word[:l]
    return word
def textParse2(bigString):
    listOfTokens=fillteringWord(bigString)
    return [closeWord(tok.lower()) for tok in listOfTokens if len(tok)>2 and enableWord(tok.lower())==1]

def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList=textParse(open('email/spam/%d.txt'% i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textParse(open('email/ham/%d.txt'% i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList=createVocabList(docList)
    
    trainingSet=range(50); testSet=[]
    for i in range(10):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
      
        del(trainingSet[randIndex])     # to prevent duplication
    trainMat=[]; trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    
    p0V,p1V,pSpam=trainNB0(array(trainMat),array(trainClasses))
    errorCount=0
    print pSpam
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
                errorCount+=1
    print 'the error rate is: ', float(errorCount)/len(testSet)
    
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict={}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq=sorted(freqDict.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]
    
def localWords(feed1,feed0):
    import feedparser
    w1='entries'
    w2='summary'
    docList=[]; classList=[]; fullText=[]
    minLen=min(len(feed1[w1]),len(feed0[w1]))
    for i in range(minLen):
        wordList=textParse(feed1[w1][i][w2])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textParse(feed0[w1][i][w2])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList=createVocabList(docList)
    top30Words=calcMostFreq(vocabList,fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:vocabList.remove(pairW[0])
    trainingSet=range(2*minLen); testSet=[]
    for i in range(20):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=trainNB0(array(trainMat),array(trainClasses))
    errorCount=0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
           errorCount+=1
    print 'the error rate is: ', float(errorCount)/len(testSet)
    return vocabList,p0V,p1V

def getTopWords(feed1,feed0):
    import operator
    vocabList,p0V,p1V=localWords(feed1,feed0)
    topFeed1=[];topFeed0=[]
    for i in range(len(p0V)):
        if p0V[i]>-6.0: topFeed0.append((vocabList[i],p0V[i]))
        if p1V[i]>-6.0: topFeed1.append((vocabList[i],p1V[i]))
    sorted0=sorted(topFeed0,key=lambda pair:pair[1],reverse=True)
    print "-----feed0-----"
    for item in sorted0:
        print item[0]
    sorted1=sorted(topFeed1,key=lambda pair:pair[1],reverse=True)
    print "-----feed1-----"
    for item in sorted1:
        print item[0]
        
def getTopWords1():
    import operator
    vocabList,p0V,p1V,p1=tweetTest()
    topFeed1=[];topFeed0=[]
    for i in range(len(p0V)):
        if p0V[i]>-6.0: topFeed0.append((vocabList[i],p0V[i]))
        if p1V[i]>-6.0: topFeed1.append((vocabList[i],p1V[i]))
    sorted0=sorted(topFeed0,key=lambda pair:pair[1],reverse=True)
    print "-----p0-----"
    for item in sorted0[:30]:
        print item[0]
    sorted1=sorted(topFeed1,key=lambda pair:pair[1],reverse=True)
    print "-----p1-----"
    for item in sorted1[:30]:
        print item[0]
def classifyArrNB(strArr,vocabList,p0V,p1V,p1):
    docList=[];
    for str in strArr:
        buff=str['text']
        buff=buff.encode('utf8')
        #buff.replace('\n','\t')
        arr=buff.split('\n')
        buff=""
        for s in arr:
            buff+=s
        docList.append(textParse2(buff))
    
    for ix in range(len(docList)):
        wordVector = bagOfWords2VecMN(vocabList,docList[ix])
        print "\nNo.%d tweet has been classified as %d"%(ix,classifyNB(array(wordVector),p0V,p1V,p1))
        print strArr[ix]['text']+"\n"
        
    
def tweetTest():
    docList=[]; classList = []; fullText =[]
    file=open("../data/twitter_data/twandlabel.txt")
    count=0
    for line in file.readlines():
        split=line.split('\t')
        wordList=textParse2(split[0])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(int(split[1][0]))
        count+=1
    vocabList=createVocabList(docList)    
    
    trainingSet=range(count); testSet=[]
   
    count=count/10
    for i in range(count):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])   
      
        del(trainingSet[randIndex])     # to prevent duplication
    trainMat=[]; trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    #print classList
    p0V,p1V,p1=trainNB0(array(trainMat),array(trainClasses))
    errorCount=0
    
    
    print p1
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,p1)!=classList[docIndex]:
                errorCount+=1
    
    print 'the error rate is: ', float(errorCount)/len(testSet)
    
    return vocabList,p0V,p1V,p1
    
    
