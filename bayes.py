{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "'''\n",
    "Created on Oct 19, 2010\n",
    "\n",
    "@author: Peter\n",
    "'''\n",
    "from numpy import *\n",
    "\n",
    "def loadDataSet():\n",
    "    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],\n",
    "                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],\n",
    "                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],\n",
    "                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],\n",
    "                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],\n",
    "                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]\n",
    "    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not\n",
    "    return postingList,classVec\n",
    "                 \n",
    "def createVocabList(dataSet):\n",
    "    vocabSet = set([])  #create empty set\n",
    "    for document in dataSet:\n",
    "        vocabSet = vocabSet | set(document) #union of the two sets\n",
    "    return list(vocabSet)\n",
    "\n",
    "def setOfWords2Vec(vocabList, inputSet):\n",
    "    returnVec = [0]*len(vocabList)\n",
    "    for word in inputSet:\n",
    "        if word in vocabList:\n",
    "            returnVec[vocabList.index(word)] = 1\n",
    "        else: print \"the word: %s is not in my Vocabulary!\" % word\n",
    "    return returnVec\n",
    "\n",
    "def trainNB0(trainMatrix,trainCategory):\n",
    "    numTrainDocs = len(trainMatrix)\n",
    "    numWords = len(trainMatrix[0])\n",
    "    pAbusive = sum(trainCategory)/float(numTrainDocs)\n",
    "    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones() \n",
    "    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0\n",
    "    for i in range(numTrainDocs):\n",
    "        if trainCategory[i] == 1:\n",
    "            p1Num += trainMatrix[i]\n",
    "            p1Denom += sum(trainMatrix[i])\n",
    "        else:\n",
    "            p0Num += trainMatrix[i]\n",
    "            p0Denom += sum(trainMatrix[i])\n",
    "    p1Vect = log(p1Num/p1Denom)          #change to log()\n",
    "    p0Vect = log(p0Num/p0Denom)          #change to log()\n",
    "    return p0Vect,p1Vect,pAbusive\n",
    "\n",
    "def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):\n",
    "    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult\n",
    "    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)\n",
    "    if p1 > p0:\n",
    "        return 1\n",
    "    else: \n",
    "        return 0\n",
    "    \n",
    "def bagOfWords2VecMN(vocabList, inputSet):\n",
    "    returnVec = [0]*len(vocabList)\n",
    "    for word in inputSet:\n",
    "        if word in vocabList:\n",
    "            returnVec[vocabList.index(word)] += 1\n",
    "    return returnVec\n",
    "\n",
    "def testingNB():\n",
    "    listOPosts,listClasses = loadDataSet()\n",
    "    myVocabList = createVocabList(listOPosts)\n",
    "    trainMat=[]\n",
    "    for postinDoc in listOPosts:\n",
    "        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))\n",
    "    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))\n",
    "    testEntry = ['love', 'my', 'dalmation']\n",
    "    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))\n",
    "    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)\n",
    "    testEntry = ['stupid', 'garbage']\n",
    "    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))\n",
    "    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)\n",
    "\n",
    "def textParse(bigString):    #input is big string, #output is word list\n",
    "    import re\n",
    "    listOfTokens = re.split(r'\\W*', bigString)\n",
    "    return [tok.lower() for tok in listOfTokens if len(tok) > 2] \n",
    "    \n",
    "def spamTest():\n",
    "    docList=[]; classList = []; fullText =[]\n",
    "    for i in range(1,26):\n",
    "        wordList = textParse(open('email/spam/%d.txt' % i).read())\n",
    "        docList.append(wordList)\n",
    "        fullText.extend(wordList)\n",
    "        classList.append(1)\n",
    "        wordList = textParse(open('email/ham/%d.txt' % i).read())\n",
    "        docList.append(wordList)\n",
    "        fullText.extend(wordList)\n",
    "        classList.append(0)\n",
    "    vocabList = createVocabList(docList)#create vocabulary\n",
    "    trainingSet = range(50); testSet=[]           #create test set\n",
    "    for i in range(10):\n",
    "        randIndex = int(random.uniform(0,len(trainingSet)))\n",
    "        testSet.append(trainingSet[randIndex])\n",
    "        del(trainingSet[randIndex])  \n",
    "    trainMat=[]; trainClasses = []\n",
    "    for docIndex in trainingSet:#train the classifier (get probs) trainNB0\n",
    "        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))\n",
    "        trainClasses.append(classList[docIndex])\n",
    "    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))\n",
    "    errorCount = 0\n",
    "    for docIndex in testSet:        #classify the remaining items\n",
    "        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])\n",
    "        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:\n",
    "            errorCount += 1\n",
    "            print \"classification error\",docList[docIndex]\n",
    "    print 'the error rate is: ',float(errorCount)/len(testSet)\n",
    "    #return vocabList,fullText\n",
    "\n",
    "def calcMostFreq(vocabList,fullText):\n",
    "    import operator\n",
    "    freqDict = {}\n",
    "    for token in vocabList:\n",
    "        freqDict[token]=fullText.count(token)\n",
    "    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) \n",
    "    return sortedFreq[:30]       \n",
    "\n",
    "def localWords(feed1,feed0):\n",
    "    import feedparser\n",
    "    docList=[]; classList = []; fullText =[]\n",
    "    minLen = min(len(feed1['entries']),len(feed0['entries']))\n",
    "    for i in range(minLen):\n",
    "        wordList = textParse(feed1['entries'][i]['summary'])\n",
    "        docList.append(wordList)\n",
    "        fullText.extend(wordList)\n",
    "        classList.append(1) #NY is class 1\n",
    "        wordList = textParse(feed0['entries'][i]['summary'])\n",
    "        docList.append(wordList)\n",
    "        fullText.extend(wordList)\n",
    "        classList.append(0)\n",
    "    vocabList = createVocabList(docList)#create vocabulary\n",
    "    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words\n",
    "    for pairW in top30Words:\n",
    "        if pairW[0] in vocabList: vocabList.remove(pairW[0])\n",
    "    trainingSet = range(2*minLen); testSet=[]           #create test set\n",
    "    for i in range(20):\n",
    "        randIndex = int(random.uniform(0,len(trainingSet)))\n",
    "        testSet.append(trainingSet[randIndex])\n",
    "        del(trainingSet[randIndex])  \n",
    "    trainMat=[]; trainClasses = []\n",
    "    for docIndex in trainingSet:#train the classifier (get probs) trainNB0\n",
    "        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))\n",
    "        trainClasses.append(classList[docIndex])\n",
    "    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))\n",
    "    errorCount = 0\n",
    "    for docIndex in testSet:        #classify the remaining items\n",
    "        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])\n",
    "        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:\n",
    "            errorCount += 1\n",
    "    print 'the error rate is: ',float(errorCount)/len(testSet)\n",
    "    return vocabList,p0V,p1V\n",
    "\n",
    "def getTopWords(ny,sf):\n",
    "    import operator\n",
    "    vocabList,p0V,p1V=localWords(ny,sf)\n",
    "    topNY=[]; topSF=[]\n",
    "    for i in range(len(p0V)):\n",
    "        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))\n",
    "        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))\n",
    "    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)\n",
    "    print \"SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**\"\n",
    "    for item in sortedSF:\n",
    "        print item[0]\n",
    "    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)\n",
    "    print \"NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**\"\n",
    "    for item in sortedNY:\n",
    "        print item[0]\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
