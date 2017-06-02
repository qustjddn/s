from numpy import *
import SVMLib as svm
reload(svm)

sVsMat = []
labelSVMat = []
svIndMat = []
alphasMat = []
bMat = []
    
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    line = fr.readline()
    M = int(line.strip().split('\t')[1])
    line = fr.readline()
    d = int(line.strip().split('\t')[1])
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        temp = []
        for i in range(d):
            temp.append(float(lineArr[i]))
        dataMat.append(temp)
        labelMat.append(float(lineArr[d]))
    return dataMat,labelMat,M,d

def TrainingSVMKernel(dataArr,labelArr,k1=1.3):
    b,alphas = svm.smoP(dataArr, labelArr, 20, 0.1, 100, ('rbf', k1))
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV = labelMat[svInd];
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = svm.kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print "the training error rate is: %f" % (float(errorCount)/m)
    
    return sVs,labelSV,svInd,alphas,b

def validationSVM(sVs,labelSV,svInd,alphas,b,dataArr,labelArr,k1=1.3):
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = svm.kernelTrans(sVs,datMat[i,:],('rbf', 1.3))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print "the test error rate is: %f" % (float(errorCount)/m)
    
def TrainingMultiSVM(dataMat,labelMat,M):
    
    for i in range(M):
        tempMat = labelMat[:]
        for j in range(len(tempMat)):
            if tempMat[j] != (i+1) : tempMat[j] = -1.0
            else : tempMat[j] = 1.0
                
        sVs,labelSV,svInd,alphas,b = TrainingSVMKernel(dataMat,tempMat)
        sVsMat.append(sVs)
        labelSVMat.append(labelSV)
        svIndMat.append(svInd)
        alphasMat.append(alphas)
        bMat.append(b)

def validationSVM(dataArr, labelArr,M,k1=1.3):  
    errorCount = 0
    datMat=mat(dataArr);
    
    for i in range(len(dataArr)):
        max = -9999999
        result = 0
        for j in range(M):
            sVs = sVsMat[j][:]
            labelSV = labelSVMat[j][:]
            svInd = svIndMat[j][:]
            alphas = alphasMat[j][:]
            b = bMat[j][:]
            
            kernelEval = svm.kernelTrans(sVs,datMat[i,:],('rbf', k1))
            predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
            if predict > max:
                max = predict
                result = j+1
        if result != labelArr[i]: errorCount += 1
    print "the test error rate is: %f" % (float(errorCount)/len(dataArr)) 