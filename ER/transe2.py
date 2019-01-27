from random import uniform, sample
from numpy import *
from copy import deepcopy

class TransE2:
    def __init__(self, entityList, relationList, tripleList, margin = 1, learingRate = 0.00001, dim = 10, L1 = True):
        self.margin = margin
        self.learingRate = learingRate
        self.dim = dim #Dimension
        self.entityList = entityList
        self.relationList = relationList
        self.tripleList = tripleList
        self.loss = 0
        self.L1 = L1

    def initialize(self):
        '''
        Initialize embeddings.
        '''
        entityVectorList = {}
        relationVectorList = {}
        for entity in self.entityList:
            n = 0
            entityVector = []
            while n < self.dim:
                ram = init(self.dim)
                entityVector.append(ram)
                n += 1
            entityVector = norm(entityVector)
            entityVectorList[entity] = entityVector
        print("entityVector size %d"%len(entityVectorList))
        for relation in self. relationList:
            n = 0
            relationVector = []
            while n < self.dim:
                ram = init(self.dim)
                relationVector.append(ram)
                n += 1
            relationVector = norm(relationVector)
            relationVectorList[relation] = relationVector
        print("relationVectorList size: %d"%len(relationVectorList))
        self.entityList = entityVectorList
        self.relationList = relationVectorList

    def transE(self, cI = 20):
        print("Begin TransE")
        for cycleIndex in range(cI):
            Sbatch = self.getSample(150)
            Tbatch = []
            for sbatch in Sbatch:
                tripletWithCorruptedTriplet = (sbatch, self.getCorruptedTriplet(sbatch))
                if(tripletWithCorruptedTriplet not in Tbatch):
                    Tbatch.append(tripletWithCorruptedTriplet)
            self.update(Tbatch)
            if cycleIndex % 100 == 0:
                print("cycleIndex %d"%cycleIndex)
                print(self.loss)
                #self.writeRelationVector("c:\\relationVector.txt")
                #self.writeEntilyVector("c:\\entityVector.txt")
                self.loss = 0

    def getSample(self, size):
        return sample(self.tripleList, size)

    def getCorruptedTriplet(self, triplet):
        '''
        training triplets with either the head or tail replaced by a random entity (but not both at the same time)
        :param triplet:
        :return corruptedTriplet:
        '''
        i = uniform(-1, 1)
        if i < 0:
            while True:
                entityTemp = sample(self.entityList.keys(), 1)[0]
                if entityTemp != triplet[0]:
                    break
            corruptedTriplet = (entityTemp, triplet[1], triplet[2])
        else:
            while True:
                entityTemp = sample(self.entityList.keys(), 1)[0]
                if entityTemp != triplet[1]:
                    break
            corruptedTriplet = (triplet[0], entityTemp, triplet[2])
        return corruptedTriplet

    def update(self, Tbatch):
        copyEntityList = deepcopy(self.entityList)
        copyRelationList = deepcopy(self.relationList)

        for tripletWithCorruptedTriplet in Tbatch:
            headEntityVector = copyEntityList[tripletWithCorruptedTriplet[0][0]]#tripletWithCorruptedTriplet
            tailEntityVector = copyEntityList[tripletWithCorruptedTriplet[0][1]]
            relationVector = copyRelationList[tripletWithCorruptedTriplet[0][2]]
            headEntityVectorWithCorruptedTriplet = copyEntityList[tripletWithCorruptedTriplet[1][0]]
            tailEntityVectorWithCorruptedTriplet = copyEntityList[tripletWithCorruptedTriplet[1][1]]

            headEntityVectorBeforeBatch = self.entityList[tripletWithCorruptedTriplet[0][0]]#tripletWithCorruptedTriplet
            tailEntityVectorBeforeBatch = self.entityList[tripletWithCorruptedTriplet[0][1]]
            relationVectorBeforeBatch = self.relationList[tripletWithCorruptedTriplet[0][2]]
            headEntityVectorWithCorruptedTripletBeforeBatch = self.entityList[tripletWithCorruptedTriplet[1][0]]
            tailEntityVectorWithCorruptedTripletBeforeBatch = self.entityList[tripletWithCorruptedTriplet[1][1]]

            if self.L1:
                distTriplet = distanceL1(headEntityVectorBeforeBatch, tailEntityVectorBeforeBatch, relationVectorBeforeBatch)
                distCorruptedTriplet = distanceL1(headEntityVectorWithCorruptedTripletBeforeBatch, tailEntityVectorWithCorruptedTripletBeforeBatch ,  relationVectorBeforeBatch)
            else:
                distTriplet = distanceL2(headEntityVectorBeforeBatch, tailEntityVectorBeforeBatch, relationVectorBeforeBatch)
                distCorruptedTriplet = distanceL2(headEntityVectorWithCorruptedTripletBeforeBatch, tailEntityVectorWithCorruptedTripletBeforeBatch ,  relationVectorBeforeBatch)
            eg = self.margin + distTriplet - distCorruptedTriplet
            if eg > 0:
                self.loss += eg
                if self.L1:
                    tempPositive = 2 * self.learingRate * (tailEntityVectorBeforeBatch - headEntityVectorBeforeBatch - relationVectorBeforeBatch)
                    tempNegtative = 2 * self.learingRate * (tailEntityVectorWithCorruptedTripletBeforeBatch - headEntityVectorWithCorruptedTripletBeforeBatch - relationVectorBeforeBatch)
                    tempPositiveL1 = []
                    tempNegtativeL1 = []
                    for i in range(self.dim):
                        if tempPositive[i] >= 0:
                            tempPositiveL1.append(1)
                        else:
                            tempPositiveL1.append(-1)
                        if tempNegtative[i] >= 0:
                            tempNegtativeL1.append(1)
                        else:
                            tempNegtativeL1.append(-1)
                    tempPositive = array(tempPositiveL1)
                    tempNegtative = array(tempNegtativeL1)

                else:
                    tempPositive = 2 * self.learingRate * (tailEntityVectorBeforeBatch - headEntityVectorBeforeBatch - relationVectorBeforeBatch)
                    tempNegtative = 2 * self.learingRate * (tailEntityVectorWithCorruptedTripletBeforeBatch - headEntityVectorWithCorruptedTripletBeforeBatch - relationVectorBeforeBatch)

                headEntityVector = headEntityVector + tempPositive
                tailEntityVector = tailEntityVector - tempPositive
                relationVector = relationVector + tempPositive - tempNegtative
                headEntityVectorWithCorruptedTriplet = headEntityVectorWithCorruptedTriplet - tempNegtative
                tailEntityVectorWithCorruptedTriplet = tailEntityVectorWithCorruptedTriplet + tempNegtative

                #Update embeddings
                copyEntityList[tripletWithCorruptedTriplet[0][0]] = norm(headEntityVector)
                copyEntityList[tripletWithCorruptedTriplet[0][1]] = norm(tailEntityVector)
                copyRelationList[tripletWithCorruptedTriplet[0][2]] = norm(relationVector)
                copyEntityList[tripletWithCorruptedTriplet[1][0]] = norm(headEntityVectorWithCorruptedTriplet)
                copyEntityList[tripletWithCorruptedTriplet[1][1]] = norm(tailEntityVectorWithCorruptedTriplet)

        self.entityList = copyEntityList
        self.relationList = copyRelationList

    def writeEntilyVector(self, dir):
        print("Write entity vector")
        entityVectorFile = open(dir, 'w')
        for entity in self.entityList.keys():
            entityVectorFile.write(entity+"\t")
            entityVectorFile.write(str(self.entityList[entity].tolist()))
            entityVectorFile.write("\n")
        entityVectorFile.close()

    def writeRelationVector(self, dir):
        print("Write relation vector")
        relationVectorFile = open(dir, 'w')
        for relation in self.relationList.keys():
            relationVectorFile.write(relation + "\t")
            relationVectorFile.write(str(self.relationList[relation].tolist()))
            relationVectorFile.write("\n")
        relationVectorFile.close()

def init(dim):
    return uniform(-6/(dim**0.5), 6/(dim**0.5))

def distanceL1(h, t ,r):
    s = h + r - t
    sum = fabs(s).sum()
    return sum

def distanceL2(h, t, r):
    s = h + r - t
    sum = (s*s).sum()
    return sum

def norm(list):
    '''
    Normalize a vector
    '''
    var = linalg.norm(list)
    i = 0
    while i < len(list):
        list[i] = list[i]/var
        i += 1
    return array(list)

def openDetailsAndId(dir,sp="\t"):
    idNum = 0
    list = []
    with open(dir) as file:
        lines = file.readlines()
        for line in lines:
            DetailsAndId = line.strip().split(sp)
            list.append(DetailsAndId[0])
            idNum += 1
    return idNum, list

def openTrain(dir,sp="\t"):
    num = 0
    list = []
    with open(dir) as file:
        lines = file.readlines()
        for line in lines:
            triple = line.strip().split(sp)
            if(len(triple)<3):
                continue
            list.append(tuple(triple))
            num += 1
    return num, list