'''
    Record Linkage Testing Script for CORA dataset using K-means Classifier method.
'''
import pandas as pd
import numpy as np
import re
import recordlinkage
import xml.etree.ElementTree


def main():
    #Read data from XML file
    data = { 'dni' : [], 'author' : [], 'publisher' : [], 'date' : [], 'title' : [],
            'journal' : [], 'volume' : [], 'pages' : [], 'address' : []}
    e = xml.etree.ElementTree.parse('CORA.xml').getroot()
    print "Sample Record: ", xml.etree.ElementTree.dump(e.find('NEWREFERENCE'))

    for record in e.findall('NEWREFERENCE'):
        data['author'].append(unicode(record.find('author').text if record.find('author') is not None else '','utf-8'))
        dni = re.search('[a-z]+[0-9]+[a-z]*', record.text)
        dni = re.search('[a-z]+', record.text) if not dni else dni
        data['dni'].append( dni.group() if dni else record.text)
        data['title'].append(unicode(record.find('title').text if record.find('title') is not None else u''))
        data['publisher'].append(unicode(record.find('publisher').text if record.find('publisher') is not None else ''))
        data['date'].append(unicode(record.find('date').text if record.find('date') is not None else ''))
        data['journal'].append(unicode(record.find('journal').text if record.find('journal') is not None else ''))
        data['volume'].append(unicode(record.find('volume').text if record.find('volume') is not None else ''))
        data['pages'].append(unicode(record.find('pages').text if record.find('pages') is not None else ''))
        data['address'].append(unicode(record.find('address').text if record.find('address') is not None else ''))

    dataFrame = pd.DataFrame(data=data)
    print "DataFrame Shape:", dataFrame.shape

    #Split into train & test datasets
    data_train, data_test = dataFrame[:1200], dataFrame[1200:]

    #Split Train data in dataset A & B
    dataA, dataB = data_train[::2], data_train[1::2]
    print "Shape of dataset A & B", dataA.shape, dataB.shape

    #Extract all possible pairs
    indexer = recordlinkage.Index()
    indexer.full()
    candidate_links = indexer.index(dataA, dataB)
    print "Candidate Pairs", (len(candidate_links))

    #Extarct true links (takes time...)
    true_links = []
    for indexA, indexB in candidate_links:
        if dataA['dni'][indexA] == dataB['dni'][indexB]:
            true_links.append((indexA, indexB))
    print "Number of true links:", len(true_links)
    true_links = pd.MultiIndex.from_tuples(true_links)

    ## Extarct Features
    compare_cl = recordlinkage.Compare()

    compare_cl.string('title', 'title', method='jarowinkler', threshold=0.85, label='title')
    compare_cl.string('author', 'author', method='jarowinkler', threshold=0.85, label='author')
    compare_cl.string('publisher', 'publisher', method='jarowinkler', threshold=0.85, label='publisher')
    compare_cl.string('date', 'date', method='jarowinkler', threshold=0.85, label='date')
    compare_cl.string('pages', 'pages', method='jarowinkler', threshold=0.85, label='pages')
    compare_cl.string('volume', 'volume', method='jarowinkler', threshold=0.85, label='volume')
    compare_cl.string('journal', 'journal', method='jarowinkler', threshold=0.85, label='journal')
    compare_cl.string('address', 'address', method='jarowinkler', threshold=0.85, label='address')

    features = compare_cl.compute(candidate_links, dataA, dataB)
    print "Features ", features.describe()

    # Train K-Means Classifier
    logrg = recordlinkage.KMeansClassifier()
    logrg.fit(features)

    result = logrg.predict(features)
    print "\nTRAINING Results ", len(result)
    print "Confusion Matrix", recordlinkage.confusion_matrix(true_links, result, len(candidate_links))
    print "FScore: ", recordlinkage.fscore(true_links, result)
    print "Training Accuracy", recordlinkage.accuracy(true_links, result, len(candidate_links))
    print "Precision: ", recordlinkage.precision(true_links, result)
    print "Recall: ", recordlinkage.recall(true_links, result)

    #Test the classifier

    #Split Test data in dataset A & B
    testDataA, testDataB = data_test[::2], data_test[1::2]
    print "Shape of dataset A & B", testDataA.shape, testDataB.shape

    #Extract all possible pairs
    indexer = recordlinkage.Index()
    indexer.full()
    test_links = indexer.index(testDataA, testDataB)
    print "Candidate Pairs", (len(test_links))

    #Extarct true links (takes time...)
    true_test_links = []
    for indexA, indexB in test_links:
        if testDataA['dni'][indexA] == testDataB['dni'][indexB]:
            true_test_links.append((indexA, indexB))
    print "Number of true links:", len(true_test_links)
    true_test_links = pd.MultiIndex.from_tuples(true_test_links)

    features = compare_cl.compute(test_links, testDataA, testDataB)
    print "Features ", features.describe()

    result = logrg.predict(features)
    print "\n TESTING Results ", len(result)
    print "Confusion Matrix", recordlinkage.confusion_matrix(true_test_links, result, len(test_links))
    print "FScore: ", recordlinkage.fscore(true_test_links, result)
    print "Test Accuracy", recordlinkage.accuracy(true_test_links, result, len(test_links))
    print "Precision: ", recordlinkage.precision(true_test_links, result)
    print "Recall: ", recordlinkage.recall(true_test_links, result)


if __name__== "__main__":
  main()
