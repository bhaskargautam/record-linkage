'''
    Record Linkage Testing Script for FEBRL dataset using Kmeans Classifier method.
'''
import recordlinkage
from recordlinkage.datasets import load_febrl4, load_febrl3
import pandas as pd

def main():
    #Read data
    dfA_complete, dfB_complete, true_links_complete = load_febrl4(return_links=True)
    print "Sample DatasetA", dfA_complete[:2]
    print "Sample DatasetB", dfB_complete[:2]
    count = 2
    for x in dfA_complete.iterrows():
        for y in dfB_complete.iterrows():
            if x[0].split('-')[1] == y[0].split('-')[1]:
                print "Matching Pair", x, y
        if count < 1:
            break
        else:
            count = count - 1

    print "Shape of datasets ", dfA_complete.shape, dfB_complete.shape
    print "Shape of True Links", true_links_complete.shape

    # Split test & train dataset
    dfA_train, dfB_train = dfA_complete[:4000], dfB_complete[:4000]
    dfA_test, dfB_test = dfA_complete[-1000:], dfB_complete[-1000:]

    # Compute candidate links
    indexer = recordlinkage.Index()
    indexer.block('given_name')
    candidate_links = indexer.index(dfA_train, dfB_train)
    print "Candidate Pairs", (len(candidate_links))

    ## Extarct Features
    compare_cl = recordlinkage.Compare()

    compare_cl.exact('given_name', 'given_name', label='given_name')
    compare_cl.string('surname', 'surname', method='jarowinkler', threshold=0.85, label='surname')
    compare_cl.exact('date_of_birth', 'date_of_birth', label='date_of_birth')
    compare_cl.exact('state', 'state', label='state')

    features = compare_cl.compute(candidate_links, dfA_train, dfB_train)
    print "Features ", features.describe()

    #Extract True Links
    true_links_train = []
    for i in candidate_links:
        if i in true_links_complete:
            true_links_train.append(i)
    true_links_train = pd.MultiIndex.from_tuples(true_links_train)

    ## Train ECM CLassifier
    logrg = recordlinkage.KMeansClassifier()
    logrg.fit(features)

    result = logrg.predict(features)
    try:
        print "Results ", len(result)
        print "Confusion Matrix", recordlinkage.confusion_matrix(true_links_train, result, len(candidate_links))
        print "FScore: ", recordlinkage.fscore(true_links_train, result)
        print "Training Accuracy", recordlinkage.accuracy(true_links_train, result, len(candidate_links))
        print "Precision: ", recordlinkage.precision(true_links_train, result)
        print "Recall: ", recordlinkage.recall(true_links_train, result)
    except ZeroDivisionError:
        print "Zero division error!!!!"

    # Test
    indexer = recordlinkage.Index()
    indexer.block('given_name')
    test_links = indexer.index(dfA_test, dfB_test)
    print "Test Pairs", (len(test_links))
    features = compare_cl.compute(test_links, dfA_test, dfB_test)
    print "Features ", features.describe()
    result = logrg.predict(features)

    #Extract True Links
    true_links_test = []
    for i in test_links:
        if i in true_links_complete:
            true_links_test.append(i)
    true_links_test = pd.MultiIndex.from_tuples(true_links_test)

    try:
        print "Results ", len(result)
        print "Confusion Matrix", recordlinkage.confusion_matrix(true_links_test, result, len(test_links))
        print "FScore: ", recordlinkage.fscore(true_links_test, result)
        print "Test Accuracy", recordlinkage.accuracy(true_links_test, result, len(test_links))
        print "Precision: ", recordlinkage.precision(true_links_test, result)
        print "Recall: ", recordlinkage.recall(true_links_test, result)
    except ZeroDivisionError:
        print "Zero division error!!!!"

if __name__== "__main__":
  main()