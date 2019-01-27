# record-linkage
Python Script to Link records using graph embeddings.

### How to test
The module uses [python unittest framework](https://docs.python.org/2.7/library/unittest.html) for testing record linkage methods. 

* To run all tests described in tests folder
```
python -m unittest discover
```

* To run all tests related to specified method in a test file
```
python -m unittest discover -p test_ecm*.py
```

The logs are accumulated in test.log file by default.

### Datasets

We work with three datasets defined in data folder. All pre-processing and spliting the data in training and test sets is defined in their specific classes.
1. Cora: Bibliographical records for scientic papers 
2. FEBRL: Syntectic Census dataset
3. Census: Real Census data from Barcelona from early 20th century. 

Raw census data is propeietry of [UAB-CVC](http://www.cvc.uab.es/) and thus included in this repositary.

### Methods

For base results we use [recordlinkage](https://github.com/J535D165/recordlinkage) toolkit to test following methods:
1. ECM Classifier
2. K-Means Clustering
3. Logistic Regression

Following graph embedding based methods are implemented
1. TransE
2. TransH **Todo**
3. KR-EAR **Todo**
4. SEEA **Todo**
