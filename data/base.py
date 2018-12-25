class Dataset(object):
    """Base class for all Dataset"""
    trainDataA = None
    trainDataB = None
    testDataA = None
    testDataB = None
    true_links = None
    candidate_links = None
    test_links = None
    true_test_links = None

    def __init__(self, arg):
        super(Dataset, self).__init__()
        self.arg = arg
