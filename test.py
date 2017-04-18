import numpy as np

from sklearn.datasets import load_digits, load_iris
from sklearn.cross_validation import train_test_split

from kperceptron import KPerceptron

if __name__ == '__main__':
    dset = load_digits()
    print dset.data.shape
    Xtrn, Xtst, ytrn, ytst = train_test_split(dset.data, dset.target, test_size=0.3)
    Xtrn, Xtst, ytrn, ytst = Xtrn, Xtst,\
            ytrn+1, ytst+1
    print Xtrn.shape, ytrn.shape, Xtst.shape, ytst.shape
    print np.unique(ytrn)
    print 'small change'

    sp = KPerceptron(kerntype='poly',kerngamma=1)
    for e in xrange(50):
        sp.partial_fit(Xtrn,ytrn)
        yHat, acc = sp.predict(Xtst), sp.score(Xtst,ytst)
        print 'e: {} acc: {}'.format(e,acc)
