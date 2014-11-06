import numpy
import vigra
import scipy

from sklearn.svm import OneClassSVM
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, roc_curve, auc
from sklearn.metrics.metrics import roc_curve
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.mixture import GMM

class OneClassRandomForest(object):
    def __init__(self, outlier_over_sampling_factor=4, *args, **kwargs):
        self.outlier_over_sampling_factor = outlier_over_sampling_factor
        self.rf = vigra.learning.RandomForest(100)
    
    def fit(self, data):
        data = numpy.require(data, numpy.float32)
        d = data.shape[1]
        n = data.shape[0]
        synt_outliers = numpy.random.random((n*self.outlier_over_sampling_factor, d))
        for i in xrange(d):
            i_min, i_max = data[:,i].min()*1.1, data[:,i].max()*1.1
            synt_outliers[:,i]*= (i_max - i_min)
            synt_outliers[:,i]+= i_min
                    
        training_data = numpy.r_[data, synt_outliers].astype(numpy.float32)
        trianing_labels = numpy.r_[numpy.zeros((n,1), dtype=numpy.uint32), numpy.ones((n * self.outlier_over_sampling_factor,1), dtype=numpy.uint32)]
        
        print 'oob', self.rf.learnRFWithFeatureSelection(training_data, trianing_labels)
    
    def predict(self, data):
        data = numpy.require(data, numpy.float32)
        res = self.rf.predictProbabilities(data.astype(numpy.float32))
        outlier = (res[:,1] > 0.05).astype(numpy.int32)*-2 + 1
        
        return outlier
    
    def decision_function(self, data):
        return numpy.ones((data.shape[0],1))
        
    @staticmethod
    def test_simple():
        d = 100
        n = 6000 
        mean = numpy.zeros((d,)) 
        cov = numpy.eye(d,d)
        
        x = numpy.random.multivariate_normal(mean, cov, n)
        rf = OneClassRandomForest()
        rf.fit(x)
        
        x_2 = numpy.random.random((n, d))
        for i in xrange(d):
            i_min, i_max = x[:,i].min()*1.2, x[:,i].max()*1.2
            x_2[:,i]*= (i_max - i_min)
            x_2[:,i]+= i_min
            
   
        testing_data = numpy.r_[x, x_2].astype(numpy.float32)
        testing_labels = numpy.r_[numpy.zeros((n,1), dtype=numpy.uint32), numpy.ones((n,1), dtype=numpy.uint32)]
        testing_pred = rf.predict(testing_data)
        
        print accuracy_score(testing_labels, testing_pred)

class OneClassAngle(object):
    def __init__(self, *args, **kwargs):
        pass
    
    def fit(self, data):
        self.data = data
        self._data_norm = []
        for row in data:
            self._data_norm.append(numpy.linalg.norm(row))
            
        result = numpy.zeros((data.shape[0], data.shape[0]))
        
        for t1 in xrange(data.shape[0]):
            for t2 in xrange(data.shape[0]):
                t1_vec = data[t1, :]
                t2_vec = data[t2, :]               
                result[t1, t2] = numpy.dot(t1_vec, t2_vec) / (self._data_norm[t1] * self._data_norm[t2])
        
        outlier_score = result.std(1)
        self.outlier_cutoff =  outlier_score.mean()*0.85
        
        print ' outlier cutoff', self.outlier_cutoff

    
    def predict(self, data_new):
        result = numpy.zeros((data_new.shape[0], self.data.shape[0]))
        
        _data_norm_new = []
        for row in data_new:
            _data_norm_new.append(numpy.linalg.norm(row))
        
        for test in xrange(data_new.shape[0]):
            for train in xrange(self.data.shape[0]):
                test_vec = data_new[test, :]
                train_vec = self.data[train, :]               
                result[test, train] = numpy.dot(test_vec, train_vec) / (_data_norm_new[test] * self._data_norm[train])
        
        outlier_score = result.std(1)
        print 'mean score', outlier_score.mean()
        
        return (outlier_score < self.outlier_cutoff)*-2+1
    
    def decision_function(self, data):
        return numpy.ones((data.shape[0],1))

class OneClassMahalanobis(object):
    def __init__(self, *args, **kwargs):
        pass
    
    def fit(self, data):
        #self.cov = MinCovDet().fit(data)
        self.cov = EmpiricalCovariance().fit(data)
    
    def predict(self, data):
        mahal_emp_cov = self.cov.mahalanobis(data)
        d = data.shape[1]
        thres = scipy.stats.chi2.ppf(0.95, d)
        
        return (mahal_emp_cov > thres).astype(numpy.int32)*-2+1
    
    def decision_function(self, data):
        return numpy.ones((data.shape[0],1))
    
class OneClassGMM(object):
    def __init__(self, *args, **kwargs):
        pass
    
    def fit(self, data):
        #self.cov = MinCovDet().fit(data)
        self.gmm = GMM(2)
        self.gmm.fit(data)
    
    def predict(self, data):
        score = self.gmm.score(data)
        return (score < numpy.percentile(score, 50)).astype(numpy.int32)*-2+1
    
    def decision_function(self, data):
        return numpy.ones((data.shape[0],1))