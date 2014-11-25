import numpy
import vigra
import scipy

from sklearn.svm import OneClassSVM
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, roc_curve, auc
from sklearn.metrics.metrics import roc_curve
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.neighbors.kde import KernelDensity
from sklearn.mixture import GMM
import svmutil3
from sklearn.neighbors import NearestNeighbors

import h5py

class BaseClassifier(object):
    def describe(self):
        desc = self.__class__.__name__ 
        if len(self.params) > 0:
            desc += "_"
            
        desc2 = []
        for p in self.params:
            v = getattr(self, p)
            if isinstance(v, float):
                desc2.append("%s_%7.6f" % (p, v))
            elif isinstance(v, int):
                desc2.append("%s_%d" % (p, v))
            elif isinstance(v, str):
                desc2.append("%s_%s" % (p, v))
            
        return desc + ("--".join(desc2))
class OneClassSVM_SKL(OneClassSVM, BaseClassifier):
    params = ['nu', 'gamma']

class OneClassSVM_LIBSVM(BaseClassifier):
    params = ['nu', 'gamma']
    def __init__(self, *args, **kwargs):
        self.kernel = 2
        self.nu = kwargs['nu']
        self.gamma = kwargs['gamma']
        self.svm_type = 2
    
    def fit(self, data):
        tmp = self._ndarray_to_libsvm_dict(data)
        self.prob = svmutil3.svm_problem([1]*len(data), tmp)
        self.param = svmutil3.svm_parameter("-s %d -t %d -n %f -g %f" % (self.svm_type, self.kernel, self.nu, self.gamma))
        
        self.model = svmutil3.svm_train(self.prob, self.param) 
    
    def predict(self, data):
        p_label, _, p_vals = svmutil3.svm_predict([0]*len(data), self._ndarray_to_libsvm_dict(data), self.model)
        self.df = numpy.array(p_vals)
        return numpy.array(p_label)
    
    def decision_function(self, data):
        return self.df[:,0]
    
    def _ndarray_to_libsvm_dict(self, data):
        return [dict([(f, data[s,f]) for f in range(data.shape[1])]) for s in range(data.shape[0])] 

class OneClassRandomForest(BaseClassifier):
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

class OneClassAngle(BaseClassifier):
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

class OneClassMahalanobis(BaseClassifier):
    params = []
    def __init__(self, *args, **kwargs):
        pass
    
    def fit(self, data):
        #self.cov = MinCovDet().fit(data)
        self.cov = EmpiricalCovariance().fit(data)
    
    def predict(self, data):
        mahal_emp_cov = self.cov.mahalanobis(data)
        d = data.shape[1]
        thres = scipy.stats.chi2.ppf(0.95, d)
        
        self.mahal_emp_cov = mahal_emp_cov
        
        return (mahal_emp_cov > thres).astype(numpy.int32)*-2+1
    
    def decision_function(self, data):
        return self.mahal_emp_cov
    
class OneClassGMM(BaseClassifier):
    params = ['k']
    def __init__(self, *args, **kwargs):
        self.k = kwargs['k']
    
    def fit(self, data, **kwargs):
        #self.cov = MinCovDet().fit(data)
        self.gmm = GMM(self.k)
        self.gmm.fit(data)
        self.training_score = self.gmm.score(data)
        self.direct_threshold = numpy.percentile(self.training_score, 10)
    
    def predict(self, data):
        score = self.gmm.score(data)
        self.score = score
        return (score < self.direct_threshold).astype(numpy.int32)*-2+1
    
    def decision_function(self, data):
        return self.score
    
class OneClassKDE(BaseClassifier):
    params = ["bandwidth"]
    def __init__(self, *args, **kwargs):
        self.bandwidth = kwargs["bandwidth"]
    
    def fit(self, data, **kwargs):
        #self.train_data = data
        self.kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
        self.kde.fit(data)
        self.training_score = self.kde.score_samples(data)
        self.direct_thresh = numpy.percentile(self.training_score, 10)
    
    def predict(self, data):
        score = self.kde.score_samples(data)
        self.score = score
        return (score < self.direct_thresh).astype(numpy.int32)*-2+1
    
    def decision_function(self, data):
        return self.score
    
OneClassSVM = OneClassSVM_LIBSVM

if __name__ == "__main__":
    import pylab
    N = 1000
    D = 2
    TD = 20
    x_train = numpy.random.randn(N,TD)
    x_train[:N/2,:]+=4
    x_train[:,D:] = numpy.random.rand(N,TD-D) 
    x_test = numpy.random.rand(1000,TD)
    x_test[:,0] = x_test[:,0] * (x_train[:,0].max() - x_train[:,0].min()) + x_train[:,0].min() 
    x_test[:,1] = x_test[:,1] * (x_train[:,1].max() - x_train[:,1].min()) + x_train[:,1].min() 
    
    #a = OneClassSVM_LIBSVM(gamma=0.02, nu=0.1, bandwidth=1, k=5)
    a = OneClassKDE(gamma=0.02, nu=0.1, bandwidth=1.56, k=5)
    a.fit(x_train)
    p = a.predict(x_test)
    
    plot_dim = (0,1)
    
    pylab.plot(x_train[:, plot_dim[0]], x_train[:,plot_dim[1]], 'g.')
    pylab.plot(x_test[:, plot_dim[0]], x_test[:,plot_dim[1]], 'bx')
    pylab.plot(x_test[p==-1, plot_dim[0]], x_test[p==-1,plot_dim[1]], 'ro', markerfacecolor="none", markeredgecolor='r')
    
    pylab.show()
    