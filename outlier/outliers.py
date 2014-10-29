import matplotlib
import numpy
import vigra
import pylab
import scipy

import sys; sys.path.append('C:/Users/sommerc/cellh5/pysrc/')

from sklearn.svm import OneClassSVM
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, roc_curve, auc
from sklearn.metrics.metrics import roc_curve
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_distances, pairwise_kernels
from sklearn.metrics import silhouette_score
import sklearn.cluster
import sklearn.mixture

import cellh5
import cellh5_analysis
import cPickle as pickle
from numpy import recfromcsv
import pandas
import time
from matplotlib.mlab import PCA as PCAold
from scipy.stats import nanmean
import datetime
import os
from itertools import chain

import iscatter
from cellh5 import CH5File
from collections import defaultdict, OrderedDict
from matplotlib.colors import rgb2hex

from matplotlib import rcParams , RcParams
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Arial']
# rcParams['pdf.fonttype'] = 42
# rcParams['ps.usedistiller' ] = 'xpdf'
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.size'] = 20
# rcParams['font.sans-serif'] = ['Arial']
# rcParams['xtick.labelsize'] = 16 
# rcParams['ytick.labelsize'] = 16 
# rcParams['axes.labelsize'] = 16
# rcParams['lines.color'] = 'white'
# rcParams['patch.edgecolor'] = 'white'
# rcParams['text.color'] = 'white'
# rcParams['axes.facecolor'] = 'black'
# rcParams['axes.edgecolor'] = 'white'
# rcParams['axes.labelcolor'] = 'white'
# rcParams['xtick.color'] = 'white'
# rcParams['ytick.color'] = 'white'
# rcParams['grid.color'] = 'white'
# rcParams['figure.facecolor'] = 'black'
# rcParams['figure.edgecolor'] = 'black'
# rcParams['savefig.facecolor'] = 'black'
# rcParams['savefig.edgecolor'] = 'none'

from mpl_toolkits.axes_grid1 import make_axes_locatable




class OutlierDetection(cellh5_analysis.CellH5Analysis):
    def __init__(self, name, mapping_files, cellh5_files, training_sites=None, rows=None, cols=None, locations=None, gamma=None, nu=None, pca_dims=None, kernel=None):
        cellh5_analysis.CellH5Analysis.__init__(self, name, mapping_files, cellh5_files, sites=training_sites, rows=rows, cols=cols, locations=locations)
        self.gamma = gamma
        self.nu = nu
        self.pca_dims = pca_dims
        self.kernel = kernel
        self.feature_set = 'PCA'
        #self.output_dir += "/-o%s-p%d-k%s-n%f-g%f" % (self.feature_set, self.pca_dims, self.kernel, self.nu, self.gamma)
        try:
            os.makedirs(self.output_dir)
        except:
            pass
        
    def set_gamma(self, gamma):
        self.gamma = gamma
    def set_kernel(self, kernel):
        self.kernel = kernel
    def set_nu(self, nu):
        self.nu = nu
    def set_pca_dims(self, pca_dims):
        self.pca_dims = pca_dims
        
    def write_readme(self):
        with open(self.output('readme.txt'), 'w') as f:
            for attr in ('gamma', 'nu', 'pca_dim', 'kernel', 'feature_set'):
                if hasattr(self, attr):
                    tmp = getattr(self, attr)
                    f.write("%s\t%r\n" % (attr, tmp))

    def train(self, train_on=('neg',), classifier_class=OneClassSVM):
        if DEBUG:
            print 'Training OneClass Classifier for', train_on, 'on', self.feature_set
        training_matrix = self.get_data(train_on, self.feature_set)
        
        if self.feature_set == 'Object features' :
            training_matrix = self.normalize_training_data(training_matrix)
        
        self.train_classifier(training_matrix, classifier_class)
        if DEBUG:
            pass#print '%04.2f %%' % (100 * float(self.classifier.support_vectors_.shape[0]) / training_matrix.shape[0]), 'support vectors'
        
    def predict(self, test_on=('target', 'pos', 'neg')):
        if DEBUG:
            print 'Predicting OneClass Classifier for', self.feature_set
        testing_matrix_list = self.mapping[self.mapping['Group'].isin(test_on)][['Well', 'Site', self.feature_set, "Gene Symbol", "siRNA ID"]].iterrows()

        predictions = {}
        distances = {}
        
        log_file_handle = open(self.output('_outlier_detection_log.txt'), 'w')
        
        for idx, (well, site, tm, t1, t2) in  testing_matrix_list:
            if DEBUG:
                print well, site, t1, t2, "->",
            log_file_handle.write("%s\t%d\t%s\t%s" % (well, site, t1, t2))
            if isinstance(tm, (float,)) or (tm.shape[0] == 0):
                predictions[idx] = numpy.zeros((0, 0))
                distances[idx] = numpy.zeros((0, 0))
                log_file_handle.write("\t0 / 0 outliers\t0.00\n")
            else:
                if self.feature_set == 'Object features':
                    tm = self._remove_nan_rows(tm)
                pred, dist = self.predict_with_classifier(tm[:, self.rfe_selection], log_file_handle)
                predictions[idx] = pred
                distances[idx] = dist
        log_file_handle.close()
        self.mapping['Predictions'] = pandas.Series(predictions)
        self.mapping['Hyperplane distance'] = pandas.Series(distances)
        
    def estimate_gamma_by_sv(self, X, nu):
        max_training_samples = 10000
        
        xx = []
        yy = []
        for gamma in [2**g for g in range(-16, 2)]:  
            classifier = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)  
            
            
            idx = range(X.shape[0])
            numpy.random.seed(1)
            numpy.random.shuffle(idx)
            
            X = X[idx[:min(max_training_samples, X.shape[0])], :]
            
            classifier.fit(X)
            s_frac = (classifier.support_vectors_.shape[0] / float(len(X)) ) 
            print " SV fraction ", nu, gamma, "\t\t::", s_frac*100, "%"
            
            if s_frac > 0.99:
                break
            xx.append(numpy.log2(gamma))
            yy.append(s_frac)
        
        yy = numpy.array(yy)
        ind = numpy.argmax(numpy.diff(yy[yy < 1.67 * nu]))
        
        pylab.figure()
        pylab.plot(xx,yy,'y')
        pylab.plot(xx[ind], yy[ind], 'ro', markerfacecolor='none', markeredgecolor='r', lw=3)
        #pylab.show()
        
        
       
        print ' best gamma at', xx[ind], 2**xx[ind], 'with SV frac', yy[ind] 
        
        return 2**xx[ind]
        
    
    def estimate_gamma2(self, matrix):
        result = OrderedDict()
        all_gammas = [2**(ee) for ee in range(0, 30)]
        for gamma in all_gammas:
            kernel_matrix = pairwise_kernels(matrix, metric='rbf', gamma=gamma)
            mask = numpy.triu(numpy.ones(kernel_matrix.shape),1)
            l = mask.sum()
            
            temp = (kernel_matrix * mask)
            kappa = temp.sum() / float(l)
            s_2   = ((temp - kappa)**2).sum() / float(l-1)
            
            result[gamma] = indicator = s_2 / (kappa + 0.000001)
            
        for g, v in result.items():
            print 'find gamma:', g, v
        best_gamma = all_gammas[numpy.argmax(result.values())]
        
        print 'best gamma', best_gamma
        pylab.plot(numpy.log2(all_gammas), result.values())
        #pylab.show()
        return best_gamma
        

            
    def train_classifier(self, training_matrix, classifier_class=OneClassSVM):
        if self.kernel == 'rbf':
            if self.gamma is None:
                self.gamma = self.estimate_gamma_by_sv(training_matrix, self.nu)
        self.classifier = classifier_class(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
        print '  Using', classifier_class, self.kernel, self.nu, self.gamma
        if self.kernel == 'linear':
            max_training_samples = 10000
            idx = range(training_matrix.shape[0])
            numpy.random.seed(1)
            numpy.random.shuffle(idx)
            self.classifier.fit(training_matrix[idx[:min(max_training_samples, training_matrix.shape[0])],:])
            self.rfe_selection = numpy.ones((training_matrix.shape[1],), dtype=numpy.bool)
            
            # RFE
            if False:
                rfe = RFE(self.classifier, 30, step=1)
                rfe.fit2(training_matrix[idx[:min(max_training_samples, training_matrix.shape[0])],:])
                if DEBUG:
                    print rfe.ranking_
                    print rfe.support_
                self.rfe_selection = rfe.support_
                self.classifier = rfe.estimator_
        else:
            max_training_samples = 10000
            idx = range(training_matrix.shape[0])
            numpy.random.seed(1)
            numpy.random.shuffle(idx)
            self.classifier.fit(training_matrix[idx[:min(max_training_samples, training_matrix.shape[0])],:])
            self.rfe_selection = numpy.ones((training_matrix.shape[1],), dtype=numpy.bool)
            
    def compute_outlyingness(self):
        def _outlier_count(x):
            res = numpy.float32((x == -1).sum()) / len(x)
            return res
            
        res = pandas.Series(self.mapping[self.mapping['Group'].isin(('target', 'pos', 'neg'))]['Predictions'].map(_outlier_count))
        self.mapping['Outlyingness'] = res
        
    def predict_with_classifier(self, test_matrix, log_file_handle=None):
        prediction = self.classifier.predict(test_matrix)
        distance = self.classifier.decision_function(test_matrix)[:,0]
        log = "\t%d / %d outliers\t%3.2f" % ((prediction == -1).sum(),
                                             len(prediction),
                                             (prediction == -1).sum() / float(len(prediction)))
        if DEBUG:
            print log
        if log_file_handle is not None:
            log_file_handle.write(log+"\n")
        return prediction, distance
        

    def cluster_get_k(self, training_data):
        max_k = 10
        bics = numpy.zeros(max_k)
        bics[0] = 0
        if True:
            if True:
                for k in range(1, max_k):
                    gmm = sklearn.mixture.GMM(k, covariance_type='full')
                    gmm.fit(training_data)
                    cluster_assignment = gmm.predict(training_data)
                    res = numpy.array([numpy.count_nonzero(cluster_assignment==kk) for kk in numpy.unique(cluster_assignment)]) / float(len(cluster_assignment))
                    b = gmm.bic(training_data)
                    bics[k] = -b
                    
                    print ' cluster k', k, 'distro', res, 'bic', b
                    
                bicd = numpy.diff(bics[1:])
                bicd = bicd > numpy.mean(bicd)
                kkk = 1
                for bd in bicd:
                    if bd:
                        kkk+=1
                    else:
                        break
                    
                bics /= 10000.0
                       
                K = numpy.argmax(numpy.diff(numpy.diff(numpy.diff(bics[1:]))))
                #K = numpy.argmin(bics[1:])
                K = kkk
                fig = pylab.figure()
                pylab.plot(range(1, max_k), bics[1:], lw=2, label=r"$-2 \ln \left( \mathcal{L}(x \vert \mathrm{G}_k) \right) + \vert \mathrm{G}_k \vert \ln(n)$")
                pylab.xlim(0, len(bics)+0.5)
                pylab.plot(K, bics[K], 'r', marker='o', markersize=20, markeredgecolor="r", markerfacecolor="none", ls="", label="Knee point")
                pylab.xlabel("Number of clusters (k)")
                pylab.ylabel("Bayesian Information Criterion BIC ($\\times -10^4$)")
                ax = pylab.gca()
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()
                pylab.legend(loc=4)
                
                pylab.tight_layout()
                pylab.savefig(self.output("outlier_clustering_bic.pdf"))
            else:
                K=3
            
            
        else:
            bics[0] = -2
            # Sillouette
            pd = pairwise_distances(training_data)
            for k in range(2, max_k):
                cluster = sklearn.cluster.KMeans(k, init='random')
                cluster_assignment = cluster.fit_predict(training_data)
                res = numpy.array([numpy.count_nonzero(cluster_assignment==k) for k in numpy.unique(cluster_assignment)]) / float(len(cluster_assignment))
                b = silhouette_score(pd, cluster_assignment, metric='precomputed')
                print ' cluster k', k, 'distro', res, 'silhouette', b
                bics[k] = b
            K = numpy.argmax(bics[2:])
            K+=2
            
        

        return K
    
    def cluster_outliers(self, cluster_all=False):
        # setup clustering
        if DEBUG:
            print 'Setup clustering'
        training_data = []
        label = []
        for _ , (data, prediction, w, p, c, g, s, ch5i) in self.mapping[['PCA', 'Predictions', 'Well', 'Site', 'Object count', 'Gene Symbol', 'siRNA ID', 'CellH5 object index']].iterrows():
            print w,p,g,s,c, type(data)
            if c > 0:
                if not cluster_all:
                    data_i = data[prediction == -1, :]
                else:
                    data_i = data
                training_data.append(data_i)
                label.append([(w,p,g,s, cc) for cc in ch5i])
            
        training_data = numpy.concatenate(training_data)
        label = numpy.concatenate(label)
        
        
        
        #k = self.cluster_get_k(training_data)
        k=4
        if DEBUG:
            print 'Run clustering for training data shape ', training_data.shape, 'with k = ', k
        km = sklearn.mixture.GMM(k, covariance_type='full')
        km.fit(training_data)
        
        cluster_vectors = {}
        
        cluster_teatment_vectors = OrderedDict()
        
        if DEBUG:
            print 'Apply Clustering'
        for idx , (data, prediction, c, g, s, ch5_idx, plate, well, site)  in self.mapping[['PCA', 'Predictions', 'Object count', 'Gene Symbol', 'siRNA ID', 'CellH5 object index', 'Plate', 'Well', 'Site']].iterrows():        
            if c > 0:
                if cluster_all:
                    cluster_predict = km.predict(data)
                else:
                    cluster_predict = km.predict(data) + 1
                    cluster_predict[prediction==1] = 0
                if False:
                    c5f = self.cellh5_handles[plate]
                    c5p = c5f.get_position(well, str(site))
                    img_list = []
                    for j in range(k+1):
                        img_j = c5p.get_gallery_image_matrix(ch5_idx[cluster_predict == j], (10, 10))
                        img_list.append(img_j)
                        img_list.append(numpy.ones((img_j.shape[0],5))*255)
                        vigra.impex.writeImage(img_j.swapaxes(1,0), self.output('single_%s_%s_%s_%s_cluster_%d.png' % (g,s,well, site,j)))
                    img_list = numpy.concatenate(img_list[:-1],1)
                    vigra.impex.writeImage(img_list.swapaxes(1,0), self.output('all_%s_%s_cluster_class.png' % (well, site)))
            else:
                cluster_predict = numpy.array([])
            cluster_vectors[idx] = cluster_predict
#             cluster_teatment_vectors['%s\n%s' % (g, s)] = cluster_predict
            cluster_teatment_vectors['%s' % (g, )] = cluster_predict
            
        if cluster_all:
            self.mapping['Outlier clustering'] = pandas.Series(cluster_vectors)
        else:
            self.mapping['Outlier clustering'] = pandas.Series(cluster_vectors)
        
        # make plot
        if False:
            cluster_teatment_vectors = {}
            treatment_group = self.mapping.groupby(['Gene Symbol','siRNA ID'])
            for tg in treatment_group:
                treatment = "%s - %s" % tg[0]
                wells = list(tg[1]['Well'])
                cluster_vecs = numpy.concatenate(list(tg[1]['Outlier clustering']))  
                if treatment in cluster_teatment_vectors:
                    print 'Error', 
                if len(cluster_vecs) < 8:
                    continue
                cluster_teatment_vectors[treatment] = cluster_vecs
        
            fig = pylab.figure(figsize=(22,8))
            ax = pylab.gca()
            rcParams['ytick.labelsize'] = 6
            rcParams['xtick.labelsize'] = 6
            rcParams['axes.labelsize'] = 8
             
            if cluster_all:
                labels = ["Cluster %d" % d for d in range(1,k+1)]
            else:
                labels = ['Inlier',] + ["Cluster %d" % d for d in range(1,k+1)]
                
            
            treatmentStackedBar(ax, cluster_teatment_vectors, {0: COLOR_LUT_6['green'], 
                                                               1: COLOR_LUT_6['red'], 
                                                               2: COLOR_LUT_6['yellow'],  
                                                               3: COLOR_LUT_6['blue'],  
                                                               4: COLOR_LUT_6['magenta'],   
                                                               5: COLOR_LUT_6['cyan'],   
                                                               6:'w', 
                                                               7:'k'}, labels, 1)
            
            
            rcParams['ytick.labelsize'] = 14
            rcParams['xtick.labelsize'] = 14
            rcParams['axes.labelsize'] = 18
            
            pylab.savefig(self.output("outlier_clustering.pdf"))
            #pylab.show()
        #self.ward_cluster(training_data, label)
        
    def ward_cluster(self, training_data, label):
        rand_idx = range(training_data.shape[0])
        numpy.random.shuffle(rand_idx)
        rand_idx = rand_idx[:100]
        
        training_data = training_data[rand_idx, :]
        label = label[rand_idx] 
        
        
        import scipy.cluster.hierarchy as sch

        D = pairwise_distances(training_data, metric="cityblock")
        # Compute and plot first dendrogram.
        fig = pylab.figure(figsize=(8,8))
        ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
        Y = sch.linkage(D, method='centroid')
        Z1 = sch.dendrogram(Y, orientation='right')
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Compute and plot second dendrogram.
        ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
        Y = sch.linkage(D, method='centroid')
        Z2 = sch.dendrogram(Y)
        ax2.set_xticks([])
        ax2.set_yticks([])
                
        # Plot distance matrix.
        axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
        idx1 = Z1['leaves']
        idx2 = Z2['leaves']
        D = D[idx1,:]
        D = D[:,idx2]
        im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
        axmatrix.set_xticks(range(D.shape[0]))
        axmatrix.set_yticks(range(D.shape[1]))
        axmatrix.set_xticklabels(["%s %s %s %s %s" % ( w,str(p),g,s, cc) for w,p,g,s, cc in label[idx1]], rotation=90)
        axmatrix.set_yticklabels(["%s %s %s %s %s" % ( w,str(p),g,s, cc) for w,p,g,s, cc in label[idx2]])
        
        # Plot colorbar.
        axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
        pylab.colorbar(im, cax=axcolor)

        # Display and save figure.
        pylab.show()
        
    def evaluate_roc(self):
        y_true = []
        y_est = []
        y_fixed = []
        for i, each_row in self.mapping.iterrows():
            plate_name = each_row['Plate']
            well = each_row['Well']
            site = each_row['Site']
            treatment = tuple(each_row[['Gene Symbol', 'siRNA ID']]) 
            outlier_est = -numpy.array(each_row['Hyperplane distance'])
            outlier_prediction = numpy.array(each_row['Predictions'])
            outlier_prediction_b = outlier_prediction == -1
            outlier_prediction_2 = ((outlier_prediction*-1)+1)/2
            
            cellh5_idx = list(each_row['CellH5 object index'])
            c5f = self.cellh5_handles[plate_name]
            c5p = c5f.get_position(well, str(site))
            class_prediction = c5p.get_class_prediction()['label_idx'][cellh5_idx]
            class_prediction_b = (class_prediction > 3).astype(numpy.uint8)
            
            y_true.extend(class_prediction_b)
            y_est.extend(outlier_est)
            y_fixed.extend(outlier_prediction_2)
            
        fpr, tpr, th = roc_curve(y_true, y_est)
        roc_auc = auc(fpr, tpr)
        
        fpr_f, tpr_f, th  = roc_curve(y_true, y_fixed)
              
        pylab.figure()
        ax = pylab.subplot(111)
        ax.plot(fpr, tpr, 'g-', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        
        ax.plot(fpr_f, tpr_f, 'ko', lw=2, label='one-class SVM decision' % roc_auc)
        
        ax.plot([0, 1], [0, 1], 'k--')
        pylab.xlim([0.0, 1.0])
        pylab.ylim([0.0, 1.0])
        pylab.xlabel('False Positive Rate')
        pylab.ylabel('True Positive Rate')
        pylab.title('Receiver operating characteristic (ROC)')
        pylab.legend(loc="lower right")
        
        pylab.tight_layout()
        pylab.savefig(self.output('outlier_roc.png' ))
        
        return fpr, tpr, th, roc_auc, fpr_f, tpr_f  
    
    def export_to_file(self, n_classes=6):
        plate_name = str(self.mapping["Plate"].iloc[0])
        c5f = self.cellh5_handles[plate_name]
        class_names = c5f.class_definition('primary__primary')['name']
        assert len(class_names) == n_classes
        
        fh = open(self.output("Classification_vs_outlier.txt"), 'wb')
        fh.write("Plate\tWell\tGene\tsiRNA\tOutlyingness\tCount\t" + "\t".join(class_names) + "\n")
        
        for (plate_name, well, gene, sirna), each_row in self.mapping[self.mapping['Object count'] > 0].groupby(['Plate', 'Well', 'Gene Symbol', 'siRNA ID']):
            c5f = self.cellh5_handles[plate_name]
            outlyingness = (each_row['Outlyingness'] * each_row['Object count']).sum() / each_row['Object count'].sum()
            
            class_count = numpy.zeros((n_classes,), dtype=numpy.float32)
            obj_count = 0
            for _, (site, cellh5_idx, o_count) in each_row[['Site', 'CellH5 object index', 'Object count']].iterrows():
                c5p = c5f.get_position(well, str(site))
                class_prediction = c5p.get_class_prediction()['label_idx'][cellh5_idx]
                class_count += numpy.bincount(class_prediction.astype(numpy.int32), minlength=n_classes) / float(len(class_prediction))
                obj_count += o_count
                
            class_count /= class_count.sum()
            fh.write(("%s\t%s\t%s\t%s\t%f\t%d" + "\t%f"*n_classes + "\n") % ((plate_name, well, gene, sirna, float(outlyingness), obj_count, ) + tuple([class_count[kk] for kk in range(n_classes)])))   

        fh.close()
            
            

    def evaluate_outlier_detection(self):
        # gather information
        plate_name = str(self.mapping["Plate"].iloc[0])
        c5f = self.cellh5_handles[plate_name]
        class_names = c5f.class_definition('primary__primary')['name']
           
        acc = []
        
        cm = numpy.zeros((len(class_names),2), 'float32')
        cm_2 = numpy.zeros((2,2), 'float32')
        cm_3 = numpy.zeros((len(class_names),5), 'float32')
        
        for i, each_row in self.mapping.iterrows():
            plate_name = each_row['Plate']
            well = each_row['Well']
            site = each_row['Site']
            if each_row['Object count'] == 0:
                print "\tOmmiting (count ==0)", plate_name, well, site
                continue
            treatment = tuple(each_row[['Gene Symbol', 'siRNA ID']]) 
            outlier_prediction = numpy.array(each_row['Predictions'])
            
            cluster_prediction = numpy.array(each_row['Outlier clustering'])
            
            outlier_prediction_b = outlier_prediction == -1
            outlier_prediction_2 = ((outlier_prediction*-1)+1)/2
            
            cellh5_idx = list(each_row['CellH5 object index'])
            c5f = self.cellh5_handles[plate_name]
            c5p = c5f.get_position(well, str(site))
            class_prediction = c5p.get_class_prediction()['label_idx'][cellh5_idx]
            class_prediction_b = class_prediction > 1
            
            for c, o, cl in zip(class_prediction, outlier_prediction_2, cluster_prediction):
                cm[int(c), int(o)]     += 1
                cm_2[int(c>1), int(o)] += 1
                cm_3[int(c), int(cl)] += 1
                
                
            acc.append(accuracy_score(class_prediction_b.astype('uint8'), outlier_prediction_b.astype('uint8')))
        
        for r in range(cm.shape[0]):
            cm[r,:] = cm[r,:] / float(cm[r,:].sum()) 
        
        for r in range(cm_2.shape[0]):
            cm_2[r,:] = cm_2[r,:] / float(cm_2[r,:].sum())
             
        for r in range(cm_3.shape[0]):
            cm_3[r,:] = cm_3[r,:] / float(cm_3[r,:].sum()) 
        
        print cm
        print cm_2
        print cm_3
        
#         self.plot_confusion(cm, ['Interphase', 'Prometaphase', 'Metaphase', 'Anaphase', 'Grape', 'Prometaphase\narrest', 'Polylobed'], ['Inlier', 'Outlier'], 'vs_classi_full', 4, print_values=True)
#         self.plot_confusion(cm_2, ['Wildtype', 'Phenotype'], ['Inlier', 'Outlier'], 'vs_classi_collapsed', print_values=True)
#         self.plot_confusion(cm_3, ['Interphase', 'Prometaphase', 'Metaphase', 'Anaphase', 'Grape', 'Prometaphase\narrest', 'Polylobed'], ['Inlier']+['Cluster %d' % d for d in range(1,cm_3.shape[1]+1) ], 'vs_class', 4, print_values=True)
#         
        self.plot_confusion(cm, class_names, ['Inlier', 'Outlier'], 'vs_classi_full', 2, print_values=True)
        self.plot_confusion(cm_2, ['Wildtype', 'Phenotype'], ['Inlier', 'Outlier'], 'vs_classi_collapsed', print_values=True)
        self.plot_confusion(cm_3, class_names, ['Inlier']+['Cluster %d' % d for d in range(1,cm_3.shape[1]+1) ], 'vs_class', 5, print_values=True)
        
        
        return numpy.mean(acc), cm, cm_2, cm_3
            
    def plot_confusion(self, cm, row_names, col_names, title='', row_sep=1, col_sep=1, print_values=False, lw=3):
        #rcParamsBackup = rcParams.copy()
        
#         rcParams['lines.color'] = 'white'
#         rcParams['patch.edgecolor'] = 'white'
#         rcParams['text.color'] = 'white'
#         rcParams['axes.facecolor'] = 'black'
#         rcParams['axes.edgecolor'] = 'black'
#         rcParams['axes.labelcolor'] = 'white'
#         rcParams['xtick.color'] = 'white'
#         rcParams['ytick.color'] = 'white'
#         rcParams['grid.color'] = 'white'
#         rcParams['figure.facecolor'] = 'black'
#         rcParams['figure.edgecolor'] = 'black'
#         rcParams['savefig.facecolor'] = 'black'
#         rcParams['savefig.edgecolor'] = 'none'
        
        pylab.figure(figsize=(7,7))
        ax = pylab.subplot(111)
        
        im = ax.pcolor(cm, cmap=YlBlCMap, vmin=0, vmax=1)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.5)

        cbar = pylab.colorbar(im, cax=cax, ticks=[0, 0.5, 1])

        cbar.ax.set_yticklabels(['0', '0.5', '1'])
        ax.set_xticks(numpy.arange(cm.shape[1]) + 0.5, minor=False)
        ax.set_yticks(numpy.arange(cm.shape[0]) + 0.5, minor=False)
        ax.set_xticklabels(col_names)
        ax.set_yticklabels(row_names)
        ax.invert_yaxis()
        for c in range(cm.shape[0]):
            for o in range(cm.shape[1]):
                if not print_values:
                    continue
#                 t = ax.text(o + 0.5, c + 0.5,  "%3.2f" % cm[c,o], horizontalalignment='center', verticalalignment='center', fontsize=20)
#                 if cm[c,o] > 0.3*cm.max():
#                     t.set_color('k')
                    
        ax.hlines(row_sep,0, cm.shape[1], colors='k', lw=lw)
        ax.vlines(col_sep,0, cm.shape[0], colors='k', lw=lw)
        
        #ax.set_title(title)
        if cm.shape[0] == cm.shape[1]:
            ax.set_aspect(1)
        
        cbar.ax.axis('off')
        ax.axis('off')
        
        pylab.tight_layout()

        pylab.savefig(self.output('outlier_classification_confusion_%s.pdf' % title))
        pylab.savefig(self.output('outlier_classification_confusion_%s.png' % title))
        #pylab.show()
        

    def plot(self):
        f_x = 1
        f_y = 0
        
        x_min, y_min = 1000000, 100000
        x_max, y_max = -100000, -100000
        ch5_file = cellh5.CH5File(self.cellh5_file)
        
        for i in range(len(self.mapping['PCA'])):
            data = self.mapping['PCA'][i]
            prediction = self.mapping['Predictions'][i]
            # print self.mapping['siRNA ID'][i], data.shape
            
            if self.mapping['Group'][i] in ['pos', 'target']:
                pylab.scatter(data[prediction == -1, f_x], data[prediction == -1, f_y], c='red', marker='d', s=42)
                pylab.scatter(data[prediction == 1, f_x], data[prediction == 1, f_y], c='white', marker='d', s=42)
            else:
                pylab.scatter(data[prediction == -1, f_x], data[prediction == -1, f_y], c='white', s=42)
                pylab.scatter(data[prediction == 1, f_x], data[prediction == 1, f_y], c='white', s=42)
            
            x_min_cur, x_max_cur = data[:, f_x].min(), data[:, f_x].max()
            y_min_cur, y_max_cur = data[:, f_y].min(), data[:, f_y].max()
        
            x_min = min(x_min, x_min_cur)
            y_min = min(y_min, y_min_cur)
            x_max = max(x_max, x_max_cur)
            y_max = max(y_max, y_max_cur)
            
            
            
            import vigra
            well = str(self.mapping['Well'][i])
            site = str(self.mapping['Site'][i])
            ch5_pos = ch5_file.get_position(well, site)
            
            img = ch5_pos.get_gallery_image_matrix(numpy.where(prediction == -1)[0], (10, 10))
            vigra.impex.writeImage(img.swapaxes(1,0), '%s_%s_outlier.png' % (well, site))
            
            img = ch5_pos.get_gallery_image_matrix(numpy.where(prediction == 1)[0], (10, 10))
            vigra.impex.writeImage(img.swapaxes(1,0), '%s_%s_inlier.png' % (well, site))

            
        x_min = -12
        y_min = -25
        
        x_max = 42
        y_max = 42    
            
        xx, yy = numpy.meshgrid(numpy.linspace(x_min, x_max, 100), numpy.linspace(y_min, y_max, 100))
        # Z = self.classifier.decision_function(numpy.c_[xx.ravel(), yy.ravel()])
        matrix = numpy.zeros((100 * 100, self.pca_dims))
        matrix[:, f_x] = xx.ravel()
        matrix[:, f_y] = yy.ravel()
        
        
        Z = self.classifier.decision_function(matrix)
        Z = Z.reshape(xx.shape)
        # print Z
        # Z = (Z - Z.min())
        # Z = Z / Z.max()
        # print Z.min(), Z.max()
        # Z = numpy.log(Z+0.001)
        
        pylab.contourf(xx, yy, Z, levels=numpy.linspace(Z.min(), Z.max(), 8), cmap=pylab.matplotlib.cm.Greens, hold='on', alpha=0.5)
        # a = pylab.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
        # pylab.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')
        
        
        pylab.axis('tight')
        
        pylab.xlim((x_min, x_max))
        pylab.ylim((y_min, y_max))
        pylab.axis('off')
        pylab.show(block=True)
        
    def make_heat_map(self):
        if DEBUG:
            print 'Make heat map plot'
        for plate_name in self.cellh5_files.keys():
            rows = sorted(numpy.unique(self.mapping['Row']))
            cols = sorted(numpy.unique(self.mapping['Column']))
            
            target_col = 'Outlyingness'
            fig = pylab.figure(figsize=(len(cols)*0.8 +4, len(rows)*0.6+2))
            
            heatmap = numpy.zeros((len(rows), len(cols)), dtype=numpy.float32)
            
            for r_idx, r in enumerate(rows):
                for c_idx, c in enumerate(cols):
                    target_value = self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c) & (self.mapping['Plate'] == plate_name)][target_col]
                    target_count = self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c) & (self.mapping['Plate'] == plate_name)]['Object count']
                    target_grp =   self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c) & (self.mapping['Plate'] == plate_name)]['Group']
                    
                    if target_count.sum() == 0:
                        value = -1
                    else:
                        value = (target_value * target_count).sum() / float(target_count.sum())
                        # value = nanmean(target_value) 
                    
                    
                    if numpy.isnan(value):
                        print 'Warning: there are nans...'
                    if target_count.sum() > 0:
                        heatmap[r_idx, c_idx] = value
    #                 else:
    #                     heatmap[r_idx, c_idx] = -1
    #                     
    #                 if target_grp.iloc[0] in ('neg', 'pos'):
    #                     heatmap[r_idx, c_idx] = -1
                    
            cmap = pylab.matplotlib.cm.Greens
            cmap.set_under(pylab.matplotlib.cm.Oranges(0))
            # cmap.set_under('w')
                 
            if DEBUG:       
                print 'Heatmap', heatmap.max(), heatmap.min()    
            #fig = pylab.figure(figsize=(40,25))
            
            ax = pylab.subplot(111)
            pylab.pcolor(heatmap, cmap=cmap, vmin=0, vmax=1)
            pylab.colorbar()
            ax.set_xlim(0,len(cols))
    
            for r_idx, r in enumerate(rows):
                for c_idx, c in enumerate(cols):
                    try:
                        count = self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c) & (self.mapping['Plate'] == plate_name)]['Object count'].sum()
                        text_grp = str(self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c) & (self.mapping['Plate'] == plate_name)]['Group'].iloc[0]) + " " + ("%0.2f" % heatmap[r_idx, c_idx])[1:]
                        text_gene = str(self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c) & (self.mapping['Plate'] == plate_name)]['siRNA ID'].iloc[0])
                        text_gene2 = str(self.mapping[(self.mapping['Row'] == r) & (self.mapping['Column'] == c) & (self.mapping['Plate'] == plate_name)]['Gene Symbol'].iloc[0])
                        
                    except IndexError:
                        text_grp = "empty"
                        text_gene = "empty"
                        text_gene2 = "empty"
                        count = -1
                        
                    t = pylab.text(c_idx + 0.5, r_idx + 0.5, '%s\n%s\n%s\n%d' % (text_grp, text_gene, text_gene2, count), horizontalalignment='center', verticalalignment='center', fontsize=8)
                    if heatmap[r_idx, c_idx] > 0.3:
                        t.set_color('w')
                        
            # put the major ticks at the middle of each cell
            ax.set_xticks(numpy.arange(heatmap.shape[1]) + 0.5, minor=False)
            ax.set_yticks(numpy.arange(heatmap.shape[0]) + 0.5, minor=False)
            
            # want a more natural, table-like display
            # ax.invert_yaxis()
            ax.xaxis.tick_top()
            
            ax.set_xticklabels(list(cols), minor=False)
            ax.set_yticklabels(list(rows), minor=False)
            
            for label in ax.get_xticklabels() + ax.get_yticklabels(): 
                 label.set_fontsize(16) 
            
            #pylab.title("%s %s" % (self.name, plate_name))
            
            pylab.tight_layout()
            pylab.savefig(self.output('outlier_heatmap_%s.pdf' % plate_name))
        
    def make_hit_list_single_feature(self, feature_name):
        if DEBUG:
            print 'Make hit single list plot for', feature_name
        group_on = ['Plate', 'siRNA ID', 'Gene Symbol']
        
        f =  self.cellh5_handles.values()[0]
        feature_idx = f.get_object_feature_idx_by_name('primary__primary', feature_name)
        
        feature_agg_mean = lambda x: numpy.sum([numpy.sum(y[:, feature_idx]) for y in x]) / numpy.sum([len(y[:, feature_idx]) for y in x])
        feature_agg_median = lambda x: numpy.median(list(chain.from_iterable([y[:, feature_idx] for y in x])))
        feature_agg = feature_agg_median
        
        min_object_count_site = 0
        min_object_coutn_group = 15
        # get global values of all plates        
        group = self.mapping[(self.mapping['Object count'] > min_object_count_site) ].groupby(group_on)
        
        overall_min = group['Object features'].apply(feature_agg).min()
        overall_max = group['Object features'].apply(feature_agg).max()
        
        neg_group = self.mapping[(self.mapping['Object count'] > min_object_count_site) & (self.mapping['Group'] == 'neg')].groupby(group_on)
        neg_mean = neg_group['Object features'].apply(feature_agg).mean()
        neg_std = neg_group['Object features'].apply(feature_agg).std()
        
        pos_group = self.mapping[(self.mapping['Object count'] > min_object_count_site) & (self.mapping['Group'] == 'pos')].groupby(group_on)
        pos_mean = pos_group['Object features'].apply(feature_agg).mean()       
        
        #iterate over plates and make hit list figure
        for plate_name in self.mapping_files.keys():
            group = self.mapping[(self.mapping['Object count'] > min_object_count_site) & (self.mapping['Plate'] == plate_name)].groupby(['Well', 'siRNA ID', 'Gene Symbol'])
            
            means = group['Object features'].apply(feature_agg)
            means.sort()
            
            genes = []
            stds = []
            values = []
            for g, m in means.iteritems():
                count = self.mapping['Object count'][group.groups[g]].sum()
                if count > min_object_coutn_group:
                    stds.append(0)
                    values.append(m)
                    genes.append("%s %s %s (%d)" % (g + (count,)))
            n = len(values)
            fig = pylab.figure(figsize=(len(genes)/6 + 3, 10))
            ax = pylab.subplot(111)
            ax.errorbar(range(n), values, yerr=stds, fmt='o', markeredgecolor='none')
            ax.set_xticks(range(n), minor=False)
            ax.set_xticklabels(genes, rotation=90)
            ax.axhline(numpy.mean(values), label='Target mean')
            ax.axhline(numpy.mean(values) + numpy.std(values) * 2, color='k', linestyle='--', label='Target cutoff at +2 sigma')
            ax.axhline(numpy.mean(values) - numpy.std(values) * 2, color='k', linestyle='--', label='Target cutoff at -2 sigma')
            ax.axhline(neg_mean, color='g', label='Negative control mean')
            ax.axhline(neg_mean + neg_std * 2, color='g', linestyle='--', label='Negative control +2 sigma')
            ax.axhline(neg_mean - neg_std * 2, color='g', linestyle='--', label='Negative control -2 sigma')
            #ax.axhline(pos_mean, color='r', label='Positive control mean')
            
            pylab.legend(loc=2)
            pylab.ylabel('Outlyingness (%s)' % feature_name)
            pylab.xlabel('Target genes')
            pylab.title('%s' % plate_name)
            pylab.ylim(overall_min-0.1, overall_max+0.1)
            pylab.xlim(-1, n+1)
            pylab.tight_layout()
            
            pylab.savefig(self.output('%s_hit_list_%s.pdf' % (plate_name, feature_name)))
        #pylab.show()
    
    
    def make_top_hit_list(self, top=150, for_group=('neg', 'pos', 'target')):
        if DEBUG:
            print 'Make hit list plot'
        min_object_coutn_group = 8
        group_on = ['Plate', 'siRNA ID', 'Gene Symbol']

        # get global values of all plates        
        group = self.mapping[(self.mapping['Object count'] > 0) ].groupby(group_on)
        overall_max = group['Outlyingness'].max().max()
        
        neg_group = self.mapping[(self.mapping['Object count'] > 0) & (self.mapping['Group'] == 'neg')].groupby(group_on)
        neg_mean = neg_group.mean()['Outlyingness'].mean()
        
        pos_group = self.mapping[(self.mapping['Object count'] > 0) & (self.mapping['Group'] == 'pos')].groupby(group_on)
        pos_mean = pos_group.mean()['Outlyingness'].mean()        
        
        #iterate over plates and make hit list figure
        label = []
        values = []
        take_it = []

        group = self.mapping[(self.mapping['Object count'] > 0)].groupby(['Plate', 'Well', 'siRNA ID', 'Gene Symbol', 'Group'])            
        means = group.apply(lambda x: (x['Outlyingness'] * x['Object count']).sum() / x['Object count'].sum())
        

        for g, m in means.iteritems():
            count = self.mapping['Object count'][group.groups[g]].sum()
            values.append(m)
            label.append(g + (count,))
            take_it.append(count > min_object_coutn_group and self.mapping['Group'][group.groups[g]].iloc[0] in for_group)
                
                    
        svalues, slabel, stake_it = zip(*sorted(zip(values, label, take_it)))
        

        svalues = svalues[-top:]
        slabel = slabel[-top:]
        stake_it = stake_it[-top:]
        
        
          
        if False:
            svalues_val = [ss for ss, t in zip(svalues, stake_it) if t ]
            slabel_val = [ss for ss, t in zip(slabel, stake_it) if t ]
            # Do plot      
            fig = pylab.figure(figsize=(top/6 + 3, 10))
            ax = pylab.subplot(111)
            ax.errorbar(range(len(svalues_val)), svalues_val, fmt='o', markeredgecolor='none')
            ax.set_xticks(range(len(svalues_val)), minor=False)

            ax.set_xticklabels(["%s %s %s %s %s (%d)" % g for g in slabel_val], rotation=90)
            for i, tl in enumerate(ax.get_xticklabels()):
                if slabel_val[i][-2].startswith('pos'):
                    tl.set_color("red") 
                elif slabel_val[i][-2].startswith('neg'):
                    tl.set_color("green")  
            
            ax.axhline(numpy.mean(values), label='Target mean')
            ax.axhline(numpy.mean(values) + numpy.std(values) * 2, color='k', linestyle='--', label='Target cutoff at +2 sigma')
            ax.axhline(numpy.mean(values) - numpy.std(values) * 2, color='k', linestyle='--', label='Target cutoff at -2 sigma')
            ax.axhline(neg_mean, color='g', label='Negative control mean')
            #ax.axhline(pos_mean, color='r', label='Positive control mean')
            
            pylab.legend(loc=2)
            pylab.ylabel('Outlyingness (OC-SVM)')
            pylab.xlabel('Target genes')
            pylab.ylim(0, overall_max+0.1)
            pylab.xlim(-1, len(svalues_val)+1)
            pylab.tight_layout()
            pylab.savefig(self.output('top_%d_hit_list.pdf' % top))
            #pylab.show()
            
        prefix_lut = {}
        for ii, g in enumerate(reversed(slabel)):
            prefix_lut[(g[0], g[1])] = "%04d" % (ii+1)
            
            
        # Export to file
        with open(self.output('top_outlier_list.txt'), 'wb') as fw:
            fw.write("\t".join(['Plate', 'Well', 'siRNA ID', 'Gene Symbol', 'Group', 'CellCount', 'Outlyingness']) + "\n")
            for info, value in zip(slabel, svalues):
                tmp = "\t".join(map(str,info)) + "\t" + str(value) + "\n"
                fw.write(tmp)
        
        #self.make_outlier_galleries(prefix_lut, include_excluded=False)
            

    def make_hit_list(self):
        if DEBUG:
            print 'Make hit list plot'
        min_object_coutn_group = 15
        group_on = ['Plate', 'siRNA ID', 'Gene Symbol']

        # get global values of all plates        
        group = self.mapping[(self.mapping['Object count'] > 0) ].groupby(group_on)
        overall_max = group['Outlyingness'].max().max()
        
        neg_group = self.mapping[(self.mapping['Object count'] > 0) & (self.mapping['Group'] == 'neg')].groupby(group_on)
        neg_mean = neg_group.mean()['Outlyingness'].mean()
        
        pos_group = self.mapping[(self.mapping['Object count'] > 0) & (self.mapping['Group'] == 'pos')].groupby(group_on)
        pos_mean = pos_group.mean()['Outlyingness'].mean()        
        
        #iterate over plates and make hit list figure
        
        for plate_name in self.mapping_files.keys():
            group = self.mapping[(self.mapping['Object count'] > 0) & (self.mapping['Plate'] == plate_name)].groupby(['Well', 'siRNA ID', 'Gene Symbol'])
            
            means = group.apply(lambda x: (x['Outlyingness'] * x['Object count']).sum() / x['Object count'].sum())
            means.sort()
            
            genes = []
            stds = []
            values = []
            for g, m in means.iteritems():
                count = self.mapping['Object count'][group.groups[g]].sum()
                if count > min_object_coutn_group:
                    stds.append(0)
                    values.append(m)
                    genes.append("%s %s %s (%d)" % (g + (count,)))
                
            fig = pylab.figure(figsize=(len(genes)/6 + 3, 10))
            ax = pylab.subplot(111)
            ax.errorbar(range(len(values)), values, yerr=stds, fmt='o', markeredgecolor='none')
            ax.set_xticks(range(len(values)), minor=False)
            ax.set_xticklabels(genes, rotation=90)
            ax.axhline(numpy.mean(values), label='Target mean')
            ax.axhline(numpy.mean(values) + numpy.std(values) * 2, color='k', linestyle='--', label='Target cutoff at +2 sigma')
            ax.axhline(numpy.mean(values) - numpy.std(values) * 2, color='k', linestyle='--', label='Target cutoff at -2 sigma')
            ax.axhline(neg_mean, color='g', label='Negative control mean')
            #ax.axhline(pos_mean, color='r', label='Positive control mean')
            
            pylab.legend(loc=2)
            pylab.ylabel('Outlyingness (OC-SVM)')
            pylab.xlabel('Target genes')
            pylab.title('%s' % plate_name)
            pylab.ylim(0, overall_max+0.1)
            pylab.xlim(-1, len(values)+1)
            pylab.tight_layout()
            
            pylab.savefig(self.output('%s_hit_list.pdf' % plate_name))
        #pylab.show()
        
        
    def interactive_plot(self, shape=(7, 25)):
        sample_names = []
        data_features = []
        data_pca = []
        cellh5_list = []
        wells = []
        sites = []
        predictions = []
        classifications = []
        all_clustering = []
        
        
        for row_index, row in self.mapping[self.mapping['Object count'] > 0].iterrows():
            
            features = self.mapping['Object features'].iloc[row_index]

            plate = row['Plate']
            well = row['Well']
            site = row['Site']
            sirna = row['siRNA ID']
            group = row['Group']
            gene = row['Gene Symbol']
            
            c5f = self.cellh5_handles[plate]
            
            pca = self.mapping['PCA'].iloc[row_index][:,:20]
            cellh5idx = numpy.array(row['CellH5 object index'])
            prediction = numpy.array(row['Predictions'])
            clustering = numpy.array(row['Outlier clustering'])
            all_clustering.append(clustering)
            c5p = c5f.get_position(well, str(site))
            class_prediction = c5p.get_class_prediction()['label_idx'][cellh5idx]
            classifications.append(class_prediction)
            
            
            
            predictions.append(prediction)
            data_features.append(features)
            data_pca.append(pca)
            for _ in range(features.shape[0]):
                sample_names.append((plate, well, str(site), sirna, gene, group))
            cellh5_list.append(cellh5idx)
            
            
            
        predictions = numpy.concatenate(predictions)
        predictions *= -1
        data_pca = numpy.concatenate(data_pca)
        data_features = numpy.concatenate(data_features)
        data_cellh5 = numpy.concatenate(cellh5_list)
        classifications = numpy.concatenate(classifications)
        all_clustering = numpy.concatenate(all_clustering)
  
        cf = self.cellh5_handles.values()[0]
        feature_names = cf.object_feature_def()
        feature_names = [feature_names[ff] for ff in self._non_nan_feature_idx]



        pca_names = ['PCA %d' % d for d in range(data_pca.shape[1])]
        
        app = iscatter.start_qt_event_loop()
        
        def img_gen(treat, ch5_ids):
            gen = []
            max_count = shape[0] * shape[1]
            for i, (treat, c) in enumerate(zip(treat, ch5_ids)):
                plate = treat[0]
                well = treat[1]
                site = str(treat[2])
                cf = self.cellh5_handles[plate]
                img_gen = cf.gallery_image_matrix_gen(((well, site, (c,)),))
                gen.append(img_gen)
                if i > max_count:
                    break
                
            img_gens = chain.from_iterable(gen)
            
            img = CH5File.gallery_image_matrix_layouter(img_gens, shape)
            return img
        
        features = numpy.c_[data_pca, data_features[:,self._non_nan_feature_idx], all_clustering, predictions, classifications]
    
        names = pca_names + feature_names  + ['Clustering', 'Outliers', 'Classification']
  
        def contour_eval(xlim, ylim, xdim, ydim):
            xx, yy = numpy.meshgrid(numpy.linspace(xlim[0], xlim[1], 100), numpy.linspace(ylim[0], ylim[1], 100))
            # Z = self.classifier.decision_function(numpy.c_[xx.ravel(), yy.ravel()])
            matrix = numpy.zeros((100 * 100, self.pca_dims))
            matrix[:, xdim] = xx.ravel()
            matrix[:, ydim] = yy.ravel()
            Z = self.classifier.decision_function(matrix)
            return xx, yy, Z.reshape(xx.shape)
        
       
        iscatter_widget = iscatter.IScatterWidgetHisto()
        #iscatter_widget.set_countour_eval_cb(contour_eval)
        iscatter_widget.set_data(features, names, sample_names, 0, 1, data_cellh5, img_gen)
        
        iscatter_widget2 = iscatter.IScatterWidgetHisto()
        #iscatter_widget2.set_countour_eval_cb(contour_eval)
        iscatter_widget2.set_data(features, names, sample_names, 0, 1, data_cellh5, img_gen)
    
        mw = iscatter.IScatter(iscatter_widget, iscatter_widget2)
        
        
        
        
        mw.show()
        app.exec_()
        
        
    def make_pca_scatter(self):    
        KK = min(5, self.pca_dims)
            
        pcs = [(x, y) for x in range(KK) for y in range(KK)]
        legend_there = False
        for ii, (f_x, f_y) in enumerate(pcs):
            if f_x >= f_y:
                continue
            
            fig = pylab.figure(figsize=(20, 15))
            legend_points = []
            legend_labels = []

            treatment_group = self.mapping.groupby(['Gene Symbol','siRNA ID'])
            
            x_min, y_min = 1000000, 100000
            x_max, y_max = -100000, -100000

            for tg in treatment_group:
                treatment = "%s %s" % tg[0]
                wells = list(tg[1]['Well'])
                pca_components = numpy.concatenate(list(tg[1]['PCA']))  
                prediction = numpy.concatenate(list(tg[1]['Predictions']))  
        
                x_min_cur, x_max_cur = pca_components[:, f_x].min(), pca_components[:, f_x].max()
                y_min_cur, y_max_cur = pca_components[:, f_y].min(), pca_components[:, f_y].max()
                
                x_min = min(x_min, x_min_cur)
                y_min = min(y_min, y_min_cur)
                x_max = max(x_max, x_max_cur)
                y_max = max(y_max, y_max_cur)

                ax = pylab.subplot(1, 2, 1)
                ax.set_title("Outlier detection PCA=%d vs %d, nu=%f, g=%f (%s)" %(f_x+1, f_y+1, self.nu, self.gamma, self.classifier.kernel))
                    
                if "Taxol" in treatment:
                    color = 'blue'
                    if "No Reversine" in treatment:
                        color = "cyan"
                elif "Noco" in treatment:
                    color = "red"
                    if "No Reversine" in treatment:
                        color = "orange"
                else:
                    assert 'wt control' in treatment
                    color = "green"
                        
                #if color=='green':
                points = ax.scatter(pca_components[prediction == 1, f_x], pca_components[prediction == 1, f_y], c=color, marker="o", facecolors=color, zorder=999, edgecolor="none", s=20)
                legend_points.append(points)
                legend_labels.append("Inlier " + treatment)
                
                points = ax.scatter(pca_components[prediction == -1, f_x], pca_components[prediction == -1, f_y], c=color, marker="o", facecolors='none', zorder=999, edgecolor=color, s=20)
                legend_points.append(points)
                legend_labels.append("Outlier " + treatment)    
                
                ax = pylab.subplot(1, 2, 2)
                cluster_vectors = numpy.concatenate(list(tg[1]['Outlier clustering']))
                ax.set_title("Outlier clustering (%d)" % cluster_vectors.max())
                cluster_colors = {0:'k', 1:'r', 2:'g', 3:'b', 4:'y', 5:'m'}
                for k in range(1, cluster_vectors.max()+1):
                    points = ax.scatter(pca_components[cluster_vectors == k, f_x], pca_components[cluster_vectors == k, f_y], c=cluster_colors[k], marker="o", facecolors=cluster_colors[k], zorder=999, edgecolor="none", s=20)
                        

            
            ax = pylab.subplot(1, 2, 1)    
            xx, yy = numpy.meshgrid(numpy.linspace(x_min, x_max, 100), numpy.linspace(y_min, y_max, 100))
            # Z = self.classifier.decision_function(numpy.c_[xx.ravel(), yy.ravel()])
            matrix = numpy.zeros((100 * 100, self.pca_dims))
            matrix[:, f_x] = xx.ravel()
            matrix[:, f_y] = yy.ravel()
            
            Z = self.classifier.decision_function(matrix)
            Z = Z.reshape(xx.shape)
            
            ax.contourf(xx, yy, Z, levels=numpy.linspace(Z.min(), 0, 17), cmap=pylab.matplotlib.cm.Reds_r, alpha=0.2)
            ax.contour(xx, yy, Z, levels=[0], linewidths=1, colors='k')
            ax.contourf(xx, yy, Z, levels=numpy.linspace(0, Z.max(), 17), cmap=pylab.matplotlib.cm.Greens, alpha=0.3)
            
                    
            if not legend_there:
                pylab.figlegend(legend_points, legend_labels, loc = 'lower center', ncol=4, labelspacing=0.1 )
                lengend_there = True
            
            ax = pylab.subplot(1, 2, 1)
            ax.set_xlim((x_min, x_max))
            ax.set_ylim((y_min, y_max))    
            pylab.xticks([])
            pylab.yticks([])
            
            ax = pylab.subplot(1, 2, 2)
            ax.set_xlim((x_min, x_max))
            ax.set_ylim((y_min, y_max))    
            pylab.xticks([])
            pylab.yticks([])
            
            
            pylab.subplots_adjust(wspace=0.05, hspace=0.05)
            pylab.tight_layout()
            pylab.savefig(self.output("outlier_detection_pca_%d_vs_%d.pdf" %(f_x+1, f_y+1)))   
            
            
    def make_outlier_galleries_per_pos_per_class(self, prefix_lut=None):  
        group = self.mapping[(self.mapping['Object count'] > 0)].groupby(('Plate', 'Well'))

        for grp, grp_values in group:
            if prefix_lut is None:
                pass
            elif not  (grp_values['Plate'].unique()[0],  grp_values['Well'].unique()[0]) in prefix_lut:
                continue

            for _, (plate_name, well, site, ge, si, prediction, hyper, ch5_ind) in grp_values[['Plate', 'Well', 'Site', 'Gene Symbol', 'siRNA ID', 'Predictions', "Hyperplane distance", "CellH5 object index"]].iterrows():
                in_ch5_index = ch5_ind[prediction == 1]
                in_dist =  hyper[prediction == 1]
                in_sorted_ch5_index = zip(*sorted(zip(in_dist, in_ch5_index), reverse=True))
                if len(in_sorted_ch5_index) > 1:
                    in_sorted_ch5_index = in_sorted_ch5_index[1]
                else:
                    in_sorted_ch5_index = []
                    
                out_ch5_index = ch5_ind[prediction == -1]
                out_dist =  hyper[prediction == -1]
                out_sorted_ch5_index = zip(*sorted(zip(out_dist, out_ch5_index), reverse=True))
                if len(out_sorted_ch5_index) > 1:
                    out_sorted_ch5_index = out_sorted_ch5_index[1]
                else:
                    out_sorted_ch5_index = []
                
                  
                cf = self.cellh5_handles[plate_name]
                class_def = cf.class_definition('primary__primary')["label"]

                cp = cf.get_position(well, str(site))  
                
                class_pred = cp.get_class_prediction()['label_idx']
                
                inlier_img = []
                outlier_img = []
                for l in range(len(class_def)):
                    this_class = numpy.nonzero(class_pred == l)[0]
                    tmp_idx = []
                    for in_idx in in_sorted_ch5_index:
                        if in_idx in this_class:
                            tmp_idx.append(in_idx)
                    inlier_img.append(cp.get_gallery_image_matrix(tmp_idx, (8, 32)).swapaxes(1,0))
                    inlier_img.append(numpy.ones((inlier_img[0].shape[0],3))*255)
                                        
                    tmp_idx = []
                    for out_idx in out_sorted_ch5_index:
                        if out_idx in this_class:
                            tmp_idx.append(out_idx)
                    outlier_img.append(cp.get_gallery_image_matrix(tmp_idx, (8, 32)).swapaxes(1,0))
                    outlier_img.append(numpy.ones((outlier_img[0].shape[0],3))*255)
                
                inlier_img = numpy.concatenate(inlier_img, 1)
                outlier_img = numpy.concatenate(outlier_img, 1)
                
                img = numpy.concatenate((inlier_img, numpy.ones((3, inlier_img.shape[1]))*255, outlier_img))
                   
                if prefix_lut is None:
                    img_name = 'xcgal_%s_%s_%s_.png' % (plate_name, str(well) , str(site))
                else:
                    img_name = '%s_%s_%s_%s_%s.png' % ((prefix_lut[well, site],) + (plate_name, str(well) , str(site)))
                vigra.impex.writeImage(img, self.output(img_name))
                if DEBUG:
                    print 'Exporting gallery matrix image for', info 
        
    def make_outlier_galleries(self, prefix_lut=None, include_excluded=True):  
        group = self.mapping[(self.mapping['Object count'] > 0)].groupby(('Plate', 'Well'))
        plates_exludes = defaultdict(list)
        plates_inlier = defaultdict(list)
        plates_outlier = defaultdict(list)
        plates_info  = defaultdict(list)
         
        for grp, grp_values in group:
            index_tpl_in = []
            index_tpl_out = []
            index_tpl_exc = []
            
            if prefix_lut is None:
                pass
            elif not  (grp_values['Plate'].unique()[0],  grp_values['Well'].unique()[0]) in prefix_lut:
                continue

            for _, (plate_name, well, site, ge, si, prediction, hyper, ch5_ind, ch5_ind_excl) in grp_values[['Plate', 'Well', 'Site', 'Gene Symbol', 'siRNA ID', 'Predictions', "Hyperplane distance", "CellH5 object index", "CellH5 object index excluded"]].iterrows():
                in_ch5_index = ch5_ind[prediction == 1]
                in_dist =  hyper[prediction == 1]
                in_sorted_ch5_index = zip(*sorted(zip(in_dist, in_ch5_index), reverse=True))
                if len(in_sorted_ch5_index) > 1:
                    in_sorted_ch5_index = in_sorted_ch5_index[1]
                else:
                    in_sorted_ch5_index = []
                    
                out_ch5_index = ch5_ind[prediction == -1]
                out_dist =  hyper[prediction == -1]
                out_sorted_ch5_index = zip(*sorted(zip(out_dist, out_ch5_index), reverse=True))
                if len(out_sorted_ch5_index) > 1:
                    out_sorted_ch5_index = out_sorted_ch5_index[1]
                else:
                    out_sorted_ch5_index = []
                    
                index_tpl_in.append((well, site, in_sorted_ch5_index))
                index_tpl_out.append((well, site, out_sorted_ch5_index))
                index_tpl_exc.append((well, site, ch5_ind_excl))
                    
            plates_inlier[plate_name].append(index_tpl_in)
            plates_outlier[plate_name].append(index_tpl_out)
            plates_exludes[plate_name].append(index_tpl_exc)
            plates_info[plate_name].append((plate_name, well, ge, si))
            
           
        shape = (16,8)
        for plate_name in plates_inlier.keys():
            pl_in_index_tpl = plates_inlier[plate_name]
            pl_on_index_tpl = plates_outlier[plate_name]
            pl_ex_index_tpl = plates_exludes[plate_name]
            pl_info = plates_info[plate_name]
            
            for index_tpl_in, index_tpl_out, index_tpl_ex, info in zip(pl_in_index_tpl, pl_on_index_tpl, pl_ex_index_tpl, pl_info): 
                cf = self.cellh5_handles[plate_name]
                for object_ in ['primary__primary', 'secondary__inside']:
                    if include_excluded:
                        excluded_img = cf.get_gallery_image_matrix(index_tpl_ex, shape, object_=object_).swapaxes(1,0) 
                    outlier_img = cf.get_gallery_image_matrix(index_tpl_out, shape, object_=object_).swapaxes(1,0)
                    inlier_img = cf.get_gallery_image_matrix(index_tpl_in, shape, object_=object_).swapaxes(1,0)
                    
                    if include_excluded:
                        img = numpy.concatenate((excluded_img, numpy.ones((5, inlier_img.shape[1]))*255, inlier_img, numpy.ones((5, inlier_img.shape[1]))*255, outlier_img))
                    else:
                        img = numpy.concatenate((inlier_img, numpy.ones((5, inlier_img.shape[1]))*255, outlier_img))
                    assert plate_name == info[0]
                    if prefix_lut is None:
                        img_name = 'xgal_%s_%s_%s_%s_%s.png' % info  + (object_,)
                    else:
                        img_name = '%s_%s_%s_%s_%s_%s.png' % ((prefix_lut[info[0], info[1]],) + info + (object_,))
                    vigra.impex.writeImage(img, self.output(img_name))
                    if DEBUG:
                        print 'Exporting gallery matrix image for', info, object_

                

        

      

    
        

