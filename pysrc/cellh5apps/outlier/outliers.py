import matplotlib
import numpy
import vigra
import scipy
import h5py
import time

from matplotlib import pyplot as plt
from cellh5apps.utils.plots import matplotlib_black_background

from scipy import stats
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
from cellh5 import CH5Analysis, CH5File
import pandas

import qimage2ndarray
         
         
from cellh5apps.utils.colormaps import QtColorMapFromHex

from itertools import product

from cellh5apps.outlier.learner import OneClassSVM_LIBSVM, OneClassKDE, OneClassMahalanobis, OneClassGMM, OneClassSVM_SKL, ClusterGMM
from _functools import partial
from matplotlib.mlab import dist
YlBlCMap = matplotlib.colors.LinearSegmentedColormap.from_list('asdf', [(0,0,1), (1,1,0)])


from PyQt4 import QtGui, QtCore
def blend_images_max(images):
    """
    blend a list of QImages together by "lighten" composition (lighter color
    of source and dest image is selected; same effect as max operation)
    """
    assert len(images) > 0, 'At least one image required for blending.'
    pixmap = QtGui.QPixmap(images[0].width(), images[0].height())
    # for some reason the pixmap is NOT empty
    pixmap.fill(QtCore.Qt.black)
    painter = QtGui.QPainter(pixmap)
    painter.setCompositionMode(QtGui.QPainter.CompositionMode_Lighten)
    for image in images:
        if not image is None:
            painter.drawImage(0, 0, image)
    painter.end()
    return pixmap

def _predict_od_(xxx, classifier, feature_set):
    data_ = xxx[feature_set]
    if len(data_) == 0 or isinstance(data_, (float,)):
        return numpy.zeros((0, 0)), numpy.zeros((0, 0))
    prediction = classifier.predict(data_)
    distance = classifier.decision_function(data_)
    return prediction, distance 

class OutlierDetection(CH5Analysis):
    def __init__(self, name, mapping_files, cellh5_files, max_training_samples=3000, training_sites=None, rows=None, cols=None, locations=None, gamma=None, nu=None, pca_dims=None, kernel=None):
        CH5Analysis.__init__(self, name, mapping_files, cellh5_files, sites=training_sites, rows=rows, cols=cols, locations=locations)
        self.set_max_training_sample_size(max_training_samples)
        
    def train(self, train_on=('neg',), classifier_class=OneClassSVM_SKL, in_classes=None, feature_set="Object features", **kwargs):
        training_matrix = self.get_data(train_on, feature_set, in_classes)
        self.train_classifier(training_matrix, classifier_class, **kwargs)
        
    def predict2(self, feature_set="Object features"):
        
        
        prediction, distance = cellh5.pandas_ms_apply(self.mapping, partial(_predict_od_, classifier=self.classifier, feature_set=feature_set), 16)
        self.mapping['Predictions2'] = pandas.Series(prediction)
        self.mapping['Score2'] = distance
        
        

    def predict(self, test_on=('target', 'pos', 'neg'), feature_set="Object features"):
        testing_matrix_list = self.mapping[self.mapping['Group'].isin(test_on)][['Well', 'Site', feature_set, "Gene Symbol", "siRNA ID"]].iterrows()

        predictions = {}
        distances = {}
        self.log.info('Classification prediction')
        with open(self.output('_outlier_detection_prediciton_log.txt'), 'w') as log_file_handle:
            for idx, (well, site, tm, t1, t2) in  testing_matrix_list:
                log_file_handle.write("%s\t%d\t%s\t%s" % (well, site, t1, t2))
                if isinstance(tm, (float,)) or (tm.shape[0] == 0):
                    predictions[idx] = numpy.zeros((0, 0))
                    distances[idx] = numpy.zeros((0, 0))
                    log_file_handle.write("\t0 / 0 outliers\t0.00\n")
                else:
                    pred, dist = self.predict_with_classifier(tm, log_file_handle)
                    predictions[idx] = pred
                    distances[idx] = dist

        self.mapping['Predictions'] = pandas.Series(predictions)
        self.mapping['Score'] = pandas.Series(distances)
        
    def train_classifier(self, training_matrix, classifier_class, **kwargs):
        if classifier_class in (OneClassSVM, OneClassSVM_LIBSVM):
            self.classifier = classifier_class(kernel=kwargs["kernel"], nu=kwargs["nu"], gamma=kwargs["gamma"])
        elif classifier_class == OneClassKDE:
            self.classifier = classifier_class(**kwargs)
        elif classifier_class == OneClassMahalanobis:
            self.classifier = classifier_class(**kwargs)
        elif  classifier_class == OneClassGMM:
            self.classifier = classifier_class(**kwargs)
        elif   classifier_class == OneClassSVM_SKL:
            self.classifier = classifier_class(**kwargs)
        else:
            raise RuntimeError("Classifier unknown")
            
        if training_matrix.shape[0] > self.max_training_samples:
            idx = range(training_matrix.shape[0])
            numpy.random.seed(43)
            numpy.random.shuffle(idx)
            idx = idx[:self.max_training_samples]
            training_matrix = training_matrix[idx, :]
            
        self.last_training_matrix = training_matrix
        self.log.info('Classification training with matrix of shape %s' % str(training_matrix.shape))
        self.classifier.fit(training_matrix)
        
    def set_max_training_sample_size(self, max_training_samples):
        self.log.info("Set maximum training samples to %d" % max_training_samples)
        self.max_training_samples = max_training_samples

            
    def compute_outlyingness(self):
        def _outlier_count(x):
            res = numpy.float32((x == -1).sum()) / len(x)
            return res
            
        res = pandas.Series(self.mapping[self.mapping['Group'].isin(('target', 'pos', 'neg'))]['Predictions'].map(_outlier_count))
        self.mapping['Outlyingness'] = res
        
    def predict_with_classifier(self, test_matrix, log_file_handle=None):
        prediction = self.classifier.predict(test_matrix)
        distance = self.classifier.decision_function(test_matrix)
        log = "\t%d / %d outliers\t%3.2f" % ((prediction == -1).sum(),
                                             len(prediction),
                                             (prediction == -1).sum() / float(len(prediction)))

        if log_file_handle is not None:
            log_file_handle.write(log + "\n")
        return prediction, distance
    
    
class OutlierDetectionAssay(object):
    def __init__(self, ch5analysis):
        self.ca = ch5analysis
        self.mapping = ch5analysis.mapping
        
    def normalize(self, matrix):
        row_sums = matrix.sum(axis=1)
        return matrix / row_sums[:, numpy.newaxis]
    
class OutlierGalleryImages(object):
    def plot_ouutlier_galleries(self):
        pass
    
    def plot_cluster_galleries(self):
        pass
        
class OutlierClusterPlots(OutlierDetectionAssay):

    def cluster(self, cluster_class, feature_names=None, **kwargs):
        feature_set = "Object features"
        if feature_names is None:
            self.ca.pca()
            feature_set = "PCA"
        else:
            feature_idx = [self.ca.all_features.index(name) for name in feature_names]
            # feature_idx.sort()
            
        self.cluster_feature_idx = feature_idx
            
        
        outlier_prediction = self.ca.get_column_as_matrix("Predictions")
        data = self.ca.get_column_as_matrix(feature_set)
        training_data = data[outlier_prediction==-1, :]
        training_data = training_data[:, self.cluster_feature_idx]
        
        self.cluster_class = cluster_class(**kwargs)
        self.cluster_class.fit(training_data)
        self.k = self.cluster_class.n_components
        
        def _cluster_(xxx):
            data = xxx[feature_set][:, self.cluster_feature_idx]
            cluster = self.cluster_class.predict(data)
            outliers = xxx["Predictions"]
            outliers_score = xxx["Score"]
            distance = numpy.zeros(cluster.shape)
            for kk in range(self.k):
                tmp = numpy.linalg.norm(data[cluster==kk, :] - self.cluster_class.means_[kk, :], axis=1)
                assert distance[cluster==kk].shape == tmp.shape
                distance[cluster==kk] = tmp
            cluster +=1
            cluster[outliers==1] = 0
            distance[outliers==1] = -outliers_score[outliers==1]
            return cluster, distance 
        
        cluster, distance = cellh5.pandas_apply(self.mapping, _cluster_)
        self.mapping['Outlier clustering'] = pandas.Series(cluster)
        self.mapping['Outlier clustering distance'] = pandas.Series(distance)
        
    def export_cluster_representatives(self, img_shape=(8,4), feature_names=None):
        cluster, index = self.ca.get_column_as_matrix('Outlier clustering', True)
        distance = self.ca.get_column_as_matrix('Outlier clustering distance')
        ch5_idx =  self.ca.get_column_as_matrix('CellH5 object index 2')
        
        for kk in range(0, self.k+1):
            cluster_k = [cluster == kk]
            distance_k = distance[cluster_k]
            index_k = index[cluster_k]
            ch5_idx_k = ch5_idx[cluster_k]
            
            sort_idx = numpy.argsort(distance_k)
            
             
            
            first_indek_k = index_k[sort_idx[:numpy.prod(img_shape)]]
            first_ch5_k = ch5_idx_k[sort_idx[:numpy.prod(img_shape)]]
            
            img_k = self.get_galleries(first_indek_k, first_ch5_k)
            img_k = CH5File.gallery_image_matrix_layouter_rgb(iter(img_k), img_shape)
            
            file_name = self.ca.output("cluster_centers_cluster_%d.png" % kk)
            vigra.impex.writeImage(img_k.swapaxes(1,0), file_name)
            
        
        if feature_names is not None:
            means_df = pandas.DataFrame(self.cluster_class.means_, columns=feature_names)
        else:
            means_df = pandas.DataFrame(self.cluster_class.means_)
            
        html_str = means_df.to_html()
        fn = self.ca.output("cluster_means.html")
        with open(fn, "wb") as f:
            f.write(html_str)

            
    def get_galleries(self, row_index, ch5_index):
        img = []
        for ri, ci in zip(row_index, ch5_index):
            plate = self.mapping.loc[ri]["Plate"]
            well = self.mapping.loc[ri]["Well"]
            site = self.mapping.loc[ri]["Site"]
            
            ch5pos = self.ca.cellh5_handles[plate].get_position(well, str(site))
            img.append(ch5pos.get_gallery_image_contour(ci))
        return img
        
    def evaluate(self, split):
        cluster = self.ca.get_column_as_matrix("Outlier clustering")
        classi = self.ca.get_column_as_matrix("Object classification label")
        
        cm = confusion_matrix(classi, cluster).astype(numpy.float32)[:,:(self.k+1)]
        self.export_confusion(cm)
        self.export_result_table()
        
        
    def export_result_table(self):
        export = self.ca.mapping[["Plate", "Well", "Site", "Gene Symbol", "siRNA ID", "Group", "Object count", "Outlyingness"]]
        filename = self.ca.output("cluster_result_table_%s.txt" % self.cluster_class.describe())
        
        def _class_count(x, cls):
            res = numpy.float32((x == cls).sum()) / len(x)
            return res

        cluster = pandas.Series(self.mapping[self.mapping['Object count'] > 0]['Outlier clustering'])
        class_dict = dict([(j, "Cluster") for j in range(self.k+1)])
        for class_i, name in class_dict.items():                     
            export['%02d_%s' % (class_i, name)] = cluster.map(partial(_class_count, cls=class_i))

        export.to_csv(filename, sep="\t")
        
    def export_confusion(self, cm):
        fig = plt.figure()
        
        ax = plt.subplot(121)
        res = ax.pcolor(cm, cmap=YlBlCMap,)
        for i, cas in enumerate(cm):
            for j, c in enumerate(cas):
                if c>0:
                    ax.text(j+0.2, i+0.5, "%d"%c, fontsize=10)
        ax.set_ylim(0,cm.shape[0])
        ax.invert_yaxis()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        cm2 = self.normalize(cm)
          
        ax = plt.subplot(122)
        res = ax.pcolor(cm2, cmap=YlBlCMap, vmin=0, vmax=1)
        for i, cas in enumerate(cm2):
            for j, c in enumerate(cas):
                if c>0:
                    ax.text(j+0.2, i+0.5, "%3.2f" % c, fontsize=10)
        ax.set_ylim(0, cm2.shape[0])
        ax.invert_yaxis()         
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        filename = self.ca.output("cluster_confusion_%s.png" % self.cluster_class.describe())
        fig.savefig(filename)
        plt.close(fig)
        
    def export_galleries_per_pos(self):
        for i, row in self.ca.mapping.iterrows():
            oc = row["Object count"]
            if oc > 0:
                plate = row["Plate"]
                well  = row["Well"]
                site  = row["Site"]
                gene  = row["Gene Symbol"]
                
                print plate, well, site
                
                ch5pos = self.ca.cellh5_handles[plate].get_position(well, str(site))
                
                cluster = row["Outlier clustering"]
                ch5_index_bool = row["CellH5 object index"]
                
                ch5_index = numpy.nonzero(ch5_index_bool)[0]
                
                image = []
                for kk in range(self.k+1):
                    img = ch5pos.get_gallery_image_matrix(ch5_index[cluster==kk], (20,5))
                    image.append(img)
                    sep = numpy.ones((img.shape[0],3,3))*255
                    image.append(sep)
                
                vigra.impex.writeImage(numpy.concatenate(image[:-1],1).swapaxes(1,0), self.ca.output("cluster_gal_%s_%s_%s_%s.png"%(plate, well, str(site), gene)))
                    
                
                print cluster.shape
                print ch5_index.shape
                
                
                 
        
        
        
        

class OutlierDetectionSingleCellPlots(OutlierDetectionAssay):
    def grid_search(self, func, **kwargs):
        for grid_params in product(*kwargs.values()):
            cur_args = {}
            for i, k in enumerate(kwargs.keys()):
                cur_args[k] = grid_params[i] 
                
            func(**cur_args)
    
    def get_outlier_confusion(self, on_group=('neg', 'target', 'pos'), compare_to='Predictions', outlier_indicator=-1):
        group_selection = self.mapping['Group'].isin(on_group)
        sl = numpy.concatenate(list(self.mapping[self.mapping['Object count'] >0 & group_selection]['Object classification label']))
        od = numpy.concatenate(list(self.mapping[self.mapping['Object count'] >0 & group_selection][compare_to]))
        
        cm = confusion_matrix(sl, od==outlier_indicator)[:,:2].astype(numpy.float32)
        return cm
    
    def get_cluster_confusion(self, on_group=('neg', 'target', 'pos'), compare_to='Simple clustering', outlier_indicator=None):
        group_selection = self.mapping['Group'].isin(on_group)
        sl = numpy.concatenate(list(self.mapping[self.mapping['Object count'] >0 & group_selection]['Object classification label']))
        cc = numpy.concatenate(list(self.mapping[self.mapping['Object count'] >0 & group_selection][compare_to]))
        
        if outlier_indicator is None:
            outlier_indicator = numpy.argmin(numpy.bincount(cc))
        
        cm = confusion_matrix(sl, cc==outlier_indicator)[:,:2].astype(numpy.float32)
        return cm
    
    
    
    def get_stats(self, cm, split):
        result = {}
        tp = cm[split:, 1].sum()
        fp = cm[:split, 1].sum()
        tn = cm[:split, 0].sum()
        fn = cm[split:, 0].sum()
        
        result['acc'] = acc = ((tp + tn) / cm.sum())
        result['tpr'] = tpr = (tp / (tp+fn))
        result['fpr'] = fpr = (fp / (fp+tn))
        result['pre'] = pre = (tp / (tp+fp))
        
        result['f1']  = 2*tpr*pre / (tpr+pre)
        result['f2']  = 2*tpr*(1-fpr) / (tpr+(1-fpr))
        result['bacc'] = (tpr + (1-fpr)) / 2
        
        return result
    
    def export_result_table(self):
        export = self.ca.mapping[["Plate", "Well", "Site", "Gene Symbol", "siRNA ID", "Group", "Object count", "Outlyingness"]]
        filename = self.ca.output("result_table_%s.txt" % self.ca.classifier.describe())
        
        def _class_count(x, cls):
            res = numpy.float32((x == cls).sum()) / len(x)
            return res
        
        class_dict = self.ca.get_object_classificaiton_dict()

        class_labels = pandas.Series(self.mapping[self.mapping['Object count'] > 0]['Object classification label'])
        for class_i, name in class_dict.items():                     
            export['%02d_%s' % (class_i, name)] = class_labels.map(partial(_class_count, cls=class_i))

        export.to_csv(filename, sep="\t")
        
    def export_confusion(self, cm, split, prefix=""):
        fig = plt.figure()
        
        ax = plt.subplot(121)
        res = ax.pcolor(cm, cmap=YlBlCMap,)
        for i, cas in enumerate(cm):
            for j, c in enumerate(cas):
                ax.text(j+0.3, i+0.5, "%d"%c, fontsize=10)
        ax.set_ylim(0,cm.shape[0])
        ax.invert_yaxis()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        stats = self.get_stats(cm, split)
        ax.set_title("acc %3.2f, tpr %3.2f, fpr %3.2f \npre %3.2f, f1 %3.2f, bacc %3.2f" % (stats["acc"], stats["tpr"], stats["fpr"], stats["pre"], stats["f1"], stats["bacc"]))
        
        cm2 = self.normalize(cm)
          
          
        ax = plt.subplot(122)
        res = ax.pcolor(cm2, cmap=YlBlCMap, vmin=0, vmax=1)
        for i, cas in enumerate(cm2):
            for j, c in enumerate(cas):
                ax.text(j+0.5, i+0.5, "%3.2f" % c, fontsize=10)
        ax.set_ylim(0, cm2.shape[0])
        ax.invert_yaxis()         
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        stats = self.get_stats(cm2, split)
        ax.set_title("acc %3.2f, tpr %3.2f, fpr %3.2f \npre %3.2f, f1 %3.2f, bacc %3.2f" % (stats["acc"], stats["tpr"], stats["fpr"], stats["pre"], stats["f1"], stats["bacc"]))
        
        filename = self.ca.output("confusion_%s.png" % self.ca.classifier.describe())
        fig.savefig(filename)
        plt.close(fig)
        
        filename = self.ca.output("confusion_%s.txt" % self.ca.classifier.describe())
        numpy.savetxt(filename, cm, delimiter="\t")
        
    
    def evaluate(self, split, return_='bacc'):
#         self.export_result_table()
        cm = self.get_outlier_confusion()
        self.export_confusion(cm, split)
        stats = self.get_stats(cm, split)
        sv_frac = (self.ca.classifier.support_vectors_.shape[0] / float(self.ca.last_training_matrix.shape[0])) 
    
        output=  "%5.4f\t%5.4f\t%5.4f\t%4.3f\t%4.3f\t%4.3f\t%s\t%f\t%f\t%f\n" % (stats['acc'], stats['tpr'], stats['fpr'], stats['pre'], stats['f1'], stats['bacc'], self.ca.classifier.describe(), sv_frac, self.ca.classifier.nu , self.ca.classifier.gamma)
        with open(self.ca.output("report.txt"), "a") as myfile:
            myfile.write(output)
            
        if return_ == "all":
            return stats
        return stats[return_]
    
    def evaluate_cluster(self, split, return_='bacc'):
        cm = self.get_cluster_confusion()
        self.export_confusion(cm, split)
        stats = self.get_stats(cm, split)
        sv_frac = (self.ca.classifier.support_vectors_.shape[0] / float(self.ca.last_training_matrix.shape[0])) 
    
        output=  "%5.4f\t%5.4f\t%5.4f\t%4.3f\t%4.3f\t%4.3f\t%s\t%f\t%f\t%f\n" % (stats['acc'], stats['tpr'], stats['fpr'], stats['pre'], stats['f1'], stats['bacc'], self.ca.classifier.describe(), sv_frac, self.ca.classifier.nu , self.ca.classifier.gamma)
        with open(self.ca.output("report.txt"), "a") as myfile:
            myfile.write(output)
        if return_ == "all":
            return stats
        return stats[return_]
    
    def plot(self, split):
        cm = self.get_outlier_confusion()
        cm_n = self.normalize(cm)
        self.get_stats(cm, split)
        
    def show_feature_space(self, split, for_group, cut_to_percentile=None, bin_size=128):
        matrix = self.ca.get_data(for_group, "PCA")
        xmin, xmax = matrix[:,0].min(), matrix[:,0].max()
        ymin, ymax = matrix[:,1].min(), matrix[:,1].max()
        if cut_to_percentile is not None:
            p_l = lambda x: numpy.percentile(x, cut_to_percentile)
            p_h = lambda x: numpy.percentile(x, 100-cut_to_percentile)
            
            xmin, xmax = p_l(matrix[:,0]), p_h(matrix[:,0])
            ymin, ymax = p_l(matrix[:,1]), p_h(matrix[:,1])
            
        bins = (numpy.linspace(xmin, xmax, bin_size), numpy.linspace(ymin, ymax, bin_size))
        
        def mysavefig(filename):
            ax.set_xlabel("Principal component 1")
            ax.set_ylabel("Principal component 2")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect(1)
            plt.tight_layout()
            plt.savefig(self.ca.output(filename), transparent=True,bbox_inches='tight')
            plt.clf()
        
        with open(self.ca.output("__counts_%s_%s.txt" % ("_".join(for_group), self.ca.classifier.describe())), 'wb') as fh:
            fh.write("%s\t%s\t%s\n" % (self.ca.name, "_".join(for_group), self.ca.classifier.describe()))
            fh.write('\n')
            
            class_dict = self.ca.get_object_classificaiton_dict()
            max_class = max(class_dict.keys())
            
            normal_class = range(split)        
            data_normal = self.ca.get_data(for_group, "PCA", in_classes=tuple(normal_class))
            ax = plt.subplot(111) 
            image1, image1_org  = self.myscatter(ax, data_normal[:, 0], data_normal[:, 1], bins=bins)    
            vigra.impex.writeImage(image1_org.swapaxes(1,0), self.ca.output("%s_%s_%d_%s.tif" % ("_".join(for_group), "normal", image1_org.sum(), self.ca.classifier.describe())))
            
            qimage_normal = qimage2ndarray.gray2qimage(image1, False)
            qimage_normal.setColorTable(QtColorMapFromHex("#00FF00"))
            
            
            abnormal_class = range(split, max_class+1)
            data_abnormal = self.ca.get_data(for_group, "PCA", in_classes=tuple(abnormal_class))
            ax = plt.subplot(111) 
            image1, image1_org  = self.myscatter(ax, data_abnormal[:, 0], data_abnormal[:, 1], bins=bins)    
            vigra.impex.writeImage(image1_org.swapaxes(1,0), self.ca.output("%s_%s_%d_%s.tif" % ("_".join(for_group), "abnormal", image1_org.sum(), self.ca.classifier.describe())))
             
            qimage_abnormal = qimage2ndarray.gray2qimage(image1, False)
            qimage_abnormal.setColorTable(QtColorMapFromHex("#FF0000"))
            
            qpixmap = blend_images_max([qimage_normal, qimage_abnormal])
            qpixmap.save(self.ca.output("__sl_%s_%s.png" % ("_".join(for_group), self.ca.classifier.describe())))
            
            fh.write("Data normal\t%d\t%f\n" % (len(data_normal), len(data_normal)    / float(len(data_normal) + len(data_abnormal)) ))
            fh.write("Data abnormal\t%d\t%f\n" % (len(data_abnormal), len(data_abnormal) / float(len(data_normal) + len(data_abnormal)) ))
            fh.write('\n')
            ###
            inlier_target = self.ca.get_data(for_group, "PCA", in_classes=(1,), in_class_type="Predictions")
            outlier_target = self.ca.get_data(for_group, "PCA", in_classes=(-1,), in_class_type="Predictions")

            ax = plt.subplot(111)    
      
            image_in , orig_image_in  = self.myscatter(ax, inlier_target [:, 0], inlier_target[:, 1], bins=bins)
            image_out, orig_image_out = self.myscatter(ax, outlier_target[:, 0], outlier_target[:, 1], bins=bins) 
            
            fh.write("Inlier\t%d\t%f\n" % (len(inlier_target), len(inlier_target)    / float(len(inlier_target) + len(outlier_target)) ))
            fh.write("Outlier\t%d\t%f\n" % (len(outlier_target), len(outlier_target) / float(len(inlier_target) + len(outlier_target)) ))
            fh.write('\n')
            
            qimage_inlier = qimage2ndarray.gray2qimage(image_in, False)
            qimage_inlier.setColorTable(QtColorMapFromHex("#00FF00"))
            qimage_outlier = qimage2ndarray.gray2qimage(image_out, False)
            qimage_outlier.setColorTable(QtColorMapFromHex("#FF0000"))
     
            qpixmap = blend_images_max([qimage_inlier, qimage_outlier])
            qpixmap.save(self.ca.output("__od_%s_%s.png" % ("_".join(for_group), self.ca.classifier.describe())))
            
            vigra.impex.writeImage(orig_image_in.swapaxes(1,0), self.ca.output("od_inlier_%s_%d_%s.tif" % ("_".join(for_group), orig_image_in.sum(), self.ca.classifier.describe())))
            vigra.impex.writeImage(orig_image_out.swapaxes(1,0), self.ca.output("od_outlier_%s_%d_%s.tif"  % ("_".join(for_group), orig_image_out.sum(), self.ca.classifier.describe())))
            
            ###
            cluster_all_0 = self.ca.get_data(for_group, "PCA", in_classes=(0,), in_class_type="Simple clustering")
            cluster_all_1 = self.ca.get_data(for_group, "PCA", in_classes=(1,), in_class_type="Simple clustering")
            
            ax = plt.subplot(111)    
      
            image_sc0, orig_image_sc0  = self.myscatter(ax, cluster_all_0 [:, 0], cluster_all_0[:, 1], bins=bins)
            image_sc1, orig_image_sc1 = self.myscatter(ax, cluster_all_1[:, 0], cluster_all_1[:, 1], bins=bins) 
            
            fh.write("All-cluster 0\t%d\t%f\n" % (len(cluster_all_0), len(cluster_all_0)    / float(len(cluster_all_0) + len(cluster_all_1)) ))
            fh.write("All-cluster 1\t%d\t%f\n" % (len(cluster_all_1), len(cluster_all_1)    / float(len(cluster_all_0) + len(cluster_all_1)) ))
            fh.write('\n')
            
            qimage_inlier = qimage2ndarray.gray2qimage(image_sc0, False)
            qimage_inlier.setColorTable(QtColorMapFromHex("#00FF00"))
            qimage_outlier = qimage2ndarray.gray2qimage(image_sc1, False)
            qimage_outlier.setColorTable(QtColorMapFromHex("#FF0000"))
     
            qpixmap = blend_images_max([qimage_inlier, qimage_outlier])
            qpixmap.save(self.ca.output("__all_cluster_%s.png" % ("_".join(for_group)) ))
            
            vigra.impex.writeImage(orig_image_sc0.swapaxes(1,0), self.ca.output("all_cluster_0_%s_%d.tif" % ("_".join(for_group), orig_image_in.sum())))
            vigra.impex.writeImage(orig_image_sc1.swapaxes(1,0), self.ca.output("all_cluster_1_%s_%d.tif"  % ("_".join(for_group), orig_image_out.sum())))
            
            for c, cname in class_dict.items():
                data = self.ca.get_data(for_group, "PCA", in_classes=(c,))
                fh.write("Class %d %s\t%d\t%f\n" % (c, cname, len(data), len(data) / float(len(data_normal) + len(data_abnormal)) ))
                ax = plt.subplot(111) 
                image1, image1_org  = self.myscatter(ax, data[:, 0], data[:, 1], bins=bins)    
                vigra.impex.writeImage(image1_org.swapaxes(1,0), self.ca.output("%s_%s_%d_%s.tif" % ("_".join(for_group), "class_%d_%s" % (c, cname), image1_org.sum(), self.ca.classifier.describe())))
                
            fh.write("\n***\nConfusion Outlierdetection\n")
            conf = self.get_outlier_confusion()
            for row in conf:
                fh.write("\t".join(map(str, row)) + "\n")
                
            fh.write("\n")
            for k, v in self.get_stats(conf, split).items():
                fh.write("%s\t%f\n" % (k, v))
                
            fh.write("\n***\nSimple Clustering 1\n")
            conf = self.get_cluster_confusion()
            for row in conf:
                fh.write("\t".join(map(str, row)) + "\n")
                
            fh.write("\n")
            for k, v in self.get_stats(conf, split).items():
                fh.write("%s\t%f\n" % (k, v))
            
            fh.write("\n***\nSimple Clustering 2\n")
            conf = self.get_outlier_confusion(compare_to="Simple clustering", outlier_indicator=0)
            for row in conf:
                fh.write("\t".join(map(str, row)) + "\n")
                
            fh.write("\n")
            for k, v in self.get_stats(conf, split).items():
                fh.write("%s\t%f\n" % (k, v))
            
            

    def myscatter(self, ax, x_vals, y_vals, bins=100, cmap=None):
        if cmap is None: 
            cmap = matplotlib.cm.jet
        cmap._init(); 
        cmap._lut[0,:] = 0
         
        xmin, xmax = x_vals.min(), x_vals.max()
        ymin, ymax = y_vals.min(), y_vals.max()
         
        image, x_edges, y_edges = numpy.histogram2d(x_vals, y_vals, bins=bins)
         
        image = numpy.flipud(image.swapaxes(1,0))
        image = image.astype(numpy.float32)
        orig_img = image.copy()
        image /= image.max()
        image*=254
        image[image>0]+=1
        
         
        ax.imshow(image, interpolation="nearest", extent=[xmin, xmax, ymin, ymax], cmap=cmap)
        
        return image, orig_img

class OutlierFeatureSelection(object):
    def __init__(self, outlier_detection):
        self.outlier_detection = outlier_detection
        self.active_set = []
        self.active_set_score = []
        self.training_matrix = self.outlier_detection.get_data(("neg",), "Object features")
        self.training_matrix = self.outlier_detection.normalize_training_data(self.training_matrix)

        
    
    def _evaluate_fs(self):
        mapping = self.outlier_detection.mapping
        
        neg_out = mapping[(mapping['Object count'] > 10) & (mapping['Group'] == 'neg')]['Outlyingness']
        pos_out = mapping[(mapping['Object count'] > 10) & (mapping['Group'] == 'pos')]['Outlyingness']
        
        return 1-3*(neg_out.std() + pos_out.std()) / (numpy.abs(neg_out.mean()-pos_out.mean())) 
        
        
        
        
    def zfactor_fs_rank(self, stop_after):
        result = {}
        for f_idx in range(self.training_matrix.shape[1]):
            if f_idx not in self.active_set:
                current_features = sorted(self.active_set + [f_idx])
                self.outlier_detection.set_gamma(0.01)
                
                current_matrix = self.training_matrix[:, current_features]
                self.outlier_detection.train_classifier(current_matrix)
                self.outlier_detection.predict()
                self.outlier_detection.compute_outlyingness()
                
                result[f_idx] = self._evaluate_fs()        
        best_feature_idx = sorted([(v,k) for k,v in result.items()])[-1][1]
        self.active_set.append(best_feature_idx)
        self.active_set_score.append(sorted([(v,k) for k,v in result.items()]))
        if len(self.active_set) < stop_after:
            self.zfactor_fs_rank(stop_after)
            
        
                
                
                
            
            
        
        
        
        
        
        
        
        

      

    
        

