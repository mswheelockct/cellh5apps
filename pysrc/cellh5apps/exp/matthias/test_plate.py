import numpy
import vigra
import pylab

from cellh5apps.outlier.learner import OneClassMahalanobis, OneClassAngle, OneClassGMM
from cellh5apps.outlier import OutlierDetection, OutlierFeatureSelection
from sklearn.svm.classes import OneClassSVM

from matplotlib import pyplot as plt 
from cellh5apps.utils.colormaps import YlBlCMap

EXP = {'matthias_od':
        {
        'mapping_files' : {
           '002324': 'M:/experiments/Experiments_002300/002324/meta/CellCog/mapping/MD9_Grape_over_Time.txt',
        },
        'cellh5_files' : {
           '002324': 'M:/experiments/Experiments_002300/002324/meta/CellCog/analysis_outlier_3/hdf5/_all_positions.ch5',
        },
        'locations' : (
            ("A",  8), ("B", 8), ("C", 8), ("D", 8),
            ("H", 6), ("H", 7), ("G", 6), ("G", 7),
            ("H",12), ("H",13), ("G",12), ("G",13),
            ),
        'rows' : list("ABCDEFGHIJKLMNOP")[:3],
        'cols' : tuple(range(19,25)),
        'gamma' : 0.0001,
        'nu' : 0.10,
        'pca_dims' : 10,
        'kernel' :'linear'
#         'gamma' : 0.005,
#         'nu' : 0.12,
#         'pca_dims' : 100,
#         'kernel' :'rbf'
        },
       
        'matthias_figure_1':
        {
         'mapping_files' : {
            '002324': 'M:/experiments/Experiments_002300/002324/meta/CellCog/mapping/MD9_Grape_over_Time.txt',
        },
        'cellh5_files' : {
           '002324': 'M:/experiments/Experiments_002300/002324/meta/CellCog/analysis_outlier_3/hdf5/_all_positions.ch5',
        },
        'locations' : (
            ("A",  8), ("B", 8), ("C", 8), ("D", 8),
            ("H", 6), ("H", 7), ("G", 6), ("G", 7),
            ("H",12), ("H",13), ("G",12), ("G",13),
            ("A",  8), ("B", 8), ("C", 8), ("D", 8), ("E", 8),
            #                                 ("D",  13), ("F",  13), ("H",  13), # Taxol No Rev
            ("D",  7), ("F",  7), ("H",  7), # Noco No Rev 
            ("D",  12), ("F",  12), ("H",  12), # Taxol 300 Rev
            ("D",  6), ("F",  6), ("H",  6), # Noco 300 Rev
            #                                 ("D",  9), ("F",  9), ("H",  9), # Taxol 900 Rev
            #                                 ("D",  3), ("F",  3), ("H",  3), # Noco 900 Rev
            #                                 
            #                                 ("J",  13), ("L",  13), ("N",  13), # Taxol No Rev
            ("J",  7), ("L",  7), ("N",  7), # Noco No Rev 
            ("J",  12), ("L",  12), ("N",  12), # Taxol 300 Rev
            ("J",  6), ("L",  6), ("N",  6), # Noco 300 Rev
            #                                 ("J",  9), ("L",  9), ("N",  9), # Taxol 900 Rev
            #                                 ("J",  3), ("L",  3), ("N",  3), # Noco 900 Rev
        
        
        #                             ("B",  19), ("C",  19), ("D",  19), ("E",  19), # NEG
        #                             ("D",  24), ("F",  24), ("H",  24), # Taxol No Rev
        #("D",  18), ("F",  18), ("H",  18), # Noco No Rev 
        #("D",  23), ("F",  23), ("H",  23), # Taxol 300 Rev
        #("D",  17), ("F",  17), ("H",  17), # Noco 300 Rev
        #                             ("D",  20), ("F",  20), ("H",  20), # Taxol 900 Rev
        #                             ("D",  14), ("F",  14), ("H",  14), # Noco 900 Rev
        ),
        'gamma' : 0.0012,
        'nu' : 0.1,
        'pca_dims' : 68,
        'kernel' :'rbf',
        }
   }
        

class MatthiasOutlier(object):
    def __init__(self, name, mapping_files, cellh5_files, rows=None, cols=None, locations=None, gamma=None, nu=None, pca_dims=None, kernel=None):
        if True:
            self.od = self._init(name, mapping_files, cellh5_files, rows=rows, cols=cols, locations=locations)
            self.od.set_gamma(gamma)
            self.od.set_nu(nu)
            self.od.set_pca_dims(pca_dims)
            self.od.set_kernel(kernel)
            self.od.train_pca()
            self.od.predict_pca()
            self.od.train()
            self.od.predict()
            self.od.compute_outlyingness()
            print self.od.evaluate()
            self.od.interactive_plot()
            self.od.make_outlier_galleries_per_pos_per_class()
            
            
        else:
            result = {}
            self.od = self._init(name, mapping_files, ch5_files, rows=rows, cols=cols, locations=locations)
            for kernel in ['rbf', 'linear']:
                for nu in [0.05, 0.1, 0.2]:
                    for pca_dims in [10, 100, 239]:
                        for gamma in [0.001, 0.0001, 0.00001, 0.000001]:
                            self.od.set_nu(nu)
                            self.od.set_gamma(gamma)
                            self.od.set_pca_dims(pca_dims)
                            self.od.set_kernel(kernel)
                            self.od.train_pca()
                            self.od.predict_pca()
                            self.od.train()
                            self.od.predict()
                            self.od.compute_outlyingness()
                            tmp = self.od.evaluate()
                            result[(kernel, nu, pca_dims, gamma)] = tmp
                            print (kernel, nu, pca_dims, gamma), '----->', tmp
                            
            for k, rank in enumerate(sorted([(result[k],k) for k in result], reverse=True)):
                print rank
                if k > 30:
                    break
        
        
    def _init(self, name, mapping_files, ch5_files, rows, cols, locations):
        od = OutlierDetection(name,
                                  mapping_files,
                                  ch5_files,
                                  rows=rows,
                                  cols=cols,
                                  locations=locations,
                                  )
        
        od.set_read_feature_time_predicate(numpy.equal, 8)
        
        od.read_feature()
        
        return od

        #self.od.cluster_outliers()                    
        #self.od.make_pca_scatter()
        #self.od.interactive_plot()
        #self.od.make_outlier_galleries()
        
        #self.od.make_heat_map()
        #self.od.make_hit_list()
        #self.od.make_outlier_galleries()
        #print 'Results:', self.od.output_dir
        #os.startfile(os.path.join(os.getcwd(), self.od.output_dir))
        
class MatthiasOutlierFigure1(MatthiasOutlier):
    def __init__(self, name, mapping_files, cellh5_files, rows=None, cols=None, locations=None, gamma=None, nu=None, pca_dims=None, kernel=None):
        self.od = self._init(name, mapping_files, cellh5_files, rows=rows, cols=cols, locations=locations)
        
        if False:
            self.od.set_gamma(gamma)
            self.od.set_nu(nu)
            self.od.set_pca_dims(pca_dims)
            self.od.set_kernel(kernel)
            
            
    
            self.od.train_pca()
            self.od.predict_pca()
            self.od.train(classifier_class=OneClassSVM)
            
            self.od.predict()
            self.od.compute_outlyingness()
            
            self.od.cluster_outliers(True)
            cm = self.od.get_sl_od_confusion_matrix()[0]
            self.evaluate(cm)
            self.od.interactive_plot()
        
        elif True:
            fs = [208, 150, 217, 227, 221, 226, 223, 219, 238, 237, ]#228, 235, 233, 231, 189, 218, 169, 190, 214, 220, 229, 234, 236, 230, 232, 224, 225, 222, 201, 210, 204, 209, 192, 191, 199, 212, 211, 213, 216, 215, 193, 203, 207,   7, 188, 110, 133, 194, 197, 200]
            #fs = [110, 25, 212, 218, 220, 219, 221, 223, 217, 226, 225, 232, 236, 235, 238, 237, 230, 231, 228, 224, 227, 229, 27, 234, 233, 139, 152, 66, 104, 141, 169, 222, 215, 216, 151, 210, 194, 211, 214, 213, 204, 209, 197, 196, 199, 205, 207, 208, 198, 206]

            result = {}
            self.od.set_kernel('rbf')
            output=  "gamma\tnu\tpca_dim\tacc\ttpr\tfpr\tprecision\tFM\tclassifier\n"
            with open(self.od.output("test.txt"), "w") as myfile:
                myfile.write(output)
            self.od.feature_set = "Object features"
            self.od.set_feature_selection(None)
            self.od.set_pca_dims(-1)
            
            for gamma in [0.1,0.01, 0.001, 0.0001, 0.00001, 0.000001]:
                for nu in [0.01, 0.05, 0.1, 0.2, 0.5, 0.99]:
                    for classifier in [OneClassGMM, ]:
                        self.od.set_gamma(gamma)
                        self.od.set_nu(nu)
                        self.od.train(classifier_class=classifier)
                        self.od.predict()
                        self.od.compute_outlyingness()
                        cm = self.od.get_sl_od_confusion_matrix()[0] 
                        self.evaluate(cm)
                
            self.od.feature_set = "PCA"
            self.od.set_feature_selection(None)
            self.od.set_pca_dims(50)
            self.od.train_pca(('neg',))
            for gamma in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]:
                for nu in [0.01, 0.05, 0.1, 0.2, 0.5, 0.99]:
                    for pca_dims in [50]:
                        for classifier in [OneClassGMM,]:
                            self.od.set_gamma(gamma)
                            self.od.set_nu(nu)
                            self.od.set_pca_dims(pca_dims)
                            self.od.train_pca()
                            self.od.predict_pca()
                            self.od.train(classifier_class=classifier)
                            self.od.predict()
                            self.od.compute_outlyingness()
                            cm = self.od.get_sl_od_confusion_matrix()[0] 
                            self.evaluate(cm)
                            
        elif False:
            fs = [208, 150, 217, 227, 221, 226, 223, 219, 238, 237, 228, 235, 233, 231, 189, 218, 169, 190, 214, 220, 229, 234, 236, 230, 232, 224, 225, 222, 201, 210, 204, 209, 192, 191, 199, 212, 211, 213, 216, 215, 193, 203, 207,   7, 188, 110, 133, 194, 197, 200]
            #fs = [110, 25, 212, 218, 220, 219, 221, 223, 217, 226, 225, 232, 236, 235, 238, 237, 230, 231, 228, 224, 227, 229, 27, 234, 233, 139, 152, 66, 104, 141, 169, 222, 215, 216, 151, 210, 194, 211, 214, 213, 204, 209, 197, 196, 199, 205, 207, 208, 198, 206]
            #fs = [99, 189, 229, 231, 232, 236, 211, 216, 220, 223, 237, 235, 227, 219, 234, 230, 111, 217, 238, 205, 198, 201, 210, 228, 233, 221, 215, 226, 224, 44, 212, 225, 222, 18, 74, 213, 12, 214, 66, 70, 9, 206, 202, 169, 218, 8, 174, 187, 88, 178]

            self.od.set_kernel('rbf')
            output=  "gamma\tnu\tdim\tacc\ttpr\tfpr\tprecision\tFM\tclassifier\n"
            with open(self.od.output("test.txt"), "w") as myfile:
                myfile.write(output)
                
            self.od.feature_set = "Object features"
            self.od.set_feature_selection(fs)
            self.od.set_pca_dims(-1)
            
            for gamma in [0.01, 0.05, 0.001,]:
                for nu in [0.1, 0.2,]:
                    self.od.set_pca_dims(239)
                    self.od.set_feature_selection(None)
                    self.od.set_gamma(gamma)
                    self.od.set_nu(nu)
                    self.od.train(classifier_class=OneClassSVM)
                    self.od.predict()
                    self.od.compute_outlyingness()
                    cm = self.od.get_sl_od_confusion_matrix()[0] 
                    self.evaluate(cm) 
                    
                    for f in range(2,len(fs)):
                        self.od.set_pca_dims(f)
                        self.od.set_feature_selection(fs[:f])
                        self.od.set_gamma(gamma)
                        self.od.set_nu(nu)
                        self.od.train(classifier_class=OneClassSVM)
                        self.od.predict()
                        self.od.compute_outlyingness()
                        cm = self.od.get_sl_od_confusion_matrix()[0] 
                        self.evaluate(cm) 
                                  
    def evaluate(self, cm):
        acc = (cm[:4, 0].sum() + cm[4:, 1].sum()) / cm.sum()
        tpr = cm[4:,1].sum() / cm[4:,:].sum()
        fpr = cm[:4,1].sum() / cm[:4,:].sum()
        pre = cm[:4,1].sum() / cm[:,1].sum()
        output=  "%7.6f\t%4.3f\t%d\t%5.4f\t%5.4f\t%5.4f\t%4.3f\t%4.3f\t%s\n" % (self.od.gamma, self.od.nu, self.od.pca_dims, acc, tpr, fpr, pre, 2*tpr*pre / (tpr+pre), self.od.classifier.__class__.__name__)
        with open(self.od.output("test.txt"), "a") as myfile:
            myfile.write(output)
        self.export_cm(cm)
               
    def export_cm(self, cm):
        fig = plt.figure()
        
        ax = plt.subplot(121)
        res = ax.pcolor(cm, cmap=YlBlCMap,)
        for i, cas in enumerate(cm):
            for j, c in enumerate(cas):
                if c>0:
                    ax.text(j+0.3, i+0.5, "%d"%c, fontsize=10)
        ax.set_ylim(0,cm.shape[0])
        ax.invert_yaxis()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        acc = (cm[:4, 0].sum() + cm[4:, 1].sum()) / cm.sum()
        tpr = cm[4:,1].sum() / cm[4:,:].sum()
        fpr = cm[:4,1].sum() / cm[:4,:].sum()
        pre = cm[:4,1].sum() / cm[:,1].sum()
        
        ax.set_title("acc %3.2f, tpr %3.2f, fpr %3.2f \npre %3.2f, f1 %3.2f" % (acc, tpr, fpr, pre, 2*tpr*pre / (tpr+pre)))
        
        cm2 = cm.copy()   
        for i in range(cm2.shape[0]): cm2[i,:] /= cm2[i,:].sum()
          
        ax = plt.subplot(122)
        res = ax.pcolor(cm2, cmap=YlBlCMap, vmin=0, vmax=1)
        for i, cas in enumerate(cm2):
            for j, c in enumerate(cas):
                if c>0:
                    ax.text(j+0.5, i+0.5, "%3.2f" % c, fontsize=10)
        ax.set_ylim(0,cm2.shape[0])
        ax.invert_yaxis()         
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        acc = (cm2[:4, 0].sum() + cm2[4:, 1].sum()) / cm2.sum()
        tpr = cm2[4:,1].sum() / cm2[4:,:].sum()
        fpr = cm2[:4,1].sum() / cm2[:4,:].sum()
        pre = cm2[:4,1].sum() / cm2[:,1].sum()
        
        ax.set_title("acc %3.2f, tpr %3.2f, fpr %3.2f \n pre %3.2f, f1 %3.2f" % (acc, tpr, fpr, pre, 2*tpr*pre / (tpr+pre)))
        
        fig.savefig(self.od.output("g%7.6f__n%4.3f__p%d.png" % (self.od.gamma, self.od.nu, self.od.pca_dims)))
        plt.close(fig)
          
def test_fs():
    od = OutlierDetection('fs', **EXP['matthias_figure_1'])    
    greater_less = lambda x, cv: numpy.logical_and(numpy.greater(x, cv[0]), numpy.less(x, cv[1]))
    od.set_read_feature_time_predicate(greater_less, (5, 7))
    od.read_feature()
    od.feature_set = "Object features"
    
    od.set_kernel('rbf')
    od.set_nu(0.1)
    
    fs = OutlierFeatureSelection(od)
    fs.zfactor_fs_rank(50)
    
    print "\n\n\n"
    print "*"*20
    print "\n"
    print [i for i in range(239) if i not in od._non_nan_feature_idx]
    print fs.active_set
    
    cf = od.cellh5_handles.values()[0]
    feature_names = cf.object_feature_def()
    feature_names = [feature_names[ff] for ff in od._non_nan_feature_idx]
    
    for f in fs.active_set: print f, feature_names[f]
    
    print 'fs fini'
    
    
    
if __name__ == "__main__":
    print __file__
    MatthiasOutlierFigure1('matthias_figure_1', **EXP['matthias_figure_1'])
#     test_fs()

    print "*** fini ***"