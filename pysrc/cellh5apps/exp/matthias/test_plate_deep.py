import numpy
import vigra
import pylab

from cellh5apps.outlier.learner import OneClassMahalanobis, OneClassAngle, OneClassGMM, OneClassSVM, OneClassKDE,\
    OneClassSVM_SKL
from cellh5apps.outlier import OutlierDetection, OutlierFeatureSelection

from matplotlib import pyplot as plt 
from cellh5apps.utils.colormaps import YlBlCMap

EXP = {'matthias_test_plate_deep':
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
        

class MatthiasPredrug(object):
    def normalize(self, matrix):
        cm2 = matrix.astype(numpy.float32)   
        for i in range(cm2.shape[0]): cm2[i,:] /= cm2[i,:].sum()
        return cm2
    
    def get_stats(self, cm, split):
        result = {}
        result['acc'] = acc = (cm[:split, 0].sum() + cm[split:, 1].sum()) / cm.sum()
        result['tpr'] = tpr = cm[split:,1].sum() / cm[split:,:].sum()
        result['fpr'] = fpr = cm[:split,1].sum() / cm[:split,:].sum()
        result['pre'] = pre = cm[:split,1].sum() / cm[:,1].sum()
        
        result['f1']  = 2*tpr*pre / (tpr+pre)
        result['f2']  = 2*tpr*(1-fpr) / (tpr+(1-fpr))
        
        return result
    
    def evaluate_roc(self, cms, thrhs, split):
        all_stats = []
        for cm, t in zip(cms, thrhs):
            cm_n = self.normalize(cm)       
            stats = self.get_stats(cm_n, split)
            all_stats.append(stats)
            
        self.plot_roc(all_stats)
            
        
            
    def plot_roc(self, all_stats):
        tprs = [stat['tpr'] for stat in all_stats]
        fprs = [stat['fpr'] for stat in all_stats]
        
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.plot(fprs, tprs, 'k.-')
        ax.set_aspect(1)
        ax.set_title(self.od.classifier.describe())
        
        plt.tight_layout()
        plt.savefig(self.od.output("roc_%s.png"  % self.od.classifier.describe()))
        
        plt.close(fig)
        
        
    def evaluate(self, cm, cms, thrhs, split):
        cm_n = self.normalize(cm)
        stats = self.get_stats(cm_n, split)
        
        output=  "%5.4f\t%5.4f\t%5.4f\t%4.3f\t%4.3f\t%4.3f\t%s\n" % (stats['acc'], stats['tpr'], stats['fpr'], stats['pre'], stats['f1'], stats['f2'], self.od.classifier.describe())
        with open(self.od.output("test.txt"), "a") as myfile:
            myfile.write(output)
        
        self.export_cm(cm_n, stats)
               
    def export_cm(self, cm, stats):
        fig = plt.figure()
        
        ax = plt.subplot(111)
        res = ax.pcolor(cm, cmap=YlBlCMap, vmin=0, vmax=1)
        for i, cas in enumerate(cm):
            for j, c in enumerate(cas):
                if c>0:
                    ax.text(j+0.4, i+0.5, "%4.2f" % c, fontsize=10)
        ax.set_ylim(0,cm.shape[0])
        ax.invert_yaxis()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        ax.set_title("acc %3.2f, tpr %3.2f, fpr %3.2f \npre %3.2f, f1 %3.2f" % (stats['acc'], stats['tpr'], stats['fpr'], stats['pre'], stats['f2']) + "\n" + self.od.classifier.describe())
        ax.set_aspect(0.8)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        
        plt.axis('off')
        plt.tight_layout()
        

        fig.savefig(self.od.output("%s.png" % self.od.classifier.describe()))
        plt.close(fig)
        
        
        
    def __init__(self, name, mapping_files, cellh5_files, rows=None, cols=None, locations=None, training_sites=None, gamma=None, nu=None, pca_dims=None, kernel=None):
        self.od = OutlierDetection(name,
                                  mapping_files,
                                  cellh5_files,
                                  rows=rows,
                                  cols=cols,
                                  locations=locations,
                                  gamma=gamma,
                                  nu=nu,
                                  pca_dims=pca_dims,
                                  kernel=kernel,
                                  training_sites=training_sites
                                  )
        self.od.set_read_feature_time_predicate(numpy.equal, 7)
        self.od.read_feature(object_="primary__primary")
        
    def analyze_roc(self):
        self.od.set_read_feature_time_predicate(numpy.equal, 0)
        self.od.read_feature(object_="primary__primary")
        
        # GMM
        for k in [2,10,100]:
            self.od.train(classifier_class=OneClassGMM, k=k)
            self.od.predict()
            self.od.compute_outlyingness()
            cm, cms, thrs = self.od.get_sl_od_confusion_matrix()
            self.evaluate_roc(cms, thrs, 2) 
        
        # Mahala
        self.od.train(classifier_class=OneClassMahalanobis)
        self.od.predict()
        self.od.compute_outlyingness()
        cm, cms, thrs = self.od.get_sl_od_confusion_matrix()
        self.evaluate_roc(cms, thrs, 2)   
        
        # KDE 
        for bandwidth in numpy.linspace(0.01, 5, 4):
            self.od.train(classifier_class=OneClassKDE, bandwidth=bandwidth)
            self.od.predict()
            self.od.compute_outlyingness()
            cm, cms, thrs = self.od.get_sl_od_confusion_matrix()
            self.evaluate_roc(cms, thrs, 2)   
        
        # One class svm
        for nu in [0.01, 0.10, 0.2, 0.99]:
            self.od.set_nu(nu)  
            for gamma in numpy.linspace(0.0001, 0.2, 4):
                self.od.train(classifier_class=OneClassSVM, gamma=gamma, nu=nu, kernel="rbf")
                self.od.predict()
                self.od.compute_outlyingness()
                cm, cms, thrs = self.od.get_sl_od_confusion_matrix()
                self.evaluate_roc(cms, thrs, 2)         
        self.od.write_readme()
        
    def analyze(self):        
        split = 4
        
        # GMM
        for k in [2,10,100]:
            self.od.train(classifier_class=OneClassGMM, k=k)
            self.od.predict()
            self.od.compute_outlyingness()
            cm, cms, thrs = self.od.get_sl_od_confusion_matrix()
            self.evaluate(cm, cms, thrs, split) 
         
        # Mahala
        self.od.train(classifier_class=OneClassMahalanobis)
        self.od.predict()
        self.od.compute_outlyingness()
        cm, cms, thrs = self.od.get_sl_od_confusion_matrix()
        self.evaluate(cm, cms, thrs, split)   
         
        # KDE 
        for bandwidth in numpy.linspace(0.01, 5, 4):
            self.od.train(classifier_class=OneClassKDE, bandwidth=bandwidth)
            self.od.predict()
            self.od.compute_outlyingness()
            cm, cms, thrs = self.od.get_sl_od_confusion_matrix()
            self.evaluate(cm, cms, thrs, split)   
          
        # One class svm
        for nu in [0.01,]:
            self.od.set_nu(nu)  
#             for gamma in numpy.linspace(0.0001, 0.2, 4):
            for gamma in numpy.logspace(-1,-11,base=2, num=11):
                self.od.train(classifier_class=OneClassSVM_SKL, gamma=gamma, nu=nu, kernel="rbf")
                self.od.predict()
                self.od.compute_outlyingness()
                cm, cms, thrs = self.od.get_sl_od_confusion_matrix()
                self.evaluate(cm, cms, thrs, split)         
        self.od.write_readme()
        
if __name__ == "__main__":
    print __file__
    mp = MatthiasPredrug('matthias_test_plate_deep', **EXP['matthias_test_plate_deep'])
    mp.analyze()
    print "*** fini ***"
    