import numpy
import vigra
import pylab

from cellh5apps.outlier.learner import OneClassMahalanobis
from cellh5apps.outlier import OutlierDetection

EXP = {'matthias_od':
        {
        'mapping_files' : {
           '002324': 'M:/experiments/Experiments_002300/002324/meta/CellCog/mapping/MD9_Grape_over_Time.txt',
        },
        'ch5_files' : {
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
        'ch5_files' : {
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
        #                       'rows' : list("ABCDEFGHIJKLMNOP")[:3],
        #                         'cols' : tuple(range(19,25)),
        
        #                       'gamma' : 0.01,
        #                       'nu' : 0.2,
        #                       'pca_dims' : 20,
        #                       'kernel' :'rbf'
        'gamma' : 0.05,
        'nu' : 0.05,
        'pca_dims' : 68,
        'kernel' :'linear'
        }
   }
        




class MatthiasOutlier(object):
    def __init__(self, name, mapping_files, ch5_files, rows=None, cols=None, locations=None, gamma=None, nu=None, pca_dims=None, kernel=None):
        if True:
            self.od = self._init(name, mapping_files, ch5_files, rows=rows, cols=cols, locations=locations)
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
        
        greater_less = lambda x, cv: numpy.logical_and(numpy.greater(x, cv[0]), numpy.less(x, cv[1]))
        od.set_read_feature_time_predicate(numpy.equal, 7)
#         od.set_read_feature_time_predicate(greater_less, (5, 10))
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
    def __init__(self, name, mapping_files, ch5_files, rows=None, cols=None, locations=None, gamma=None, nu=None, pca_dims=None, kernel=None):
        self.od = self._init(name, mapping_files, ch5_files, rows=rows, cols=cols, locations=locations)
        
        if True:
            self.od.set_gamma(gamma)
            self.od.set_nu(nu)
            self.od.set_pca_dims(pca_dims)
            self.od.set_kernel(kernel)
            
            
    
            self.od.train_pca()
            self.od.predict_pca()
            self.od.train(classifier_class=OneClassMahalanobis)
            
            self.od.predict()
            self.od.compute_outlyingness()
            
            self.od.cluster_outliers(True)
            print self.od.evaluate_outlier_detection()
            
            self.od.interactive_plot()
        
        elif False:
            
            
            result = {}
            #for nu in [0.05, 0.1, 0.2, 0.5, 0.99]:
                
            for gamma in [0.1,0.01, 0.001, 0.0001, 0.00001, 0.000001]:
                for pca_dims in [10, 20, 50, 100, 239]:
        
                    self.od.set_gamma(gamma)
                    self.od.set_nu(nu)
                    self.od.set_pca_dims(pca_dims)
                    self.od.set_kernel(kernel)
                    
                    
            
                    self.od.train_pca()
                    self.od.predict_pca()
                    self.od.train()
                    self.od.predict()
                    self.od.compute_outlyingness()
                    
                    # Figure 1 c
                    result[(gamma,pca_dims)]=  self.od.evaluate_roc()
            #colors = dict(zip([0.05, 0.1, 0.2, 0.5, 0.99], ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3' , '#ff7f00']))
            colors = dict(zip( [0.1,0.01, 0.001, 0.0001, 0.00001, 0.00005, 0.000001], ['#FF00FF', '#00FFFF', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3' , '#ff7f00']))
            pylab.figure(figsize=(8,8))
            ax = pylab.subplot(111)
            for (nu,pca_dims), (fpr, tpr, th, roc_auc, fpr_f, tpr_f ) in sorted(result.items()):
                ax.plot(fpr, tpr, color=colors[nu], lw=2, label='nu = %3.6f %d (area = %0.2f)' % (nu, pca_dims, roc_auc))
                ax.plot(fpr_f[1], tpr_f[1], color=colors[nu], marker='o')
            
            ax.plot([0, 1], [0, 1], 'k--', label="Random guess")
            pylab.xlim([0.0, 1.0])
            pylab.ylim([0.0, 1.0])
            ax. set_aspect('equal')
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            pylab.xlabel('False Positive Rate')
            pylab.ylabel('True Positive Rate')
            pylab.title('Receiver operating characteristic (ROC)')
            pylab.legend(loc="lower right")
            
            pylab.tight_layout()
            pylab.savefig(self.od.output('outlier_roc_all.png' ))
            #pylab.show() 
            
            # Figure 1 b
            print self.od.evaluate()
        
        
    def panal_a(self):
        # select example images of all classes
        c5f = self.od.cellh5_handles.values()[0]
        
        for i, each_row in self.od.mapping.iterrows():
            plate_name = each_row['Plate']
            well = each_row['Well']
            site = each_row['Site']
            cellh5_idx = list(each_row['CellH5 object index'])
        
            c5p = c5f.get_position(well, str(site))
            class_prediction = c5p.get_class_prediction()['label_idx'][cellh5_idx]
        
            
            for c in range(7):
                idx_c = numpy.nonzero(class_prediction == c)[0]
                for idx in numpy.array(cellh5_idx)[idx_c[0:64]]:
                    img = c5p.get_gallery_image(idx)
                    vigra.impex.writeImage(img.swapaxes(1,0), self.od.output('%d_%d_%s_%02d.png' % (c, idx, well, site)))
    
    def panal_b(self):
        pass
    
if __name__ == "__main__":
    print __file__
    MatthiasOutlierFigure1('matthias_figure_1', **EXP['matthias_figure_1'])
    print "*** fini ***"