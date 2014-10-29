import numpy
from cellh5apps.outlier import OutlierDetection, PCA, OneClassSVM

EXP = {'matthias_predrug_a6':
        {
        'mapping_files' : {
            'Screen_Plate_01': 'F:/matthias_predrug_a6/Screen_Plate_01_position_map_PRE.txt',
            'Screen_Plate_02': 'F:/matthias_predrug_a6/Screen_Plate_02_position_map_PRE.txt',
            'Screen_Plate_03': 'F:/matthias_predrug_a6/Screen_Plate_03_position_map_PRE.txt',
            'Screen_Plate_04': 'F:/matthias_predrug_a6/Screen_Plate_04_position_map_PRE.txt',
            'Screen_Plate_05': 'F:/matthias_predrug_a6/Screen_Plate_05_position_map_PRE.txt',
            'Screen_Plate_06': 'F:/matthias_predrug_a6/Screen_Plate_06_position_map_PRE.txt',
            'Screen_Plate_07': 'F:/matthias_predrug_a6/Screen_Plate_07_position_map_PRE.txt',
            'Screen_Plate_08': 'F:/matthias_predrug_a6/Screen_Plate_08_position_map_PRE.txt',
            'Screen_Plate_09': 'F:/matthias_predrug_a6/Screen_Plate_09_position_map_PRE.txt',
            },
        'ch5_files' : {
            'Screen_Plate_01': 'F:/matthias_predrug_a6/Screen_Plate_01_all_positions_with_data.ch5',
            'Screen_Plate_02': 'F:/matthias_predrug_a6/Screen_Plate_02_all_positions_with_data.ch5',
            'Screen_Plate_03': 'F:/matthias_predrug_a6/Screen_Plate_03_all_positions_with_data.ch5',
            'Screen_Plate_04': 'F:/matthias_predrug_a6/Screen_Plate_04_all_positions_with_data.ch5',
            'Screen_Plate_05': 'F:/matthias_predrug_a6/Screen_Plate_05_all_positions_with_data.ch5',
            'Screen_Plate_06': 'F:/matthias_predrug_a6/Screen_Plate_06_all_positions_with_data.ch5',
            'Screen_Plate_07': 'F:/matthias_predrug_a6/Screen_Plate_07_all_positions_with_data.ch5',
            'Screen_Plate_08': 'F:/matthias_predrug_a6/Screen_Plate_08_all_positions_with_data.ch5',
            'Screen_Plate_09': 'F:/matthias_predrug_a6/Screen_Plate_09_all_positions_with_data.ch5',
            },
        'locations' : (
            ("A",  4), ("B", 23), ("H", 9), ("D", 8),
            ("H", 6), ("A", 7), ("G", 6), ("G", 7),
            ("H",12), ("H",13), ("G",12), ("A", 9),
            ),
        'rows' : list("A")[:],
        'cols' : tuple(range(4,5)),
        'training_sites' : (5,6,7,8),
        'training_sites' : (1,2,3,4),
        'gamma' : 0.001,
        'nu' : 0.2,
        'pca_dims' : 50,
        'kernel' :'rbf'
        }
       }

class MatthiasPredrug(object):
    def __init__(self, name, mapping_files, ch5_files, rows=None, cols=None, locations=None, training_sites=None, gamma=None, nu=None, pca_dims=None, kernel=None):
        self.od = OutlierDetection(name,
                                  mapping_files,
                                  ch5_files,
                                  rows=rows,
                                  cols=cols,
                                  locations=locations,
                                  gamma=gamma,
                                  nu=nu,
                                  pca_dims=pca_dims,
                                  kernel=kernel,
                                  training_sites=training_sites
                                  )
        self.od.set_read_feature_time_predicate(numpy.equal, 0)
        self.od.read_feature(object_="primary__primary")
        self.od.set_gamma(gamma)
        self.od.set_nu(nu)
        self.od.set_pca_dims(pca_dims)
        self.od.set_kernel(kernel)
        self.od.train_pca(pca_type=PCA)
        self.od.predict_pca()
        self.od.train(classifier_class=OneClassSVM)
        self.od.predict()
        self.od.compute_outlyingness()
        #self.od.export_to_file(6)
        self.od.cluster_outliers()
        print self.od.evaluate_outlier_detection()
        #self.od.make_top_hit_list(top=4000, for_group=('neg', 'target', 'pos'))
        
        self.od.interactive_plot()
        
        self.od.write_readme()
        
if __name__ == "__main__":
    print __file__
    MatthiasPredrug('mathias_predrug', **EXP['matthias_predrug_a6'])
    print "*** fini ***"
    