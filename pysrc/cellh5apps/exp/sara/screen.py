import numpy
import vigra
import os
import matplotlib
matplotlib.use('Qt4Agg')

from cellh5apps.outlier import OutlierDetection, OutlierDetectionSingleCellPlots, OutlierClusterPlots
from cellh5apps.outlier.learner import PCA, OneClassSVM, OneClassSVM_SKL, ClusterGMM, ClusterKM

from cellh5apps.exp import EXP


class SaraOutlier(object):
    @staticmethod
    def sara_mitotic_live_selector(pos, plate_name, treatment, outdir):
        pp = pos['object']["primary__primary"]
        
        fret_index = pos.definitions.get_object_feature_idx_by_name('secondary__inside', 'n2_avg')
        topro_ind = pos.definitions.get_object_feature_idx_by_name('quartiary__inside', 'n2_avg')
        yfp_ind = pos.definitions.get_object_feature_idx_by_name('tertiary__inside', 'n2_avg')
        
        fret_inside = pos.get_object_features('secondary__inside')[:, fret_index]
        fret_outside = pos.get_object_features('secondary__outside')[:, fret_index]
        
        yfp_inside = pos.get_object_features('tertiary__inside')[:, yfp_ind]
        yfp_outside = pos.get_object_features('tertiary__outside')[:, yfp_ind]
        
        topro_inside = pos.get_object_features('quartiary__inside')[:, topro_ind]
        topro_outside = pos.get_object_features('quartiary__outside')[:, topro_ind]
        
        try:
            fret_ratio = (fret_inside - fret_outside) / (yfp_inside - yfp_outside)
            
            topro_diff = topro_inside - topro_outside
            
            fret_min = 0.6
            fret_max = 0.82
            idx_1 = numpy.logical_and(fret_ratio > fret_min, fret_ratio < fret_max )
            
            topro_abs_max = 15
            idx_2 = topro_diff < topro_abs_max
            
            idx = numpy.logical_and(idx_1, idx_2)
        except:
            print "@!#$!"*100
            idx = numpy.zeros((len(pp),), dtype=numpy.bool)
        
        # Export images for live mitotic cells
        if False:
            well = pos.well
            site = pos.pos

            ch5_index_mitotic_live = numpy.nonzero(idx)[0]
            ch5_index_not_mitotic_live = numpy.nonzero(numpy.logical_not(idx))[0]
                
            outlier_img = pos.get_gallery_image_matrix(ch5_index_not_mitotic_live, (10, 5)).swapaxes(1,0)
            inlier_img = pos.get_gallery_image_matrix(ch5_index_mitotic_live, (10, 5)).swapaxes(1,0)
            
            img = numpy.concatenate((inlier_img, numpy.ones((5, inlier_img.shape[1]))*255, outlier_img))
            vigra.impex.writeImage(img, os.path.join(outdir, 'mito_live_%s_%s_%s_%s.png' % (plate_name,  well, site, treatment )))
        return idx
    
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
        self.od.read_feature(self.sara_mitotic_live_selector)
#         self.od.read_feature()
        self.od.set_gamma(gamma)
        self.od.set_nu(nu)
        self.od.set_pca_dims(pca_dims)
        self.od.set_kernel(kernel)
        self.od.train_pca(pca_type=PCA)
        self.od.predict_pca()
        self.od.train(classifier_class=OneClassSVM)
        self.od.predict()
        self.od.compute_outlyingness()
        self.od.cluster_outliers()
            
        self.od.interactive_plot()
        
        
#         self.od.cluster_outliers()                    
#         #self.od.interactive_plot()
#         
#         self.od.make_hit_list_single_feature('roisize')
#         
        self.od.make_top_hit_list(top=4000, for_group=('neg', 'target', 'pos'))
#         
#         self.od.make_heat_map()
        self.od.make_outlier_galleries()
        if DEBUG:
            print 'Results:', self.od.output_dir
        os.startfile(os.path.join(os.getcwd(), self.od.output_dir))
        
# if __name__ == "__main__":
#     print __file__
#     SaraOutlier('sara_od', **EXP['sara_od'])
#     print "*** fini ***"

if __name__ == "__main__":   
    od = OutlierDetection("sara_p18", **EXP['sara_screen_plates_1_8'])
    od.set_max_training_sample_size(10000)
    
#     od.read_feature(remove_feature=(18, 62, 92, 122, 152), idx_selector_functor=SaraOutlier.sara_mitotic_live_selector).
    od.read_feature(remove_feature=(16,  17,  18,  62,  92, 122, 137, 138, 152))
    od.pca_run()
    feature_set="Object features"
        
    od.train(classifier_class=OneClassSVM_SKL, gamma=0.035, nu=0.13, kernel="rbf", feature_set=feature_set)
    od.predict(feature_set=feature_set)
    od.compute_outlyingness()
    
    od.cluster_run(ClusterGMM, max_samples=10000, covariance_type="full", n_components=2)
    
    od_plots = OutlierDetectionSingleCellPlots(od)
    
    od_plots.evaluate(2)    
    od_plots.evaluate_cluster(2)
    
    od_plots.show_feature_space(2,("target",))
    
    
    
    