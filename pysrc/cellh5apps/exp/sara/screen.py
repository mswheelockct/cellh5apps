EXP = {'sara_od':
        {
        'mapping_files' : {
            'SP_9': 'F:/sara_adhesion_screen/sp9.txt',
            'SP_8': 'F:/sara_adhesion_screen/sp8.txt',
            'SP_7': 'F:/sara_adhesion_screen/sp7.txt',
            'SP_6': 'F:/sara_adhesion_screen/sp6.txt',
            'SP_5': 'F:/sara_adhesion_screen/sp5.txt',
            'SP_4': 'F:/sara_adhesion_screen/sp4.txt',
            'SP_3': 'F:/sara_adhesion_screen/sp3.txt',
            'SP_2': 'F:/sara_adhesion_screen/sp2.txt',
            'SP_1': 'F:/sara_adhesion_screen/sp1.txt',
        },
        'ch5_files' : {
            'SP_9': 'F:/sara_adhesion_screen/sp9__all_positions_with_data_combined.ch5',
            'SP_8': 'F:/sara_adhesion_screen/sp8__all_positions_with_data_combined.ch5',
            'SP_7': 'F:/sara_adhesion_screen/sp7__all_positions_with_data_combined.ch5',
            'SP_6': 'F:/sara_adhesion_screen/sp6__all_positions_with_data_combined.ch5',
            'SP_5': 'F:/sara_adhesion_screen/sp5__all_positions_with_data_combined.ch5',
            'SP_4': 'F:/sara_adhesion_screen/sp4__all_positions_with_data_combined.ch5',
            'SP_3': 'F:/sara_adhesion_screen/sp3__all_positions_with_data_combined.ch5',
            'SP_2': 'F:/sara_adhesion_screen/sp2__all_positions_with_data_combined.ch5',
            'SP_1': 'F:/sara_adhesion_screen/sp1__all_positions_with_data_combined.ch5',
        },
        'locations' : (
            ("F",  19), ("B", 8), ("H", 9), ("D", 8),
            ("H", 6), ("H", 7), ("G", 6), ("G", 7),
            ("H",12), ("H",13), ("G",12), ("G",13),
            ),
        'rows' : list("ABCDEFGHIJKLMNOP")[:],
        'cols' : tuple(range(19,25)),
        'training_sites' : (5,6,7,8),
        'training_sites' : (1,2,3,4),
        'gamma' : 0.005,
        'nu' : 0.15,
        'pca_dims' : 50,
        'kernel' :'rbf'
        },
       }



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
        if DEBUG:
            print "  %s_%s" % (pos.well, pos.pos), "%d/%d" % (idx.sum(), len(idx)),  'are live mitotic'
        
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
        
if __name__ == "__main__":
    print __file__