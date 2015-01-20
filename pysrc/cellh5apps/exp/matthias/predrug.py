import numpy
from cellh5apps.outlier import OutlierDetection, OutlierDetectionSingleCellPlots, OutlierClusterPlots
from cellh5apps.outlier.learner import OneClassSVM, OneClassSVM_SKL, ClusterGMM, ClusterKM
from cellh5apps.exp import EXP
import time
import pandas
import faulthandler
faulthandler.enable()

if __name__ == "__main__":
    from PyQt4 import QtGui   
    app = QtGui.QApplication([])
    if False: 
        od = OutlierDetection("matthias", **EXP['matthias_predrug_a6'])
        od.set_max_training_sample_size(8000)
        od.read_feature(remove_feature=(18, 62, 92, 122, 152))
        
        od_plots = OutlierDetectionSingleCellPlots(od)
        
        def func(nu, gamma):
            feature_set="Object features"
            od.train(classifier_class=OneClassSVM_SKL, gamma=gamma, nu=nu, kernel="rbf", feature_set="Object features")
            od.predict(feature_set=feature_set)
            od.compute_outlyingness()
            od_plots.evaluate(2)
            
    #     od_plots.grid_search(func, nu=[0.1, 0.13, 0.15], gamma=[(kk / 1000.0) for kk in range(1,50,4)])
        od_plots.grid_search(func, nu=[0.13], gamma=[0.035,])
        od_cluster = OutlierClusterPlots(od)
        od_cluster.cluster(ClusterGMM, feature_names=('n2_avg', 'n2_stddev', 'roisize',  'eccentricity',  'h4_ASM',  'h8_2COR', 'granu_close_volume_7', 'granu_open_volume_5'), n_components=5, covariance_type="diag")
        od_cluster.evaluate()
        od_cluster.export_cluster_representatives((16,8))
        od_cluster.export_galleries_per_pos()
    elif True:
        od = OutlierDetection("predrug_a_6_fig_new", **EXP['matthias_predrug_a6_plates_1_8'])
#         od = OutlierDetection("predrug_fig_1", **EXP['matthias_predrug_a6_plate_9'])
        od.set_max_training_sample_size(16000)
        od.read_feature(remove_feature=(18, 62, 92, 122, 152))
        od.pca_run()
        od.cluster_run(ClusterGMM, max_samples=10000, covariance_type="full", n_components=2)
        
        od_plots = OutlierDetectionSingleCellPlots(od)
        def func(nu, gamma):
            feature_set="Object features"
            
            od.train(classifier_class=OneClassSVM_SKL, gamma=gamma, nu=nu, kernel="rbf", feature_set=feature_set)
            tic = time.time()
            od.predict(feature_set=feature_set)
            print 'old took', time.time() - tic
            tic = time.time()
            od.predict2(feature_set=feature_set)
            print 'new took', time.time() - tic
            od.compute_outlyingness()
            od_plots.evaluate(1)
            od_plots.show_feature_space(1,('target',), cut_to_percentile=2)
            
        od_plots.grid_search(func, nu=[0.05, 0.10], gamma=[0.001, 0.006, 0.008, 0.01, 0.012, 0.015, 0.03, 0.1])
        od_plots.export_result_table()
#         od_plots.grid_search(func, nu=[0.13], gamma=[0.002, 0.05, 0.008])
#         od_cluster = OutlierClusterPlots(od)
#         od_cluster.cluster(ClusterGMM, feature_names=('n2_avg', 'n2_stddev', 'roisize',  'eccentricity',  'h4_ASM',  'h8_2COR', 'granu_close_volume_7', 'granu_open_volume_5'), n_components=5, covariance_type="diag")
#         od_cluster.evaluate()
#         od_cluster.export_cluster_representatives((16,8))
#         od_cluster.export_galleries_per_pos()

    elif True:
#         od = OutlierDetection("predrug_fig_1", **EXP['matthias_predrug_a6_plates_1_8'])
        od = OutlierDetection("predrug_fig_1", **EXP['matthias_predrug_a6_plate_9'])
        od.set_max_training_sample_size(10000)
        od.read_feature(remove_feature=(18, 62, 92, 122, 152))
        od.pca_run()


#         od.train(classifier_class=OneClassSVM_SKL, gamma=0.035, nu=0.13, kernel="rbf", feature_set=feature_set)
        
        N = 10000
        result = []
        print "OD ***************************************"
        for eps in [0, 0.025, 0.05, 0.075, 0.1]:
            for c, mr in enumerate(numpy.linspace(0.00,0.5,21)):
                ir = (1 - eps) - mr
                class_fractions = {(0,): ir, 
                                   (1,): mr,
                                   (2,3,4,5): eps}
                for r in range(5):
                    data_od = od.get_data_sampled(("neg",), in_classes=class_fractions, n_sample=N)
                    od.train_classifier (data_od, classifier_class=OneClassSVM_SKL, gamma=0.01, nu=0.10, kernel="rbf")
                    od.predict(feature_set="Object features")
                    
                    od.compute_outlyingness()
                    od_plots = OutlierDetectionSingleCellPlots(od) 
                    
                    stats_od = od_plots.evaluate(2, "all")
                    bacc_od = stats_od["bacc"]
                    tpr_od = stats_od["tpr"]
                    fpr_od = stats_od["fpr"]
                    pre_od = stats_od["pre"]   
                    
                    data_gmm = od.get_data_sampled(("neg", "target", "pos"), in_classes=class_fractions, type_="PCA", n_sample=N)
                    
                    od.cluster_run(ClusterGMM, max_samples=16000, covariance_type="full", n_components=2, data=data_gmm)
                    stats_gmm = od_plots.evaluate_cluster(2, "all")
                    bacc_gmm = stats_gmm["bacc"]
                    tpr_gmm = stats_gmm["tpr"]
                    fpr_gmm = stats_gmm["fpr"]
                    pre_gmm = stats_gmm["pre"] 
                    res = [c, r, ir, mr, eps, bacc_od, tpr_od, fpr_od, pre_od, bacc_gmm, tpr_gmm, fpr_gmm, pre_gmm ]
                    result.append(res)
                    #print "%d\t%d\t%f\t%f\t%f\t%f\t%f%f\t%f\t%f" % tuple(res)
                
        pd_result = pandas.DataFrame(numpy.array(result), columns=["run", "rep", "int_r", "mito_r", "rest_r", "od_bacc", "od_trp", "od_fpr", "od_pre", "gmm_bacc", "gmm_tpr", "gmm_fpr", "gmm_pre",])
        pd_result.to_csv(od.output("_gmm_vs_od.txt"), sep="\t")
                                     
        print "END ***************************************"
        
 
        #od_plots.show_feature_space(2,("target",))

    print "Fini"
        
        
        