import numpy
from cellh5apps.outlier import OutlierDetection, OutlierDetectionSingleCellPlots, OutlierClusterPlots
from cellh5apps.outlier.learner import OneClassSVM, OneClassSVM_SKL, ClusterGMM
from cellh5apps.exp import EXP

if __name__ == "__main__":    
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
    