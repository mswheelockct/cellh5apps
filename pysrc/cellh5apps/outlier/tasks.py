import numpy

import h5py
import cellh5
import cellh5_analysis

from sklearn.manifold import MDS, SpectralEmbedding

from matplotlib import pyplot as plt

class Task(object):
    def __init__(self, cellh5_analysis):
        self.cellh5_analysis = cellh5_analysis
        
    def run(self, *args, **kwargs):
        self._run(self, *args, **kwargs)

class ClassLabelMDS(Task):
    def _run(self, *args, **kwargs):
        ca = self.cellh5_analysis
        
        class_labels = ca.get_column_as_matrix("Object classification label")
        feature_matrix =  ca.get_column_as_matrix("Object features")
        group_assignment = ca.get_column_as_matrix("Group")
        
        map_dict = {"neg":0, "target":1, "pos":2}
        tmp = group_assignment.map(lambda x: map_dict[x])
        
        n_samples = 200
        
        mds_skl = MDS()
        classes = numpy.unique(class_labels)
        cmap = plt.get_cmap("jet", len(classes))
        examples = []
        for i, cls in enumerate(classes):
            cls_selection = class_labels==cls
            cls_selction_idx = numpy.nonzero(cls_selection)[0]
            numpy.random.shuffle(cls_selction_idx)
            examples.append(feature_matrix[cls_selction_idx[:n_samples], :])

        matrix_examples = numpy.concatenate(examples, 0)
        
        pos = mds_skl.fit_transform(matrix_examples)
        
        for i, cls in enumerate(classes):
            if i < 5:
                color = "g"
            else:
                color = "r"
            plt.scatter(pos[i*n_samples:(i+1)*n_samples, 0], pos[i*n_samples:(i+1)*n_samples,1], color=color, label="class %d" % cls) 
            print i, cls, cmap(cls)
        plt.legend()
            
        plt.show()

        
        
    
    
