import pylab
import numpy
from sklearn.svm import OneClassSVM

def mytest():
    x = numpy.random.randn(200,2)
    x[:,0] -=3
    x[:,0] *=2
    x[:100,:] = numpy.random.randn(100,2)
    x[:100,1] *=5
    x[:100,:] +=5
    x[:,1] -=1
    
    
    
    #x-=5
        
        
    classifier = OneClassSVM(kernel='linear', nu=0.1, gamma=0.25, degree=3)
    classifier.fit(x[:100,:])
    prediction = classifier.predict(x)
    print prediction
    
    pylab.scatter(x[prediction == -1, 0], x[prediction == -1, 1], c='red', marker='d', s=42,zorder=99)
    pylab.scatter(x[prediction == 1, 0], x[prediction == 1, 1], c='green', marker='d', s=42, zorder=99)
    
    x_min = -10
    y_min = -10
    
    x_max = 10
    y_max = 10    
        
    xx, yy = numpy.meshgrid(numpy.linspace(x_min, x_max, 100), numpy.linspace(y_min, y_max, 100))
    # Z = self.classifier.decision_function(numpy.c_[xx.ravel(), yy.ravel()])
    matrix = numpy.zeros((100 * 100, 2))
    matrix[:, 0] = xx.ravel()
    matrix[:, 1] = yy.ravel()
    
    
    Z = classifier.decision_function(matrix)
    Z = Z.reshape(xx.shape)
    # print Z
    # Z = (Z - Z.min())
    # Z = Z / Z.max()
    # print Z.min(), Z.max()
    # Z = numpy.log(Z+0.001)
    
    pylab.contourf(xx, yy, Z, levels=numpy.linspace(Z.min(), 0, 17), cmap=pylab.matplotlib.cm.Reds_r)
    pylab.contour(xx, yy, Z, levels=[0], linewidths=5, colors='red')
    pylab.contourf(xx, yy, Z, levels=numpy.linspace(0, Z.max(), 17), cmap=pylab.matplotlib.cm.Greens)
    
    
    
    clf = classifier
    pl = pylab
    np = numpy
    X = x
    
    w = clf.coef_[0]
    a = -w[0] / w[1]
    a = pl.contour(xx, yy, Z, levels=[0], linewidths=2, colors='blue')
     
    print clf.coef_[0], clf.intercept_
    pylab.show()
 
def scikits_test():   
    import numpy as np
    import pylab as pl
    import matplotlib.font_manager
    from sklearn import svm
    
    xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
    # Generate train data
    X = 0.3 * np.random.randn(100, 2)
    X_train = np.r_[X + 2]
    # Generate some regular novel observations
    X = 0.3 * np.random.randn(20, 2)
    X_test = np.r_[X + 2]
    # Generate some abnormal novel observations
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
    
    # fit the model
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_outliers = clf.predict(X_outliers)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
    
    # plot the line, the points, and the nearest vectors to the plane
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    pl.title("Novelty Detection")
    pl.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=pl.cm.Blues_r)
    a = pl.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
    pl.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')
    
    b1 = pl.scatter(X_train[:, 0], X_train[:, 1], c='white')
    b2 = pl.scatter(X_test[:, 0], X_test[:, 1], c='green')
    c = pl.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red')
    pl.axis('tight')
    pl.xlim((-5, 5))
    pl.ylim((-5, 5))
    pl.legend([a.collections[0], b1, b2, c],
              ["learned frontier", "training observations",
               "new regular observations", "new abnormal observations"],
              loc="upper left",
              prop=matplotlib.font_manager.FontProperties(size=11))
    pl.xlabel(
        "error train: %d/200 ; errors novel regular: %d/20 ; "
        "errors novel abnormal: %d/20"
        % (n_error_train, n_error_test, n_error_outliers))
    pl.show()
    
mytest()