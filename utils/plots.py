COLOR_LUT_6 = OrderedDict([
        ("blue"    , "#000087"),
        ("green"   , "#00A600"),
        ("red"     , "#AF0000"),
        ("magenta" , "#904190"),
        ("yellow"  , "#EFEF43"),
        ("cyan"    , "#00EFEF")])

DEBUG = True

YlBlCMap = matplotlib.colors.LinearSegmentedColormap.from_list('asdf', [(0,0,1), (1,1,0)])


def invert_plots(func):
    def wrapper(*arg):
        rcParamsBackup = rcParams.copy()
        
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.size'] = 20
        rcParams['font.sans-serif'] = ['Arial']
        rcParams['xtick.labelsize'] = 16 
        rcParams['ytick.labelsize'] = 16 
        rcParams['axes.labelsize'] = 16
        rcParams['lines.color'] = 'white'
        rcParams['patch.edgecolor'] = 'white'
        rcParams['text.color'] = 'white'
        rcParams['axes.facecolor'] = 'black'
        rcParams['axes.edgecolor'] = 'white'
        rcParams['axes.labelcolor'] = 'white'
        rcParams['xtick.color'] = 'white'
        rcParams['ytick.color'] = 'white'
        rcParams['grid.color'] = 'white'
        rcParams['figure.facecolor'] = 'black'
        rcParams['figure.edgecolor'] = 'black'
        rcParams['savefig.facecolor'] = 'black'
        rcParams['savefig.edgecolor'] = 'none'
        
        res = func(*arg)
        
        for k in rcParamsBackup:
            try:
                rcParams[k] = rcParamsBackup[k]
            except:
                pass
        
        
        return res

    return wrapper


from matplotlib import rcParams
import matplotlib
def setupt_matplot_lib_rc():
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.size'] = 20
    rcParams['font.sans-serif'] = ['Arial']
    #matplotlib.rc('xtick', labelsize=14) 
    #matplotlib.rc('ytick', labelsize=14) 
    #matplotlib.rc('axes', labelsize=14) 
    matplotlib.rc('legend',**{'fontsize':18})
    rcParams['axes.linewidth'] = rcParams['axes.linewidth']*2
    rcParams['legend.numpoints'] = 1
    #rcParams['legend.markerscale'] = 0
    rcParams['xtick.major.width'] = 2
    rcParams['ytick.major.width'] = 2
    rcParams['pdf.fonttype'] = 42
    rcParams['xtick.major.pad']= 8
    rcParams['ytick.major.pad']= 8


def treatmentStackedBar(ax, treatment_dict, color_dict, label_list,top=None):
    width=0.5
        
    labels = []
    rects = []
    x = 0 
    
    cluster_k = max(map(numpy.max,treatment_dict.values()))
    
    sorted_treatment_keys = sorted(treatment_dict, key=lambda tt: numpy.count_nonzero(numpy.array(treatment_dict.get(tt)) == 0) / float(len(treatment_dict.get(tt))))
    for treatment in sorted_treatment_keys:
        cluster_vec = treatment_dict[treatment]
        
        hs = []
        for cluster in range(cluster_k+1):
            h = len((cluster_vec==cluster).nonzero()[0])
            hs.append(float(h) / len(cluster_vec))
            
        if top is None or hs[0] < top:
            labels.append(treatment.split(" - ")[0])
            bottom=0
            for c, h in enumerate(hs):
                rect = ax.bar(x, h, width, bottom=bottom, color=color_dict[c], edgecolor = "none")
                bottom+=h
                rects.append(rect)
            x +=1  
          
    rects.append(rect)
    lg = ax.legend(rects, label_list, 
                   loc=1, 
                   ncol=4,
                   #bbox_to_anchor=(-0.1, -0.4)
                   )
    lg.draw_frame(False)
            
    ax.set_xticks(numpy.arange(len(treatment_dict))+width/2.0)
    ax.set_xticklabels(labels, rotation=90)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlim(-0.2,len(labels)-0.35)
    ax.set_ylim(0,1.2)
    
    #pylab.xlabel('Treatment')
    pylab.ylabel('Cluster (relative frequency)')
    pylab.tight_layout()