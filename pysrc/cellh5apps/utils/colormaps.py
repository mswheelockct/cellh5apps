import matplotlib
from matplotlib.colors import hex2color
YlBlCMap = matplotlib.colors.LinearSegmentedColormap.from_list('asdf', [(0,0,1), (1,1,0)])

def hex2rgb(color, mpl=False):
    """Return the rgb color as python int in the range 0-255."""
    assert color.startswith("#")
    if mpl:
        fac = 1.0
    else:
        fac = 255.0
    rgb = [int(i*fac) for i in hex2color(color)]
    return tuple(rgb)

def QtColorMapFromHex(hex_str):
    from PyQt4 import QtGui
    import numpy
    
    color = hex2rgb(hex_str)
    lut = numpy.zeros((256,4))
    
    for c in range(3):
        lut[:,c] = numpy.array(range(256)) / 255. * color[c]
    lut[:,3] = 255
    lut[0,3] = 0
    return [QtGui.qRgba(r,g,b,a) for r,g,b,a in lut]
