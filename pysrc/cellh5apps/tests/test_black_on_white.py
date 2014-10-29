from cellh5apps.utils.plots import matplotlib_black_background

if __name__ == "__main__":
    import pylab
    
    # white background
    pylab.plot(range(10), label="line")
    pylab.title('test white')
    pylab.show()
    
    # black background
    matplotlib_black_background()
    pylab.plot(range(10), label="line")
    pylab.title('test black')
    pylab.legend()
    pylab.show()