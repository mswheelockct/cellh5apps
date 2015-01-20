import numpy
import cellh5

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # specify the location of your ch5 files per plate, the plate name, as stored in your "1.ch4" == "New Folder With Items"
    cellh5_files = {"New Folder With Items": "1.ch5"}
    # to provide some more information about the positions in you plate, we provide a mapping (txt) file (see example)
    mapping_files = {"New Folder With Items": "New Folder With Items.txt"}
    
    # this constructs your class as defined above, with a name == "my_first_test" and the ch5 and mapping files per plate
    ma = cellh5.CH5FateAnalysis("output", cellh5_files=cellh5_files, mapping_files=mapping_files)
    
    # due to copying / preprocessing the time information in your files is not correct, so I set manually a time lapse for this plate of 3 min
    ma.time_lapse["New Folder With Items"] = 3
    
    # now, we read all events in from the ch5 file. Remember you specified a pre-duration for events in CecogAnalyzer
    # we also want to consider events, which occured before frame 100 -> before 300 min
    ma.read_events(onset_frame=5)
    
    # these events are still fixed window sized events (as found by CecogAnalyzer), but we can start from here and extend the tracking
    ma.track_events()
    
    
    # set transmat
    transmat = numpy.array([
                            [100,  0,   1], # "Inter" (inter can go back to mito)
                            [  0,  1,   0], # "Death" (dead is dead)
                            [  1,  0, 100], # "Mito"  (mito can go back to inter)
                          ])
    transmat = transmat.astype(numpy.float64)
    ma.setup_hmm(transmat, "hmm_config_3c.xml")
    
    # predict with hmm
    ma.predict_hmm()
    
    # get some overview as output as txt
    print "Overview:"
    print ma
    
    # get uncorrected tracks as txt
    print "Uncorrected tracks"
    ma.print_tracks("Track Labels")
    
    # get corrected tracks as txt
    print "Corrected tracks:"
    ma.print_tracks("HMM Track Labels")
    
    # show a specific track visually
    ma.report_to_csv()
    
        
        

