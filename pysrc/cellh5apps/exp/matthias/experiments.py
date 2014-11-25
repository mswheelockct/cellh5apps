EXP = {'matthias_predrug_a6':
        {
        'mapping_files' : {
            'Screen_Plate_01': 'F:/matthias_predrug_a6/Screen_Plate_01_position_map_PRE.txt',
            'Screen_Plate_02': 'F:/matthias_predrug_a6/Screen_Plate_02_position_map_PRE.txt',
            'Screen_Plate_03': 'F:/matthias_predrug_a6/Screen_Plate_03_position_map_PRE.txt',
            'Screen_Plate_04': 'F:/matthias_predrug_a6/Screen_Plate_04_position_map_PRE.txt',
            'Screen_Plate_05': 'F:/matthias_predrug_a6/Screen_Plate_05_position_map_PRE.txt',
            'Screen_Plate_06': 'F:/matthias_predrug_a6/Screen_Plate_06_position_map_PRE.txt',
            'Screen_Plate_07': 'F:/matthias_predrug_a6/Screen_Plate_07_position_map_PRE.txt',
            'Screen_Plate_08': 'F:/matthias_predrug_a6/Screen_Plate_08_position_map_PRE.txt',
            'Screen_Plate_09': 'F:/matthias_predrug_a6/Screen_Plate_09_position_map_PRE.txt',
            },
        'ch5_files' : {
            'Screen_Plate_01': 'F:/matthias_predrug_a6/Screen_Plate_01_all_positions_with_data.ch5',
            'Screen_Plate_02': 'F:/matthias_predrug_a6/Screen_Plate_02_all_positions_with_data.ch5',
            'Screen_Plate_03': 'F:/matthias_predrug_a6/Screen_Plate_03_all_positions_with_data.ch5',
            'Screen_Plate_04': 'F:/matthias_predrug_a6/Screen_Plate_04_all_positions_with_data.ch5',
            'Screen_Plate_05': 'F:/matthias_predrug_a6/Screen_Plate_05_all_positions_with_data.ch5',
            'Screen_Plate_06': 'F:/matthias_predrug_a6/Screen_Plate_06_all_positions_with_data.ch5',
            'Screen_Plate_07': 'F:/matthias_predrug_a6/Screen_Plate_07_all_positions_with_data.ch5',
            'Screen_Plate_08': 'F:/matthias_predrug_a6/Screen_Plate_08_all_positions_with_data.ch5',
            'Screen_Plate_09': 'F:/matthias_predrug_a6/Screen_Plate_09_all_positions_with_data.ch5',
            },
#         'locations' : (
#             ("A", 4), ("B", 23), ("H", 9), ("D", 8),
#             ("H", 6), ("A", 7), ("G", 6), ("G", 7),
#             ("H", 12), ("H", 13), ("G", 12), ("A", 9),
#             ),
#         'rows' : list("A")[:],
#         'cols' : tuple(range(4, 5)),
        'training_sites' : (1, 2, 3, 4),
        },
       
       
       'matthias_predrug_a6_plate_9':
        {
        'mapping_files' : {
            'Screen_Plate_09': 'F:/matthias_predrug_a6/Screen_Plate_09_position_map_PRE.txt',
            },
        'ch5_files' : {

            'Screen_Plate_09': 'F:/matthias_predrug_a6/Screen_Plate_09_all_positions_with_data.ch5',
            },
        'training_sites' : (1, 2, 3, 4),
        },
       
      'matthias_test_plate':
        {
         'mapping_files' : {
            '002324': 'M:/experiments/Experiments_002300/002324/meta/CellCog/mapping/MD9_Grape_over_Time.txt',
        },
        'cellh5_files' : {
           '002324': 'M:/experiments/Experiments_002300/002324/meta/CellCog/analysis_outlier_3/hdf5/_all_positions.ch5',
        },
        'locations' : (
            ("A", 8), ("B", 8), ("C", 8), ("D", 8),
            ("H", 6), ("H", 7), ("G", 6), ("G", 7),
            ("H", 12), ("H", 13), ("G", 12), ("G", 13),
            ("A", 8), ("B", 8), ("C", 8), ("D", 8), ("E", 8),
            #                                 ("D",  13), ("F",  13), ("H",  13), # Taxol No Rev
            ("D", 7), ("F", 7), ("H", 7),  # Noco No Rev 
            ("D", 12), ("F", 12), ("H", 12),  # Taxol 300 Rev
            ("D", 6), ("F", 6), ("H", 6),  # Noco 300 Rev
            #                                 ("D",  9), ("F",  9), ("H",  9), # Taxol 900 Rev
            #                                 ("D",  3), ("F",  3), ("H",  3), # Noco 900 Rev
            #                                 
            #                                 ("J",  13), ("L",  13), ("N",  13), # Taxol No Rev
            ("J", 7), ("L", 7), ("N", 7),  # Noco No Rev 
            ("J", 12), ("L", 12), ("N", 12),  # Taxol 300 Rev
            ("J", 6), ("L", 6), ("N", 6),  # Noco 300 Rev
            #                                 ("J",  9), ("L",  9), ("N",  9), # Taxol 900 Rev
            #                                 ("J",  3), ("L",  3), ("N",  3), # Noco 900 Rev
        
        
        #                             ("B",  19), ("C",  19), ("D",  19), ("E",  19), # NEG
        #                             ("D",  24), ("F",  24), ("H",  24), # Taxol No Rev
        # ("D",  18), ("F",  18), ("H",  18), # Noco No Rev 
        # ("D",  23), ("F",  23), ("H",  23), # Taxol 300 Rev
        # ("D",  17), ("F",  17), ("H",  17), # Noco 300 Rev
        #                             ("D",  20), ("F",  20), ("H",  20), # Taxol 900 Rev
        #                             ("D",  14), ("F",  14), ("H",  14), # Noco 900 Rev
        ),
        }
   }
       
