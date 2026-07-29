# 128 datasets
UNIVARIATE_DATASET_NAMES_2018 = ['ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ',
                                 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'BME', 'Car', 'CBF', 'Chinatown',
                                 'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers', 'CricketX',
                                 'CricketY', 'CricketZ', 'Crop', 'DiatomSizeReduction',
                                 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
                                 'DodgerLoopDay', 'DodgerLoopGame', 'DodgerLoopWeekend', 'Earthquakes', 'ECG200',
                                 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'EOGHorizontalSignal',
                                 'EOGVerticalSignal', 'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR',
                                 'FiftyWords', 'Fish', 'FordA', 'FordB', 'FreezerRegularTrain',
                                 'FreezerSmallTrain', 'Fungi', 'GestureMidAirD1', 'GestureMidAirD2',
                                 'GestureMidAirD3', 'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPoint',
                                 'GunPointAgeSpan', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung',
                                 'Ham', 'HandOutlines', 'Haptics', 'Herring', 'HouseTwenty', 'InlineSkate',
                                 'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'InsectWingbeatSound',
                                 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7',
                                 'Mallat', 'Meat', 'MedicalImages', 'MelbournePedestrian',
                                 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
                                 'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain',
                                 'MoteStrain', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2',
                                 'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme',
                                 'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP',
                                 'PLAID', 'Plane', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup',
                                 'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices',
                                 'Rock', 'ScreenType', 'SemgHandGenderCh2', 'SemgHandMovementCh2',
                                 'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ', 'ShapeletSim', 'ShapesAll',
                                 'SmallKitchenAppliances', 'SmoothSubspace', 'SonyAIBORobotSurface1',
                                 'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry', 'SwedishLeaf',
                                 'Symbols', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace',
                                 'TwoLeadECG', 'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll',
                                 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ',
                                 'Wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga']

# Missing values and unequal length datasets have a "corrected" version located in subfolder of UCR archive.
# Subfolder name is "Missing_value_and_variable_length_datasets_adjusted".
UNIVARIATE_DATASET_NAMES_2018_MISSING_VALUES = ['DodgerLoopDay', 'DodgerLoopGame', 'DodgerLoopWeekend',
                                                'MelbournePedestrian', 'AllGestureWiimoteX', 'AllGestureWiimoteY',
                                                'AllGestureWiimoteZ', 'GestureMidAirD1', 'GestureMidAirD2',
                                                'GestureMidAirD3', 'GesturePebbleZ1', 'GesturePebbleZ2',
                                                'PickupGestureWiimoteZ', 'PLAID', 'ShakeGestureWiimoteZ']

# 30 datasets (may have unequal length or missing values)
MULTIVARIATE_DATASET_NAMES_2018_ALL = ['ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions',
                                       'CharacterTrajectories', 'Cricket', 'DuckDuckGeese', 'EigenWorms',
                                       'Epilepsy', 'EthanolConcentration', 'ERing', 'FaceDetection',
                                       'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat',
                                       'InsectWingbeat', 'JapaneseVowels', 'Libras', 'LSST', 'MotorImagery', 'NATOPS',
                                       'PenDigits', 'PEMS-SF', 'PhonemeSpectra', 'RacketSports', 'SelfRegulationSCP1',
                                       'SelfRegulationSCP2', 'SpokenArabicDigits', 'StandWalkJump',
                                       'UWaveGestureLibrary']

# 26 datasets with equal length
MULTIVARIATE_DATASET_NAMES_2018 = ["ArticularyWordRecognition", "AtrialFibrillation", "BasicMotions", "Cricket",
                                   "DuckDuckGeese", "EigenWorms", "Epilepsy", "EthanolConcentration", "ERing",
                                   "FaceDetection", "FingerMovements", "HandMovementDirection", "Handwriting",
                                   "Heartbeat", "Libras", "LSST", "MotorImagery", "NATOPS", "PenDigits", "PEMS-SF",
                                   "PhonemeSpectra", "RacketSports", "SelfRegulationSCP1", "SelfRegulationSCP2",
                                   "StandWalkJump", "UWaveGestureLibrary"]

# 29 datasets
MONSTER_DATASET_NAMES_2025_ALL = ['AudioMNIST', 'AudioMNIST-DS', 'CornellWhaleChallenge', 'CrowdSourced',
                                  'DREAMERA', 'DREAMERV', 'FordChallenge', 'FruitFlies', 'InsectSound', 'LenDB',
                                  'MosquitoSound', 'Opportunity', 'PAMAP2', 'Pedestrian', 'LakeIce',
                                  'S2Agri-10pc-17', 'S2Agri-10pc-34', 'S2Agri-17', 'S2Agri-34', 'STEW',
                                  'Skoda', 'Tiselac', 'Traffic', 'UCIActivity', 'USCActivity', 'WISDM2', 'WISDM',
                                  'WhaleSounds', 'TimeSen2Crop']

# 27 datasets removing larger S2Agri
MONSTER_DATASET_NAMES_2025 = ['AudioMNIST', 'AudioMNIST-DS', 'CornellWhaleChallenge', 'CrowdSourced',
                              'DREAMERA', 'DREAMERV', 'FordChallenge', 'FruitFlies', 'InsectSound', 'LenDB',
                              'MosquitoSound', 'Opportunity', 'PAMAP2', 'Pedestrian', 'LakeIce',
                              'S2Agri-10pc-17', 'S2Agri-10pc-34', 'STEW',
                              'Skoda', 'Tiselac', 'Traffic', 'UCIActivity', 'USCActivity', 'WISDM2', 'WISDM',
                              'WhaleSounds', 'TimeSen2Crop']

# 16 < 60s / epoch
MONSTER_DATASET_NAMES_2025_FAST = ['CrowdSourced', 'WISDM', 'Opportunity', 'UCIActivity', 'Skoda', 'STEW',
                                   'FordChallenge', 'PAMAP2', 'USCActivity', 'InsectSound', 'Tiselac', 'WISDM2',
                                   'LakeIce', 'Pedestrian', 'DREAMERA', 'DREAMERV']
# 11 > 60s / epoch
MONSTER_DATASET_NAMES_2025_SLOW = ['CornellWhaleChallenge', 'AudioMNIST-DS', 'FruitFlies', 'WhaleSounds',
                                   'Traffic', 'TimeSen2Crop', 'LenDB', 'MosquitoSound', 'AudioMNIST',
                                   'S2Agri-10pc-34', 'S2Agri-10pc-17']
