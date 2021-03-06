import numpy as np
from utils.library_design import LibraryDesign

experiment = 'LazDEF_model_test'

class constants:
    '''
    Star symbol (*) is reserved for stop codons.
    Plus and underscore symbols (+ and _) are internally reserved tokens.
    Numerals (1234567890) are internally reserved for library design specification. 
    These symbols (123456790+_) should not be used to encode amino acids. 
    Other symbols are OK.
    
    Although the class holds multiple attributes, only
    the codon table should be edited. Everything else
    is inferred automatically from it.
    '''
    codon_table = {
                    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
                    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
                    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
                    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
                    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
                    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
                    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
                    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
                    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
                    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
                    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
                    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
                    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
                    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
                    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
                    'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
                   }


    bases = ('T', 'C', 'A', 'G')
    complement_table = str.maketrans('ACTGN', 'TGACN')
    
    #probabilities of individual bases in randomized positions
    base_calls = {
                    'T': (1.00, 0.00, 0.00, 0.00),
                    'C': (0.00, 1.00, 0.00, 0.00),
                    'A': (0.00, 0.00, 1.00, 0.00),
                    'G': (0.00, 0.00, 0.00, 1.00),
                    'K': (0.50, 0.00, 0.00, 0.50),
                    'R': (0.00, 0.00, 0.50, 0.50),
                    'Y': (0.50, 0.50, 0.00, 0.00),
                    'S': (0.00, 0.50, 0.00, 0.50),
                    'W': (0.50, 0.00, 0.50, 0.00),
                    'M': (0.00, 0.50, 0.50, 0.00),
                    'D': (0.34, 0.00, 0.33, 0.33),
                    'H': (0.34, 0.33, 0.33, 0.00),
                    'V': (0.00, 0.34, 0.33, 0.33),                                
                    'B': (0.34, 0.33, 0.00, 0.33),
                    'N': (0.25, 0.25, 0.25, 0.25)
                    
                 }

    #Cys is represented as its IAA-alkylation reaction product
    aaSMILES = [    'N[C@@H](C)C(=O)',           
                    #'N[C@@H](CSCC(=O)N)C(=O)',  #IAA alkylation form for Cys: uncomment for LazBF
                    'N[C@@H](CS)C(=O)',        #regular Cys: uncomment for LazDEF
                    'N[C@@H](CC(=O)O)C(=O)',
                    'N[C@@H](CCC(=O)O)C(=O)',
                    'N[C@@H](Cc1ccccc1)C(=O)',
                    'NCC(=O)',
                    'N[C@@H](Cc1c[nH]cn1)C(=O)',
                    'N[C@@H]([C@H](CC)C)C(=O)',
                    'N[C@@H](CCCCN)C(=O)',        
                    'N[C@@H](CC(C)C)C(=O)',
                    'N[C@@H](CCSC)C(=O)',
                    'N[C@@H](CC(=O)N)C(=O)',
                    'O=C[C@@H]1CCCN1',  
                    'N[C@@H](CCC(=O)N)C(=O)',
                    'N[C@@H](CCCNC(=N)N)C(=O)',
                    'N[C@@H](CO)C(=O)',
                    'N[C@@H]([C@H](O)C)C(=O)',
                    'N[C@@H](C(C)C)C(=O)',
                    'N[C@@H](Cc1c[nH]c2c1cccc2)C(=O)',
                    'N[C@@H](Cc1ccc(O)cc1)C(=O)'
              ]


    global _reserved_aa_names
    _reserved_aa_names = ('_', '+', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0')    
    
    aas = tuple(sorted(set(x for x in codon_table.values() if x not in _reserved_aa_names)))
    codons = tuple(sorted(set(x for x in codon_table.keys())))
    aa_dict = {aa: i for i,aa in enumerate(aas)}
          
class ParserConfig:

    #a RE pattern that has to match in order to initiate the orf
    #used when force_translation == False
    utr5_seq = 'AGGAGAT......ATG'
        
    #DNA library design
    D_design = LibraryDesign(
        
                    templates=[
                               'ttgccggaaaatggggcg112112112112112112tgc112112112112112112ggaggatacccatacgacgtgcccgactatgcaggggaattagctagaccctaggacggggggcggaaa'.upper(),
                              ],
            
                    monomers={
                              1: ('A', 'G', 'T', 'C'),
                              2: ('G', 'T'),
                             },
                    
                    lib_type='dna'
                        
                           )    
    
    #peptide library design
    P_design = LibraryDesign(
        
                    templates=[
                               'LPENGA111111C111111GGYPYDVPDYAGELARP',
                              ],
            
                    monomers={
                              1: ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'),
                             },
                    
                    lib_type='pep'
                        
                           )
 
class TrackerConfig:
    
    #directory holding sequencing data files (fastq or fastq.gz)
    seq_data = '../sequencing_data'
    
    #directory holding most of the ML-related files
    ml_data = '../ml_data'
    
    #directory for ML models
    model = '../model'
    
    #directory for writing logs to
    logs = '../logs'
    
    #directory that stores fastq parser outputs
    parser_out = '../parser_outputs'
    
    #directory that stores feature matrices
    f_matrix = '../feature_matrices'
    
class LoggerConfig:
    
    #logger name
    name = experiment + '_logger'
    
    #verbose loggers print to the console
    verbose = True
    
    #logger level ('INFO', 'WARNING', 'ERROR')
    level = 'INFO'
    
    #write logs to file
    log_to_file = True
    
    #log filename; when None, the name will be inferred by the Logger itself
    log_fname = os.path.join(TrackerConfig.logs, experiment + ' logs')
    
class PreproConfig:
    
    #featurization matrix
    F = np.load(os.path.join(TrackerConfig.f_matrix, 
                              'DENSE_Morgan_F_r=4_LazDEF.npy'))     
    
class ClassifierConfig:

    #-----------------------------------
    #model architecture
    #-----------------------------------
    from tf.cnn_model import cnn_v5
    model = cnn_v5

    #NN architecture input_dimension
    inp_dim = (12, 208)
    
    #network dropout rate
    drop = 0.20

    #experiment name
    experiment_name = experiment

    #-----------------------------------
    #training metaparameters
    #-----------------------------------
    
    #learning rate
    from tf.schedules import NoamSchedule
    lr = NoamSchedule(inp_dim[-1], warmup_steps=4000)

    #model training optimizer object
    import tensorflow.keras as K
    optimizer = K.optimizers.Adam(
                                  learning_rate=lr,
                                  beta_1=0.9,
                                  beta_2=0.98, 
                                  epsilon=1e-9
                                 )
        
    #training loss function
    loss = 'binary_crossentropy'
    
    #training metrics
    metrics = ['accuracy']
    
    #max number of epochs; usually smaller if early_stopping callback is set
    epochs = 2
    
    #fitting process verbosity
    verbosity = 1       
    
    #training batch size
    batch_size = 2048
    
    #tf.Dataset param 
    shuffle_buffer = 10        
    
    

    
    
    
    
    
    
    