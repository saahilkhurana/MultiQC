import pandas as pd
import numpy as np
np.set_printoptions(precision=4)

class SeqModel:
    def __init__(self):
        self.obs3_seqMat = None
        self.obs5_seqMat = None
        self.exp3_seqMat = None
        self.exp5_seqMat = None
        self.valid_ = False

    def populate_model_(self, data_) :
        """read 3 int
            3 int arrays
            long nrow, ncol
            double matrix (nrow x ncol)
            4
                int rows, cols
                double arrays

        """
        import struct
        from numpy.linalg import norm

        weights = None
        model = None
        offset = 0
        int_struct = struct.Struct('@i')
        long_struct = struct.Struct('@q')
        double_struct = struct.Struct('@d')

        x1 = int_struct.unpack_from(data_[offset:])[0]
        offset += int_struct.size
        x2 = int_struct.unpack_from(data_[offset:])[0]
        offset += int_struct.size
        x3 = int_struct.unpack_from(data_[offset:])[0]
        offset += int_struct.size
        # print(x1,x2,x3)
        w1_struct = struct.Struct('@' + x1 * 'i')
        w1 = w1_struct.unpack_from(data_[offset:])
        offset += w1_struct.size
        w2_struct = struct.Struct('@' + x1 * 'i')
        w2 = w2_struct.unpack_from(data_[offset:])
        offset += w2_struct.size
        w3_struct = struct.Struct('@' + x1 * 'i')
        w3 = w3_struct.unpack_from(data_[offset:])
        offset += w3_struct.size
        # print(w1,w2,w3)
        r1 = long_struct.unpack_from(data_[offset:])[0]
        offset += long_struct.size
        c1 = long_struct.unpack_from(data_[offset:])[0]
        offset += long_struct.size
        # print(r1,c1)
        model_struct = struct.Struct('@' + r1 * c1 * 'd')
        model = model_struct.unpack_from(data_[offset:])
        offset += model_struct.size
        model = np.array(model)
        # print(model)
        row = int_struct.unpack_from(data_[offset:])[0]
        offset += int_struct.size
        col = int_struct.unpack_from(data_[offset:])[0]
        offset += int_struct.size
        # print(row, col)
        val = int_struct.unpack_from(data_[offset:])[0]
        offset += int_struct.size
        val1 = int_struct.unpack_from(data_[offset:])[0]
        offset += int_struct.size
        # print(val, val1)
        vals = struct.Struct('@' + 9 * 'd')
        m1 = vals.unpack_from(data_[offset:])
        offset += vals.size
        vals = struct.Struct('@' + 9 * 'd')
        m2 = vals.unpack_from(data_[offset:])
        offset += vals.size
        vals = struct.Struct('@' + 9 * 'd')
        m3 = vals.unpack_from(data_[offset:])
        offset += vals.size
        vals = struct.Struct('@' + 9 * 'd')
        m4 = vals.unpack_from(data_[offset:])
        offset += vals.size
        # print(m1,m2,m3,m4)
        ret = []
        ret.append(np.array(m1))
        ret.append(np.array(m2))
        ret.append(np.array(m3))
        ret.append(np.array(m4))

        return ret

    def from_file(self, dname):
        import os
        import gzip
        obs3_name = os.path.sep.join([dname, 'aux_info', 'obs3_seq.gz'])
        obs5_name = os.path.sep.join([dname, 'aux_info', 'obs5_seq.gz'])
        exp3_name = os.path.sep.join([dname, 'aux_info', 'exp3_seq.gz'])
        exp5_name = os.path.sep.join([dname, 'aux_info', 'exp5_seq.gz'])

        obs3_dat = None
        obs5_dat = None
        exp3_dat = None
        exp5_dat = None
        try:
            with gzip.open(obs3_name) as obs3_file:
                obs3_dat = obs3_file.read()
            self.obs3_seqMat = self.populate_model_(obs3_dat)
        except IOError:
            print("Could not open file {}".format(obs3_name))
            return False

        try:
            with gzip.open(obs5_name) as obs5_file:
                obs5_dat = obs5_file.read()
            self.obs5_seqMat = self.populate_model_(obs5_dat)
        except IOError:
            print("Could not open file {}".format(obs5_name))
            return False

        try:
            with gzip.open(exp3_name) as exp3_file:
                exp3_dat = exp3_file.read()
            self.exp3_seqMat = self.populate_model_(exp3_dat)
        except IOError:
            print("Could not open file {}".format(exp3_name))
            return False

        try:
            with gzip.open(exp5_name) as exp5_file:
                exp5_dat = exp5_file.read()
            self.exp5_seqMat = self.populate_model_(exp5_dat)
        except IOError:
            print("Could not open file {}".format(exp5_name))
            return False

        # print(self.obs3_seqMat)
        # print(self.obs5_seqMat)
        # print(self.exp3_seqMat)
        # print(self.exp5_seqMat)

        self.valid_ = True
        return True


class GCModel:
    def __init__(self):
        self.obs_weights_ = None
        self.exp_weights_ = None
        self.obs_ = None
        self.exp_ = None
        self.dims_ = None
        self.valid_ = False

    def populate_model_(self, data_):
        import struct
        from numpy.linalg import norm

        weights = None
        model = None
        offset = 0
        int_struct = struct.Struct('@i')
        long_struct = struct.Struct('@q')

        mspace = int_struct.unpack_from(data_[offset:])[0]
        offset += int_struct.size

        nrow = long_struct.unpack_from(data_[offset:])[0]
        offset += long_struct.size

        ncol = long_struct.unpack_from(data_[offset:])[0]
        offset += long_struct.size
        # print(nrow,ncol)

        weight_struct = struct.Struct('@' + nrow * 'd')
        weights = weight_struct.unpack_from(data_[offset:])
        offset += weight_struct.size

        model_struct = struct.Struct('@' + nrow * ncol * 'd')
        model = model_struct.unpack_from(data_[offset:])
        model = np.array(model)
        model = model.reshape(ncol, nrow).T
        model = (model.T / model.sum(axis=1)).T
        return weights, model

    def from_file(self, dname):
        import os
        import gzip
        obs_name = os.path.sep.join([dname, 'aux_info', 'obs_gc.gz'])
        exp_name = os.path.sep.join([dname, 'aux_info', 'exp_gc.gz'])

        obs_dat, exp_dat = None, None
        try:
            with gzip.open(obs_name) as obs_file:
                obs_dat = obs_file.read()
            self.obs_weights_, self.obs_ = self.populate_model_(obs_dat)
        except IOError:
            print("Could not open file {}".format(obs_name))
            return False

        try:
            with gzip.open(exp_name) as exp_file:
                exp_dat = exp_file.read()
            self.exp_weights_, self.exp_ = self.populate_model_(exp_dat)
        except IOError:
            print("Could not open file {}".format(exp_name))
            return False

        self.valid_ = True
        return True

if __name__ == '__main__':
    s = SeqModel()
    s.from_file('/Users/alok/Downloads/project4/project4/ERR188041/bias/')
    # g = GCModel()
    # g.from_file('/Users/alok/Downloads/project4/project4/ERR188041/bias/')

def readThreeColumnTruth(fn, suffix=""):
    df = pd.read_csv(fn, sep=' ', skiprows=1,
                     names=['Name', 'Gene{}'.format(suffix),
                            'TPM{}'.format(suffix)], engine='c')
    df.set_index("Name", inplace=True)
    pd.to_numeric(df["TPM{}".format(suffix)], errors='ignore')
    return df

def readRSEMTruth(fn, suffix=""):
    df = pd.read_csv(fn, sep='\t', skiprows=1,
                     names=['Name', 'Gene{}'.format(suffix),
                            'Length{}'.format(suffix),
                            'EffectiveLength{}'.format(suffix),
                            'NumReads{}'.format(suffix),
                            'TPM{}'.format(suffix),
                            'FPKM{}'.format(suffix),
                            'IsoPct{}'.format(suffix)], engine='c').set_index('Name')
    for col in ["TPM", "Length", "EffectiveLength", "NumReads"]:
        pd.to_numeric(df["{}{}".format(col, suffix)], errors='ignore')
    return df

def readRSEM(fn, suffix=""):
    df = pd.read_csv(fn, sep='\t', skiprows=1,
                     names=['Name', 'Gene{}'.format(suffix),
                            'Length{}'.format(suffix),
                            'EffectiveLength{}'.format(suffix),
                            'NumReads{}'.format(suffix),
                            'TPM{}'.format(suffix),
                            'FPKM{}'.format(suffix),
                            'IsoPct{}'.format(suffix)], engine='c').set_index('Name')
    for col in ["TPM", "Length", "EffectiveLength", "NumReads"]:
        pd.to_numeric(df["{}{}".format(col, suffix)], errors='ignore')
    return df

def readStringTie(fn, suffix=""):
    """
    Not yet tested
    """
    df = pd.read_csv(fn, sep="\t", skiprows=1,
                     names=["tid{}".format(suffix),
                            "chr{}".format(suffix),
                            "strand{}".format(suffix),
                            "start{}".format(suffix),
                            "end{}".format(suffix),
                            "Name",
                            "num_exons{}".format(suffix),
                            "Length{}".format(suffix),
                            "gene_id{}".format(suffix),
                            "gene_name{}".format(suffix),
                            "cov{}".format(suffix),
                            "FPKM{}".format(suffix)])
    df.set_index('Name', inplace=True)
    pd.to_numeric(df)
    return df

def readExpress(fn, suffix=""):
    df = pd.read_csv(fn, sep="\t", skiprows=1,
                     names=["bundle_id{}".format(suffix),
                            "Name",
                            "Length{}".format(suffix),
                            "EffectiveLength{}".format(suffix),
                            "tot_counts{}".format(suffix),
                            "uniq_counts{}".format(suffix),
                            "NumReads{}".format(suffix),
                            "NumReadsEffective{}".format(suffix),
                            "ambig_distr_alpha{}".format(suffix),
                            "ambig_distr_beta{}".format(suffix),
                            "fpkm{}".format(suffix),
                            "fpkm_conf_low{}".format(suffix),
                            "fpkm_conf_high{}".format(suffix),
                            "solvable{}".format(suffix),
                            "TPM{}".format(suffix)]).set_index('Name')
    for col in ["TPM", "Length", "EffectiveLength", "NumReads"]:
        pd.to_numeric(df["{}{}".format(col, suffix)], errors='ignore')
    return df

def readKallistoBoot(fn, suffix=""):
    import h5py
    import os
    import numpy as np
    h5file = os.path.sep.join([fn,"abundance.h5"])
    f = h5py.File(h5file)
    names = map(str, f['aux']['ids'].value)
    nboot = len(f['bootstrap'])
    boots = []
    for i in xrange(nboot):
        boots.append(f['bootstrap']['bs{}'.format(i)].value)
    y = pd.DataFrame(data=boots, dtype=np.float64).T
    y = y.assign(Name=names).set_index('Name').sort_index()
    y = y.apply(np.sort, axis=1)
    return y

def readSalmonBoot(fn, suffix=""):
    import os
    import gzip
    import pandas as pd
    import numpy as np
    import struct
    import json
    auxDir = "aux"
    # Check for a custom auxDir
    with open(os.path.sep.join([fn, "cmd_info.json"])) as cmdFile:
        dat = json.load(cmdFile)
        if 'auxDir' in dat:
            auxDir = dat['auxDir']

    bootstrapFile = os.path.sep.join([fn, auxDir, "bootstrap", "bootstraps.gz"])
    nameFile = os.path.sep.join([fn, auxDir, "bootstrap", "names.tsv.gz"])
    bootstrapFile = os.path.sep.join([fn, auxDir, "bootstrap", "bootstraps.gz"])
    nameFile = os.path.sep.join([fn, auxDir, "bootstrap", "names.tsv.gz"])
    if not os.path.isfile(bootstrapFile):
       print("The required bootstrap file {} doesn't appear to exist".format(bootstrapFile))
       sys.exit(1)
    if not os.path.isfile(nameFile):
       print("The required transcript name file {} doesn't appear to exist".format(nameFile))
       sys.exit(1)

    txpNames = None
    with gzip.open(nameFile) as nf:
        txpNames = nf.read().strip().split('\t')

    ntxp = len(txpNames)
    print("Expecting bootstrap info for {} transcripts".format(ntxp))

    with open(os.path.sep.join([fn, auxDir, "meta_info.json"])) as fh:
        meta_info = json.load(fh)

    stype = None
    if meta_info['samp_type'] == 'gibbs':
        s = struct.Struct('@' + 'd' * ntxp)
        stype = 'g'
    elif meta_info['samp_type'] == 'bootstrap':
        s = struct.Struct('@' + 'd' * ntxp)
        stype = 'b'
    else:
        print("Unknown sampling method: {}".format(meta_info['samp_type']))
        sys.exit(1)

    numBoot = 0
    samps = []
    convert = float
    # Now, iterate over the bootstrap samples and write each
    with gzip.open(bootstrapFile) as bf:
        while True:
            try:
                x = s.unpack_from(bf.read(s.size))
                xs = map(convert, x)
                samps.append(xs)
                numBoot += 1
            except:
                print("read all posterior values")
                break

    print("wrote {} bootstrap samples".format(numBoot))
    print("converted bootstraps successfully.")
    y = pd.DataFrame(data=samps, dtype=np.float64).T
    y = y.assign(Name=txpNames).set_index('Name').sort_index()
    y = y.apply(np.sort, axis=1)
    return y

def readSailfish(fn, suffix=""):
    df = pd.read_table(fn, engine='c').set_index('Name')
    df.columns = [ "{}{}".format(cn, suffix) for cn in df.columns.tolist()]
    for col in ["TPM", "Length", "EffectiveLength", "NumReads"]:
        pd.to_numeric(df["{}{}".format(col, suffix)], errors='ignore')
    return df

def readSalmon(fn, suffix=""):
    return readSailfish(fn, suffix)

def readSailfishDeprecated(fn, suffix=""):
    df = pd.read_csv(fn, sep='\t', comment='#',
                     names=['Name',
                            'Length{}'.format(suffix),
                            'TPM{}'.format(suffix),
                            'RPKM{}'.format(suffix),
                            'KPKM{}'.format(suffix),
                            'NumKmers{}'.format(suffix),
                            'NumReads{}'.format(suffix)])
    df.dropna(how='all', inplace=True)
    df.convert_objects(convert_numeric=True)
    df.set_index('Name', inplace=True)
    return df

def readKallisto(fn, suffix=""):
    df = pd.read_csv(fn, sep='\t', skiprows=1,
                     names=['Name',
                            'Length{}'.format(suffix),
                            'EffectiveLength{}'.format(suffix),
                            'NumReads{}'.format(suffix),
                            'TPM{}'.format(suffix)], engine='c').set_index('Name')
    for col in ["TPM", "Length", "EffectiveLength", "NumReads"]:
        pd.to_numeric(df["{}{}".format(col, suffix)], errors='ignore')
    return df

def readProFile(fn, suffix=""):
    df = pd.read_csv(fn, sep='\t',
                     names=['Locus{}'.format(suffix),
                            'Name',
                            'Coding{}'.format(suffix),
                            'Length{}'.format(suffix),
                            'ExpFrac{}'.format(suffix),
                            'ExpNum{}'.format(suffix),
                            'LibFrac{}'.format(suffix),
                            'LibNum{}'.format(suffix),
                            'SeqFrac{}'.format(suffix),
                            'SeqNum{}'.format(suffix),
                            'CovFrac{}'.format(suffix),
                            'ChiSquare{}'.format(suffix),
                            'CV{}'.format(suffix)]).set_index('Name')
    for col in ["Length", "ExpFrac", "ExpNum", "LibFrac", "LibNum", "SeqFrac", "SeqNum", "CovFrac", "ChiSquare", "CV"]:
        pd.to_numeric(df["{}{}".format(col, suffix)], errors='ignore')
    return df

def readResFile(fn, suffix=""):
    df = pd.read_csv(fn, sep='\t',
                     names=['Name',
                            'Length{}'.format(suffix),
                            'Abund{}'.format(suffix)]).set_index('Name')
    for col in ["Length", "Abund"]:
        pd.to_numeric(df["{}{}".format(col, suffix)], errors='ignore')
    return df
