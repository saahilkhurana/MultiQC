import os
import gzip
import struct
import numpy as np
from numpy.linalg import norm
np.set_printoptions(precision=4)

class SeqModel:
    def __init__(self):
        self.obs3_seqMat = None
        self.obs5_seqMat = None
        self.exp3_seqMat = None
        self.exp5_seqMat = None
        self.valid_ = False

    def populate_model_(self, data_) :
        """ read 3 ints
            3 int arrays
            long nrow, ncol
            double matrix (nrow x ncol)
            int rows, cols
            double arrays of size 4*context_len

        """

        weights = None
        model = None
        offset = 0
        int_struct = struct.Struct('@i')
        long_struct = struct.Struct('@q')
        double_struct = struct.Struct('@d')

        context_len = int_struct.unpack_from(data_[offset:])[0]
        offset += int_struct.size
        l_context_len = int_struct.unpack_from(data_[offset:])[0]
        offset += int_struct.size
        r_context_len = int_struct.unpack_from(data_[offset:])[0]
        offset += int_struct.size
        # print(context_len,l_context_len,r_context_len)
        vlmm_struct = struct.Struct('@' + context_len * 'i')
        vlmm = vlmm_struct.unpack_from(data_[offset:])
        offset += vlmm_struct.size
        shifts_struct = struct.Struct('@' + context_len * 'i')
        shifts = shifts_struct.unpack_from(data_[offset:])
        offset += shifts_struct.size
        widths_struct = struct.Struct('@' + context_len * 'i')
        widths = widths_struct.unpack_from(data_[offset:])
        offset += widths_struct.size

        nrow = long_struct.unpack_from(data_[offset:])[0]
        offset += long_struct.size
        ncol = long_struct.unpack_from(data_[offset:])[0]
        offset += long_struct.size

        model_struct = struct.Struct('@' + nrow * ncol * 'd')
        model = model_struct.unpack_from(data_[offset:])
        offset += model_struct.size
        model = np.array(model)

        row = int_struct.unpack_from(data_[offset:])[0]
        offset += int_struct.size
        col = int_struct.unpack_from(data_[offset:])[0]
        offset += int_struct.size

        val = int_struct.unpack_from(data_[offset:])[0]
        offset += int_struct.size
        val1 = int_struct.unpack_from(data_[offset:])[0]
        offset += int_struct.size

        vals = struct.Struct('@' + 9 * 'd')
        marginal_prob1 = vals.unpack_from(data_[offset:])
        offset += vals.size
        vals = struct.Struct('@' + 9 * 'd')
        marginal_prob2 = vals.unpack_from(data_[offset:])
        offset += vals.size
        vals = struct.Struct('@' + 9 * 'd')
        marginal_prob3 = vals.unpack_from(data_[offset:])
        offset += vals.size
        vals = struct.Struct('@' + 9 * 'd')
        marginal_prob4 = vals.unpack_from(data_[offset:])
        offset += vals.size

        ret = []
        ret.append(np.array(marginal_prob1))
        ret.append(np.array(marginal_prob2))
        ret.append(np.array(marginal_prob3))
        ret.append(np.array(marginal_prob4))

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

class QuantSFModel:
    def __init__(self):
        self.ratios = []
        self.sampled_ratios = []

    def from_file(self, dname):
        try:
            headers = False
            fname = os.path.sep.join([dname, 'quant.sf'])
            # print(fname)
            with open(fname) as inp_file:
                for line in inp_file:
                    if not headers:
                        headers = line.split()
                        continue
                    data = line.split()
                    actual_length = float(data[1])
                    effective_length = float(data[2])
                    if effective_length:
                        self.ratios.append(actual_length/effective_length)
                # self.ratios.sort()
                self.sampled_ratios = self.ratios[::2000]
        except IOError:
            print("Could not open file {}".format(dname))
            return False
        return True
