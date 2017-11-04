#!/usr/bin/env python

""" MultiQC module to parse output from Salmon """

from __future__ import print_function
from collections import OrderedDict
import json
import logging
import os
from math import sqrt, log
from numpy import zeros, array
from multiqc import config
from multiqc.plots import linegraph, heatmap
from multiqc.modules.base_module import BaseMultiqcModule
from multiqc.modules.salmon.readers import GCModel, SeqModel
# Initialise the logger
logger = logging.getLogger(__name__)

class MultiqcModule(BaseMultiqcModule):

    def __init__(self):

        # Initialise the parent object
        super(MultiqcModule, self).__init__(
            name='Salmon',
            anchor='salmon',
            href='http://combine-lab.github.io/salmon/',
            info="is a tool for quantifying the expression of transcripts using RNA-seq data.")

        self.nucleotides = ['A', 'C', 'G', 'T']
        self.salmon_meta = dict()
        self.salmon_gcbias = dict()
        self.matrix_gc = []
        self.matrix_seq3 = [[] for i in range(len(self.nucleotides))]
        self.matrix_seq5 = [[] for i in range(len(self.nucleotides))]

        self.general_stats()
        if len(self.salmon_meta) > 0:
            logger.info("Found {} meta reports".format(len(self.salmon_meta)))
            self.write_data_file(self.salmon_meta, 'multiqc_salmon')

        self.fld()
        if len(self.salmon_fld) > 0:
            logger.info("Found {} fragment length distributions".format(len(self.salmon_fld)))

        self.gc_bias()
        if len(self.salmon_gcbias) > 0:
            logger.info("Found {} GC Bias distributions".format(len(self.salmon_gcbias)))

        self.seq_bias()
        if len(self.salmon_seq_bias_3) > 0:
            logger.info("Found {} Sequence 3' Bias distributions".format(len(self.salmon_seq_bias_3[0])))
        if len(self.salmon_seq_bias_5) > 0:
            logger.info("Found {} Sequence 5' Bias distributions".format(len(self.salmon_seq_bias_5[0])))

        self.heatmap()

        # Filter to strip out ignored sample names
        self.salmon_meta = self.ignore_samples(self.salmon_meta)
        self.salmon_fld = self.ignore_samples(self.salmon_fld)

        if len(self.salmon_meta) == 0 and len(self.salmon_fld) == 0:
            raise UserWarning

    def get_sample_name(self, name):
        return name.split('/')[-3]

    def general_stats(self):
        # Add alignment rate to the general stats table
        headers = OrderedDict()
        headers['percent_mapped'] = {
            'title': '% Aligned',
            'description': '% Mapped reads',
            'max': 100,
            'min': 0,
            'suffix': '%',
            'scale': 'YlGn'
        }
        headers['num_mapped'] = {
            'title': 'M Aligned',
            'description': 'Mapped reads (millions)',
            'min': 0,
            'scale': 'PuRd',
            'modify': lambda x: float(x) / 1000000,
            'shared_key': 'read_count'
        }
        self.general_stats_addcols(self.salmon_meta, headers)

    def fld(self):

        for f in self.find_log_files('salmon/meta'):
            # Get the s_name from the parent directory
            s_name = os.path.basename( os.path.dirname(f['root']) )
            s_name = self.clean_s_name(s_name, f['root'])
            self.salmon_meta[s_name] = json.loads(f['f'])
        # Parse Fragment Length Distribution logs
        self.salmon_fld = dict()
        for f in self.find_log_files('salmon/fld'):
            # Get the s_name from the parent directory
            if os.path.basename(f['root']) == 'libParams':
                s_name = os.path.basename( os.path.dirname(f['root']) )
                s_name = self.clean_s_name(s_name, f['root'])
                parsed = OrderedDict()
                for i, v in enumerate(f['f'].split()):
                    parsed[i] = float(v)
                if len(parsed) > 0:
                    if s_name in self.salmon_fld:
                        logger.debug("Duplicate sample name found! Overwriting: {}".format(s_name))
                    self.add_data_source(f, s_name)
                    self.salmon_fld[s_name] = parsed

        # Fragment length distribution plot
        pconfig = {
            'smooth_points': 500,
            'id': 'salmon_plot',
            'title': 'Salmon: Fragment Length Distribution',
            'ylab': 'Fraction',
            'xlab': 'Fragment Length (bp)',
            'ymin': 0,
            'xmin': 0,
            'tt_label': '<b>{point.x:,.0f} bp</b>: {point.y:,.0f}',
        }
        self.add_section(plot = linegraph.plot(self.salmon_fld, pconfig))

    def gc_bias(self):
        # Parse GC Bias Distribution logs

        for f in self.find_log_files('salmon/fld'):
            if os.path.basename(f['root']) == 'libParams':
                path = os.path.abspath(f['root'])
                # log.debug(path)
                path_mod = path[:-10]
                # log.debug(path_mod)
                if 'no_bias' in path_mod:
                    continue
                gcModel = GCModel()
                gcModel.from_file(path_mod)

                obs = gcModel.obs_
                obs_weights = gcModel.obs_weights_
                # log.debug(obs_weights)
                exp = gcModel.exp_
                exp_weights = gcModel.exp_weights_

                obs_weighted = obs[0]*obs_weights[0] + obs[1]*obs_weights[1] + obs[2]*obs_weights[2]
                exp_weighted = exp[0]*exp_weights[0] + exp[1]*exp_weights[1] + exp[2]*exp_weights[2]

                ratio = OrderedDict()
                for i in range(len(obs_weighted)):
                    ratio[i*4] = float(obs_weighted[i]/exp_weighted[i])
                self.matrix_gc.append(list(ratio.values()))

                s_name = os.path.abspath(f['root'])

                self.add_data_source(f, s_name)
                self.salmon_gcbias[self.get_sample_name(s_name)] = ratio

        pconfig_gcbias = {
            'smooth_points': 500,
            'id': 'salmon_plot',
            'title': 'Salmon: GC Bias Distribution',
            'ylab': 'Ratio of Observed to Expected',
            'xlab': 'Bins',
            'ymin': 0,
            'xmin': 0,
            'tt_label': '<b>{point.x:,.0f} </b>: {point.y:,.0f}',
        }
        self.add_section(plot = linegraph.plot(self.salmon_gcbias, pconfig_gcbias))

    def seq_bias(self):
        # Parse Sequence bias Distribution logs

        self.salmon_seq_bias_3 = [dict() for i in range(4)]
        self.salmon_seq_bias_5 = [dict() for i in range(4)]
        for f in self.find_log_files('salmon/fld'):
            if os.path.basename(f['root']) == 'libParams':
                path = os.path.abspath(f['root'])

                path_mod = path[:-10]

                if 'no_bias' in path_mod:
                    continue

                seqModel = SeqModel()
                seqModel.from_file(path_mod)

                obs3 = seqModel.obs3_seqMat
                obs5 = seqModel.obs5_seqMat
                exp3 = seqModel.exp3_seqMat
                exp5 = seqModel.exp5_seqMat

                ratio3 = [OrderedDict() for i in range(4)]
                ratio5 = [OrderedDict() for i in range(4)]

                s_name = os.path.abspath(f['root'])

                for i in range(len(self.nucleotides)) :

                    for j in range(len(obs3[i])):
                        ratio3[i][j] = float(obs3[i][j]*1.0)/(exp3[i][j]*1.0)
                    for j in range(len(obs5[i])):
                        ratio5[i][j] = (obs5[i][j]*1.0)/(exp5[i][j]*1.0)

                    self.salmon_seq_bias_3[i][self.get_sample_name(s_name)] = ratio3[i]
                    self.matrix_seq3[i].append(list(ratio3[i].values()))
                    self.salmon_seq_bias_5[i][self.get_sample_name(s_name)] = ratio5[i]
                    self.matrix_seq5[i].append(list(ratio5[i].values()))

                self.add_data_source(f, s_name)

        for i in range(len(self.nucleotides)) :
            pconfig_seq_bias_3 = {
                'smooth_points': 500,
                'id': 'salmon_plot',
                'title': "Salmon: Sequence 3' Bias Distribution " + self.nucleotides[i],
                'ylab': 'Ratio of Observed to Expected',
                'xlab': 'Context Length',
                'ymin': 0,
                'xmin': 0,
                'tt_label': '<b>{point.x:,.0f} </b>: {point.y:,.0f}',
            }
            self.add_section( plot = linegraph.plot(self.salmon_seq_bias_3[i], pconfig_seq_bias_3) )

        for i in range(len(self.nucleotides)) :
            pconfig_seq_bias_5 = {
                'smooth_points': 500,
                'id': 'salmon_plot',
                'title': "Salmon: Sequence 5' Bias Distribution " + self.nucleotides[i],
                'ylab': 'Ratio of Observed to Expected',
                'xlab': 'Context Length',
                'ymin': 0,
                'xmin': 0,
                'tt_label': '<b>{point.x:,.0f} </b>: {point.y:,.0f}',
            }
            self.add_section( plot = linegraph.plot(self.salmon_seq_bias_5[i], pconfig_seq_bias_5) )

    def heatmap(self):

        names = []
        for f in self.find_log_files('salmon/fld'):
            if os.path.basename(f['root']) == 'libParams':
                s_name = os.path.abspath(f['root'])
                names.append(self.get_sample_name(s_name))

        number_samples = len(self.matrix_gc)
        sims = [[0 for j in range(number_samples)] for i in range(number_samples)]
        for i in range(number_samples) :
            for j in range(number_samples) :
                sims[i][j] = self.jensen_shannon_divergence(self.matrix_gc[i], self.matrix_gc[j])
                for k in range(len(self.nucleotides)):
                    sims[i][j] += self.jensen_shannon_divergence(self.matrix_seq3[k][i], self.matrix_seq3[k][j])
                    sims[i][j] += self.jensen_shannon_divergence(self.matrix_seq5[k][i], self.matrix_seq5[k][j])
                sims[i][j] /= 9.0

        pconfig_sim = {
            'title': 'Jensen Shannon Divergence similarity',
            'xTitle': 'Samples',
            'yTitle': 'Samples',
        }

        self.add_section(plot = heatmap.plot(sims, names, pconfig=pconfig_sim))

    def kl_divergence(self, p, q) :
        """ Compute KL divergence of two vectors, K(p || q)."""
        return sum(p[x] * log((p[x]) / (q[x] if q[x] != 0 else 1)) for x in range(len(p)) if p[x] != 0.0 or p[x] != 0)

    def jensen_shannon_divergence(self, p, q):
        """ Returns the Jensen-Shannon divergence. """
        self.JSD = 0.0
        weight = 0.5
        average = zeros(len(p)) #Average
        for x in range(len(p)):
            average[x] = weight * p[x] + (1 - weight) * q[x]
            self.JSD = (weight * self.kl_divergence(array(p), average)) + ((1 - weight) * self.kl_divergence(array(q), average))
        return 1-(self.JSD/sqrt(2 * log(2)))
