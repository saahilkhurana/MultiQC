#!/usr/bin/env python

""" MultiQC module to parse output from Salmon """

from __future__ import print_function
from collections import OrderedDict
import json
import logging
import os
import numpy as np
from math import sqrt, log
from numpy import zeros, array
from multiqc import config
from multiqc.plots import linegraph, heatmap, table
from multiqc.modules.base_module import BaseMultiqcModule
from multiqc.modules.salmon.readers import GCModel, SeqModel, QuantSFModel
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
        self.salmon_gcbias_rows = [dict() for i in range(3)]
        self.salmon_seq_bias_3 = [dict() for i in range(len(self.nucleotides))]
        self.salmon_seq_bias_5 = [dict() for i in range(len(self.nucleotides))]
        self.salmon_seq_avg_bias = dict()
        self.matrix_gc = {}
        self.matrix_seq3 = [{} for i in range(len(self.nucleotides))]
        self.matrix_seq5 = [{} for i in range(len(self.nucleotides))]
        self.salmon_quant_sf = dict()

        self.general_stats()
        if len(self.salmon_meta) > 0:
            logger.info("Found {} meta reports".format(len(self.salmon_meta)))
            self.write_data_file(self.salmon_meta, 'multiqc_salmon')

        self.fld()
        if len(self.salmon_fld) > 0:
            logger.info("Found {} fragment length distributions".format(len(self.salmon_fld)))

        # Generates GC Bias linegraph plot
        self.gc_bias()
        if len(self.salmon_gcbias) > 0:
            logger.info("Found {} GC Bias distributions".format(len(self.salmon_gcbias)))

        # Generates 8 Sequence Bias linegraph plots (4 for 3', 4 for 5')
        self.seq_bias()
        if len(self.salmon_seq_bias_3) > 0:
            logger.info("Found {} Sequence 3' Bias distributions".format(len(self.salmon_seq_bias_3[0])))
        if len(self.salmon_seq_bias_5) > 0:
            logger.info("Found {} Sequence 5' Bias distributions".format(len(self.salmon_seq_bias_5[0])))

        # Generates Heatmap for samples containing all features (9 -> 1 for GC Bias and 8 for Seq bias)
        self.heatmap()

        # Generates Quant.sf linegraph plot
        self.quant_sf()
        if len(self.salmon_quant_sf) > 0:
            logger.info("Found {} quant.sf distributions".format(len(self.salmon_quant_sf)))

        # Filter to strip out ignored sample names
        self.salmon_meta = self.ignore_samples(self.salmon_meta)
        self.salmon_fld = self.ignore_samples(self.salmon_fld)

        if len(self.salmon_meta) == 0 and len(self.salmon_fld) == 0:
            raise UserWarning

    def get_sample_name(self, name):
        return name.split('/')[-3]

    def general_stats(self):
        # Add alignment rate to the general stats table
        for f in self.find_log_files('salmon/meta'):
            # Get the s_name from the parent directory
            s_name = os.path.basename( os.path.dirname(f['root']) )
            s_name = self.clean_s_name(s_name, f['root'])
            self.salmon_meta[s_name] = json.loads(f['f'])
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
            'tt_label': '<b>{point.x:,.0f} bp</b>: {point.y:,.3f}',
        }
        self.add_section(plot = linegraph.plot(self.salmon_fld, pconfig))

    def gc_bias(self):
        """
        Computes GC Bias
        Input : Reads 3 lists corresponding to observed and 3 for expected
        Calculates the weighted sum for obs, exp
        Output : For proper X-Axis labels, scales the keys in ratio dict
        """

        for f in self.find_log_files('salmon/fld'):
            if os.path.basename(f['root']) == 'libParams':
                path = os.path.abspath(f['root'])
                path_mod = path[:-10]

                if 'no_bias' in path_mod:
                    continue

                path_meta_info = os.path.join(path_mod, 'aux_info', 'meta_info.json')
                with open(path_meta_info, 'r') as info:
                    meta_info = json.load(info)
                    if not meta_info['gc_bias_correct']:
                        continue

                gcModel = GCModel()
                gcModel.from_file(path_mod)

                obs = gcModel.obs_
                obs_weights = gcModel.obs_weights_

                exp = gcModel.exp_
                exp_weights = gcModel.exp_weights_

                scale_bin_factor = 100.0/(len(obs[0])*1.0)
                s_name = os.path.abspath(f['root'])

                for i in range(3):
                    obs_weighted_row = obs[i]*obs_weights[i]
                    exp_weighted_row = exp[i]*exp_weights[i]

                    ratio = OrderedDict()
                    for j in range(len(obs_weighted_row)):
                        ratio[j*scale_bin_factor] = float(obs_weighted_row[j]/exp_weighted_row[j])

                    self.salmon_gcbias_rows[i][self.get_sample_name(s_name)] = ratio

                obs_weighted = obs[0]*obs_weights[0] + obs[1]*obs_weights[1] + obs[2]*obs_weights[2]
                exp_weighted = exp[0]*exp_weights[0] + exp[1]*exp_weights[1] + exp[2]*exp_weights[2]

                #scale_bin_factor = 100.0/(len(obs[0])*1.0)

                ratio = OrderedDict()
                for i in range(len(obs_weighted)):
                    ratio[i*scale_bin_factor] = float(obs_weighted[i]/exp_weighted[i])

                #s_name = os.path.abspath(f['root'])

                self.matrix_gc[self.get_sample_name(s_name)] = list(ratio.values())
                self.salmon_gcbias[self.get_sample_name(s_name)] = ratio

                self.add_data_source(f, s_name)

        row_labels = ['Low', 'Medium', 'High']
        for i in range(3):
            pconfig_gcbias_row = {
                'smooth_points': 500,
                'id': 'salmon_plot',
                'title': 'Salmon: GC Bias Distribution Row ' + row_labels[i],
                'ylab': 'Ratio of Observed to Expected',
                'xlab': 'Bins',
                'ymin': 0,
                'xmin': 0,
                'tt_label': '<b>{point.x:,.0f} </b>: {point.y:,.3f}',
            }
            if len(self.salmon_gcbias_rows[i]) > 0:
                self.add_section(plot = linegraph.plot(self.salmon_gcbias_rows[i], pconfig_gcbias_row))

        pconfig_gcbias = {
            'smooth_points': 500,
            'id': 'salmon_plot',
            'title': 'Salmon: GC Bias Distribution',
            'ylab': 'Ratio of Observed to Expected',
            'xlab': 'Bins',
            'ymin': 0,
            'xmin': 0,
            'tt_label': '<b>{point.x:,.0f} </b>: {point.y:,.3f}',
        }
        if len(self.salmon_gcbias) > 0:
            self.add_section(plot = linegraph.plot(self.salmon_gcbias, pconfig_gcbias))

    def seq_bias(self):
        """
        Computes Sequence Bias
        Input : Reads 4 lists (one for each nucleotide) for each of obs 3', obs5', exp3', exp5'
        Calculates the ratio of obs, exp for each nucleotide
        Output : Plots 8 linegraphs - 3' and 5' for each nucleotide
        """
        for f in self.find_log_files('salmon/fld'):
            if os.path.basename(f['root']) == 'libParams':
                path = os.path.abspath(f['root'])
                path_mod = path[:-10]

                if 'no_bias' in path_mod:
                    continue

                path_meta_info = os.path.join(path_mod, 'aux_info', 'meta_info.json')
                with open(path_meta_info, 'r') as info:
                    meta_info = json.load(info)
                    if not meta_info['seq_bias_correct']:
                        continue

                seqModel = SeqModel()
                seqModel.from_file(path_mod)

                obs3 = seqModel.obs3_seqMat
                obs5 = seqModel.obs5_seqMat
                exp3 = seqModel.exp3_seqMat
                exp5 = seqModel.exp5_seqMat

                ratio3 = [OrderedDict() for i in range(len(self.nucleotides))]
                ratio5 = [OrderedDict() for i in range(len(self.nucleotides))]
                avg = OrderedDict()

                s_name = os.path.abspath(f['root'])
                sample_name = self.get_sample_name(s_name)

                for i in range(len(self.nucleotides)) :

                    for j in range(len(obs3[i])):
                        ratio3[i][j-4] = float(obs3[i][j]*1.0)/(exp3[i][j]*1.0)
                        if j-4 not in avg:
                            avg[j-4] = 0.0
                        avg[j-4] += ratio3[i][j-4]
                    for j in range(len(obs5[i])):
                        ratio5[i][j-4] = (obs5[i][j]*1.0)/(exp5[i][j]*1.0)
                        avg[j-4] += ratio5[i][j-4]


                    self.salmon_seq_bias_3[i][sample_name] = ratio3[i]
                    self.matrix_seq3[i][sample_name] = (list(ratio3[i].values()))
                    self.salmon_seq_bias_5[i][sample_name] = ratio5[i]
                    self.matrix_seq5[i][sample_name] = (list(ratio5[i].values()))

                for key in avg.keys():
                    avg[key] /= 8.0

                self.salmon_seq_avg_bias[sample_name] = avg

                self.add_data_source(f, s_name)

        for i in range(len(self.nucleotides)) :
            pconfig_seq_bias_3 = {
                'smooth_points': 500,
                'id': 'salmon_plot',
                'title': "Salmon: Sequence 3' Bias Distribution " + self.nucleotides[i],
                'ylab': 'Ratio of Observed to Expected',
                'xlab': 'Context Length',
                'ymin': 0,
                'xmin': -4,
                'tt_label': '<b>{point.x:,.0f} </b>: {point.y:,.3f}',
            }
            if len(self.salmon_seq_bias_3) > 0:
                self.add_section( plot = linegraph.plot(self.salmon_seq_bias_3[i], pconfig_seq_bias_3) )

        for i in range(len(self.nucleotides)) :
            pconfig_seq_bias_5 = {
                'smooth_points': 500,
                'id': 'salmon_plot',
                'title': "Salmon: Sequence 5' Bias Distribution " + self.nucleotides[i],
                'ylab': 'Ratio of Observed to Expected',
                'xlab': 'Context Length',
                'ymin': 0,
                'xmin': -4,
                'tt_label': '<b>{point.x:,.0f} </b>: {point.y:,.3f}',
            }
            if len(self.salmon_seq_bias_5) > 0:
                self.add_section( plot = linegraph.plot(self.salmon_seq_bias_5[i], pconfig_seq_bias_5) )

        pconfig_avg_seq_bias = {
            'smooth_points': 500,
            'id': 'salmon_plot',
            'title': "Salmon: Sequence Average Bias Distribution ",
            'ylab': 'Ratio of Observed to Expected',
            'xlab': 'Context Length',
            'ymin': 0,
            'xmin': -4,
            'tt_label': '<b>{point.x:,.0f} </b>: {point.y:,.3f}',
        }
        if len(self.salmon_seq_avg_bias) > 0:
            self.add_section( plot = linegraph.plot(self.salmon_seq_avg_bias, pconfig_avg_seq_bias) )

    def heatmap(self):
        """
        Generates Heatmap for samples considering all features
        Computes the Jensen Shannon Divergence between all samples
        Value 0 corresponds to similar samples and 1 to dissimilar samples
        Output : Plots heatmap and table showing presence of missing features in samples
        """
        names = []
        gc_names = []
        seq_names = []
        missing_names = {}
        gc_exists = {}
        seq_exists = {}
        for f in self.find_log_files('salmon/fld'):
            if os.path.basename(f['root']) == 'libParams':
                s_name = os.path.abspath(f['root'])
                path = s_name[:-10]
                sample_name = self.get_sample_name(s_name)

                if 'no_bias' in s_name:
                    continue
                path_meta_info = os.path.join(path, 'aux_info', 'meta_info.json')
                with open(path_meta_info, 'r') as info:
                    meta_info = json.load(info)

                    gc_exists[sample_name] = meta_info['gc_bias_correct']
                    seq_exists[sample_name] = meta_info['seq_bias_correct']

                if gc_exists[sample_name]:
                    gc_names.append(sample_name)
                    if sample_name not in missing_names:
                        missing_names[sample_name] = {}
                    missing_names[sample_name]['Missing GC Feature'] = 'No'
                else :
                    if sample_name not in missing_names:
                        missing_names[sample_name] = {}
                    missing_names[sample_name]['Missing GC Feature'] = 'Yes'
                if seq_exists[sample_name]:
                    seq_names.append(sample_name)
                    if sample_name not in missing_names:
                        missing_names[sample_name] = {}
                    missing_names[sample_name]['Missing Seq Feature'] = 'No'
                else:
                    if sample_name not in missing_names:
                        missing_names[sample_name] = {}
                    missing_names[sample_name]['Missing Seq Feature'] = 'Yes'
                if gc_exists[sample_name] and seq_exists[sample_name]:
                    names.append(sample_name)

        sims_gc = [[0 for j in range(len(gc_names))] for i in range(len(gc_names))]
        sims_3 = [[0 for j in range(len(seq_names))] for i in range(len(seq_names))]
        sims_5 = [[0 for j in range(len(seq_names))] for i in range(len(seq_names))]
        sims = [[0 for j in range(len(names))] for i in range(len(names))]

        for i in range(len(names)):
            for j in range(len(names)):
                feature_count = 0
                if gc_exists[names[i]] and gc_exists[names[j]]:
                    sims[i][j] += self.jensen_shannon_divergence(self.matrix_gc[names[i]], self.matrix_gc[names[j]])
                    feature_count += 1.0
                for k in range(len(self.nucleotides)):
                    if seq_exists[names[i]] and seq_exists[names[j]]:
                        sims[i][j] += self.jensen_shannon_divergence(self.matrix_seq3[k][names[i]], self.matrix_seq3[k][names[j]])
                        sims[i][j] += self.jensen_shannon_divergence(self.matrix_seq5[k][names[i]], self.matrix_seq5[k][names[j]])
                        feature_count += 2.0

                sims[i][j] /= feature_count
        for i in range(len(gc_names)):
            for j in range(len(gc_names)):
                if gc_exists[gc_names[i]] and gc_exists[gc_names[j]]:
                    sims_gc[i][j] += self.jensen_shannon_divergence(self.matrix_gc[gc_names[i]], self.matrix_gc[gc_names[j]])

        for i in range(len(seq_names)):
            for j in range(len(seq_names)):
                for k in range(len(self.nucleotides)):
                    if seq_exists[seq_names[i]] and seq_exists[seq_names[j]]:
                        sims_3[i][j] += self.jensen_shannon_divergence(self.matrix_seq3[k][seq_names[i]], self.matrix_seq3[k][seq_names[j]])
                        sims_5[i][j] += self.jensen_shannon_divergence(self.matrix_seq5[k][seq_names[i]], self.matrix_seq5[k][seq_names[j]])
                sims_3[i][j] /= (1.0*len(self.nucleotides))
                sims_5[i][j] /= (1.0*len(self.nucleotides))
        pconfig_sim = {
            'title': 'Sample similarity (JSD)',
            'xTitle': 'Samples',
            'yTitle': 'Samples',
        }
        pconfig_sim_gc = {
            'title': 'Feature GC Sample similarity (JSD)',
            'xTitle': 'Samples',
            'yTitle': 'Samples',
        }
        pconfig_sim_3 = {
            'title': 'Feature Seq 3 Sample similarity (JSD)',
            'xTitle': 'Samples',
            'yTitle': 'Samples',
        }
        pconfig_sim_5 = {
            'title': 'Feature Seq 5 Sample similarity (JSD)',
            'xTitle': 'Samples',
            'yTitle': 'Samples',
        }

        if len(gc_exists) > 0:
            self.add_section(plot = heatmap.plot(sims_gc, gc_names, pconfig=pconfig_sim_gc))
        if len(seq_exists) > 0:
            self.add_section(plot = heatmap.plot(sims_3, seq_names, pconfig=pconfig_sim_3))
        if len(seq_exists) > 0:
            self.add_section(plot = heatmap.plot(sims_5, seq_names, pconfig=pconfig_sim_5))
        if len(names) > 0:
            self.add_section(plot = heatmap.plot(sims, names, pconfig=pconfig_sim))

        self.add_section(plot = table.plot(missing_names))

    def jensen_shannon_divergence(self, P, Q):
        """Compute the Jensen-Shannon divergence between two probability distributions.
            0 being very similar samples
            Reference : https://stackoverflow.com/questions/15880133/jensen-shannon-divergence
        """
        def _kldiv(A, B):
            return np.sum([v for v in A * np.log2(A/B) if not np.isnan(v)])

        P = np.array(P)
        Q = np.array(Q)

        M = 0.5 * (P + Q)

        return 0.5 * (_kldiv(P, M) +_kldiv(Q, M))

    def quant_sf(self):
        """
        Computes GC Bias
        Input : Reads 3 lists corresponding to observed and 3 for expected
        Calculates the weighted sum for obs, exp
        Output : For proper X-Axis labels, scales the keys in ratio dict
        """

        for f in self.find_log_files('salmon/fld'):
            if os.path.basename(f['root']) == 'libParams':
                path = os.path.abspath(f['root'])
                path_mod = path[:-10]
                s_name = path

                quantSFModel = QuantSFModel()
                quantSFModel.from_file(path_mod)

                quantSF_ratio = quantSFModel.ratios

                ratio = OrderedDict()
                for i in range(len(quantSF_ratio)):
                    ratio[i] = quantSF_ratio[i]

                self.salmon_quant_sf[s_name] = ratio
                self.add_data_source(f, s_name)

        pconfig_quant = {
            'smooth_points': 500,
            'id': 'salmon_plot',
            'title': 'Salmon: QuantSF Distribution',
            'ylab': 'Ratio of Actual to Effective Transcript Length',
            'xlab': 'Bins',
            'ymin': 0,
            'xmin': 0,
            'tt_label': '<b>{point.x:,.0f} </b>: {point.y:,.3f}'
        }
        if len(self.salmon_quant_sf) > 0:
            self.add_section(plot = linegraph.plot(self.salmon_quant_sf, pconfig_quant))
