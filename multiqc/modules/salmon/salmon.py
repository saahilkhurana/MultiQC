#!/usr/bin/env python

""" MultiQC module to parse output from Salmon """

from __future__ import print_function
from collections import OrderedDict
import json
import logging
import os

from multiqc import config
from multiqc.plots import linegraph
from multiqc.modules.base_module import BaseMultiqcModule
from multiqc.modules.salmon.readers import GCModel, SeqModel
# Initialise the logger
log = logging.getLogger(__name__)

class MultiqcModule(BaseMultiqcModule):

    def __init__(self):

        # Initialise the parent object
        super(MultiqcModule, self).__init__(name='Salmon', anchor='salmon',
        href='http://combine-lab.github.io/salmon/',
        info="is a tool for quantifying the expression of transcripts using RNA-seq data.")

        # Parse meta information. JSON win!
        self.salmon_meta = dict()
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
                        log.debug("Duplicate sample name found! Overwriting: {}".format(s_name))
                    self.add_data_source(f, s_name)
                    self.salmon_fld[s_name] = parsed

        # Parse GC Bias Distribution logs
        self.salmon_gcbias = dict()
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
                    ratio[i] = float(obs_weighted[i]/exp_weighted[i])

                s_name = os.path.abspath(f['root'])
                # log.debug(s_name)
                self.add_data_source(f, s_name)
                self.salmon_gcbias[s_name] = ratio

        # Parse Sequence bias Distribution logs
        self.salmon_seq_bias_3 = dict()
        self.salmon_seq_bias_5 = dict()
        for f in self.find_log_files('salmon/fld'):
            if os.path.basename(f['root']) == 'libParams':
                path = os.path.abspath(f['root'])
                # log.debug(path)
                path_mod = path[:-10]
                # log.debug(path_mod)
                if 'no_bias' in path_mod:
                    continue
                seqModel = SeqModel()
                seqModel.from_file(path_mod)

                obs3 = seqModel.obs3_seqMat[0]
                log.debug(obs3)
                obs5 = seqModel.obs5_seqMat[0]
                exp3 = seqModel.exp3_seqMat[0]
                exp5 = seqModel.exp5_seqMat[0]

                ratio3 = OrderedDict()
                for i in range(len(obs3)):
                    ratio3[i] = float(obs3[i]/exp3[i])
                ratio5 = OrderedDict()
                for i in range(len(obs5)):
                    ratio5[i] = float(obs5[i]/exp5[i])

                s_name = os.path.abspath(f['root'])
                # log.debug(s_name)
                self.add_data_source(f, s_name)
                self.salmon_seq_bias_3[s_name] = ratio3
                self.salmon_seq_bias_5[s_name] = ratio5

        # Filter to strip out ignored sample names
        self.salmon_meta = self.ignore_samples(self.salmon_meta)
        self.salmon_fld = self.ignore_samples(self.salmon_fld)

        if len(self.salmon_meta) == 0 and len(self.salmon_fld) == 0:
            raise UserWarning

        if len(self.salmon_meta) > 0:
            log.info("Found {} meta reports".format(len(self.salmon_meta)))
            self.write_data_file(self.salmon_meta, 'multiqc_salmon')
        if len(self.salmon_fld) > 0:
            log.info("Found {} fragment length distributions".format(len(self.salmon_fld)))
        if len(self.salmon_gcbias) > 0:
            log.info("Found {} GC Bias distributions".format(len(self.salmon_gcbias)))
        if len(self.salmon_seq_bias_3) > 0:
            log.info("Found {} Sequence 3' Bias distributions".format(len(self.salmon_seq_bias_3)))
        if len(self.salmon_seq_bias_5) > 0:
            log.info("Found {} Sequence 5' Bias distributions".format(len(self.salmon_seq_bias_5)))

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
        self.add_section( plot = linegraph.plot(self.salmon_fld, pconfig) )

        pconfig_gcbias = {
            'smooth_points': 500,
            'id': 'salmon_plot',
            'title': 'Salmon: GC Bias Distribution',
            'ylab': 'Ratio of Observed to Expected',
            'xlab': 'Bins',
            'ymin': 0,
            'xmin': 0,
            'tt_label': '<b>{point.x:,.0f} bp</b>: {point.y:,.0f}',
        }
        self.add_section( plot = linegraph.plot(self.salmon_gcbias, pconfig_gcbias) )

        pconfig_seq_bias_3 = {
            'smooth_points': 500,
            'id': 'salmon_plot',
            'title': "Salmon: Sequence 3' Bias Distribution",
            'ylab': 'Ratio of Observed to Expected',
            'xlab': 'Context Length',
            'ymin': 0,
            'xmin': 0,
            'tt_label': '<b>{point.x:,.0f} bp</b>: {point.y:,.0f}',
        }
        self.add_section( plot = linegraph.plot(self.salmon_seq_bias_3, pconfig_seq_bias_3) )

        pconfig_seq_bias_5 = {
            'smooth_points': 500,
            'id': 'salmon_plot',
            'title': "Salmon: Sequence 5' Bias Distribution",
            'ylab': 'Ratio of Observed to Expected',
            'xlab': 'Context Length',
            'ymin': 0,
            'xmin': 0,
            'tt_label': '<b>{point.x:,.0f} bp</b>: {point.y:,.0f}',
        }
        self.add_section( plot = linegraph.plot(self.salmon_seq_bias_5, pconfig_seq_bias_5) )
