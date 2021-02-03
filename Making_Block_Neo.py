# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 16:46:20 2020

@author: F.LARENO-FACCINI
"""
import matplotlib.pyplot as plt
import extrapy.Organize as og
import neo
import numpy as np
import quantities as pq
import extrapy.Behaviour as bv
import pandas as pd

group = 15
mouse = 173
gender = 'Female'
experiment = 'Fixed Delay'
condition = 'No Stim'
protocol = 'P0'

ch_group = ([14,9,12,11,10,13,8,15],[7,0,5,2,3,4,1,6])

if experiment == 'Random Delay':
    skip_last = True
elif experiment == 'Fixed Delay':
    skip_last = False

path = fr'\\equipe2-nas1\F.LARENO-FACCINI\BACKUP FEDE\Ephy\Group {group}\{mouse} (CM16-Buz - {gender})\{experiment}\{protocol}\{condition}'
files = og.file_list(path, ext='.rbf')
bl = neo.Block()

# Analog Signal
for file in files:
    new_path = path+f'\{file}.rbf'
    
    sig = np.fromfile(new_path, dtype='float64').reshape(-1,16)
    anal_sig = neo.AnalogSignal(sig, 
                                units = 'V', 
                                t_start = 0*pq.s, 
                                sampling_rate = 20*pq.kHz, 
                                file_origin = new_path, 
                                name = file)
    
    anal_sig.array_annotate(channel_id = [ch for ch in range(len(sig.T))], channel_group = ['A','A','A','A','A','A','A','A','B','B','B','B','B','B','B','B']) 
    anal_sig.annotate(mouse_id = [f'{mouse}'],
                      group_id = [f'{group}'],
                      mouse_gender = [f'{gender}'])
    
    seg = neo.Segment()
    
    seg.analogsignals.append(anal_sig)
    
    bl.segments.append(seg)

# add annotation to segments
for seg in bl.segments:
    seg.annotate(has_events=True)
    
if experiment == 'Random Delay':
    bl.segments[-1].annotate(has_events=False)
    
    
# Spikes
spike_path = r'D:/F.LARENO.FACCINI/RESULTS/Spike Sorting/Spike Times/Fixed Delay/173-EF-P0_Spike_times_changrp0.xlsx' #TODO automatize this path
spike_df = pd.read_excel(spike_path, sheet_name=None)

for k,v in spike_df.items():
    if 'Cluster ' in k:
        for ind, i in enumerate(v):
            spikes = v[i]
            spikes = spikes[np.isfinite(spikes)]
        
            spike_neo = neo.SpikeTrain(spikes,
                                       t_stop = bl.segments[ind].analogsignals[0].t_stop, 
                                       file_origin = spike_path, 
                                       units = pq.s,
                                       name = k)
            
            bl.segments[ind].spiketrains.append(spike_neo)

# Events
param_path = fr'\\equipe2-nas1\F.LARENO-FACCINI\BACKUP FEDE\Behaviour\Group {group}\{mouse} (CM16-Buz - Female)\{experiment}\P0'

param_files = og.file_list(param_path, no_extension=False, ext='.param')
new_param_path = param_path + f'\{param_files[0]}'

delays2 = bv.extract_random_delay(new_param_path, skip_last=skip_last, fixed_delay=500)
cue1_types,cue1_lens = bv.extract_cue(new_param_path, skip_last=skip_last, cue=1)
cue2_types,cue2_lens = bv.extract_cue(new_param_path, skip_last=skip_last, cue=2)
delays1 = bv.extract_first_delay(new_param_path, skip_last=skip_last)
waters = bv.extract_water_duration(new_param_path, skip_last=skip_last)

for trial_index, trial in enumerate(bl.segments):
    if trial.annotations['has_events']:
        delay2 = delays2[trial_index][0] * pq.ms
        cue1_len = cue1_lens[trial_index] * pq.ms
        cue2_len = cue2_lens[trial_index] * pq.ms
        delay1 = delays1[trial_index] * pq.ms
        water = waters[trial_index] * pq.ms
        
        # This dictionary contains only the durantion of the events.
        # The actual duration is calulated in cumsum
        event_dictionary = {
            'cue_1_on' : trial.t_start.rescale(pq.s),
            'cue_1_off' : cue1_len.rescale(pq.s), 
            'cue_2_on' : delay1.rescale(pq.s), 
            'cue_2_off' : cue2_len.rescale(pq.s),
            'reward_on' : delay2.rescale(pq.s),
            'reward_off' : water.rescale(pq.s)
            }
        
        times = [t.magnitude for t in event_dictionary.values()]
        cumulative_time = np.cumsum(times) * pq.s
        ev = neo.Event(times=cumulative_time, labels=list(event_dictionary.keys()), name='Trial Events')
        
        trial.events.append(ev)    

        # Epochs
        epoch_names = ['cue_1', 'delay_1', 'cue_2',  'delay_2', 'reward']
        for label, duration, time in zip(epoch_names, list(event_dictionary.values())[1:], cumulative_time[:-1]):
            ep = neo.Epoch(times=[time]*pq.s, durations=[duration]*pq.s, labels=[label], name=[label])
            if 'cue_1' in label:
                ep.annotate(cue_type = [f'{cue1_types[trial_index]}'])
            elif 'cue_2' in label:
                ep.annotate(cue_type = [f'{cue2_types[trial_index]}'])
            
            trial.epochs.append(ep)


with neo.NixIO(f'D:\F.LARENO.FACCINI\Preliminary Results\Scripts\Trial Neo\{mouse}_{protocol}.nix', mode='ow') as writer:
    writer.write_block(bl)
    
    