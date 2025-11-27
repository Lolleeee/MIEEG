clear; close all;

% Add EEGLAB toolbox path
addpath('/home/lolly/Matlab/my_toolboxes/Multimodal_CNS/eeglab2025.0.0');

input_path  = "/media/lolly/SSD/MotorImagery/S*";
output_path = "/media/lolly/SSD/MotorImagery_Preprocessed";

% Specify target channels to keep (32 channels)
% Not all channels are available, closest are taken
% target_chans = string({
%     "Fp1","Fp2","F7","F3","Fz","F4","F8", ...
%     "FC5","FC1","FC2","FC6","T7","C3","Cz","C4","T8", ...
%     "TP9","CP5","CP1","CP2","CP6","TP10","P7","P3","Pz","P4","P8", ...
%     "PO9","O1","Oz","O2","PO10"
% });

%Swapped TP9, TP10, P09 and P010 for TP7, TP8, PO7 and P09
target_chans = string({
    "Fp1","Fp2","F7","F3","Fz","F4","F8", ...
    "FC5","FC1","FC2","FC6","T7","C3","Cz","C4","T8", ...
    "TP7","CP5","CP1","CP2","CP6","TP8","P7","P3","Pz","P4","P8", ...
    "PO7","O1","Oz","O2","PO8"
});

patients = dir(input_path);

for p = 88:length(patients)
    edf_list = dir(fullfile(patients(p).folder, patients(p).name, '*.edf'));
    
    for r = 1:length(edf_list)
        fprintf("Loading EDF: %s\n", edf_list(r).name);
        filepath = fullfile(edf_list(r).folder, edf_list(r).name);
        
        % Load EDF file as timetable (each cell = 160x1 for 1 sec at 160 Hz)
        TT = edfread(filepath);

        num_seconds = height(TT);
        Fs = 160;  % original sampling frequency
        
        % Extract channel names from timetable and clean trailing underscores
        all_chans = string(TT.Properties.VariableNames);
        all_chans_clean = strip(all_chans, 'right', '_');
        % Find indices of target channels within all channels
        [found, idx] = ismember(lower(target_chans), lower(all_chans_clean));
        if any(~found)
            warning('Missing channels:');
            disp(target_chans(~found));
        end
        
        % Keep only found channels and sync target channel list
        TT = TT(:, idx(found));
        target_chans_filtered = target_chans(found);
        num_channels = width(TT);
        if length(TT{1, "F3__"}{1}) ~= 160
            fprintf("Detected abnormal length %d", length(TT{1, "F3__"}))
            continue
        end
        % Pre-allocate continuous EEG matrix (channels x samples)
        total_samples = num_seconds * Fs;
        full_eeg = zeros(num_channels, total_samples);
        
        % Fill continuous EEG matrix by concatenating each second
        for ch = 1:num_channels
            for s = 1:num_seconds
                sample_range = (s-1)*Fs + (1:Fs);
                full_eeg(ch, sample_range) = TT{s, ch}{1};
            end
        end
        
        % Load standard 10-20 electrode location template
        locs = readlocs('standard_1020.elc');
        template_labels = string({locs.labels});
        
        % Map filtered target channels to standard 10-20 locations
        [~, loc_idx] = ismember(target_chans_filtered, template_labels);
        valid = loc_idx > 0;
        
        locs = locs(loc_idx(valid));
        full_eeg = full_eeg(valid, :);
        target_chans_filtered = target_chans_filtered(valid);
        num_channels = sum(valid);

        % Import EEG data into EEGLAB
        [ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;
        EEG = pop_importdata('dataformat', 'array', ...
                             'nbchan', num_channels, ...
                             'data', full_eeg, ...
                             'setname', 'RawEDF', ...
                             'srate', Fs, ...
                             'chanlocs', locs);
        
        EEG = eeg_checkset(EEG);
        
        [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 0, 'gui', 'off');

        % Preprocessing: resample, rereference, clean, interpolate, ICA
        EEG = pop_resample(EEG, 160);
        EEG = pop_reref(EEG, []);
        EEG = pop_clean_rawdata(EEG, ...
            'FlatlineCriterion', 5, ...
            'ChannelCriterion', 0.8, ...
            'LineNoiseCriterion', 4, ...
            'Highpass', 'off', ...
            'BurstCriterion', 20, ...
            'WindowCriterion', 'off', ...
            'BurstRejection', 'off', ...
            'Distance', 'Euclidian');
        EEG = pop_interp(EEG, locs, 'spherical');
        EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'rndreset', 'yes');
        EEG = pop_iclabel(EEG, 'default');
        
        % Remove components classified as artifacts with confidence >= 0.8 (muscle, eye)
        EEG = pop_icflag(EEG, ...
            [NaN NaN; 0.8 1; 0.8 1; NaN NaN; NaN NaN; NaN NaN; NaN NaN]);
        EEG = pop_subcomp(EEG, find(EEG.reject.gcompreject == 1), 0);
        
        % Create output folder if it doesn't exist
        output_folder = fullfile(output_path, sprintf('P%d', p));
        if ~exist(output_folder, 'dir')
            mkdir(output_folder);
        end
        
        out_eeg = EEG.data;
        varName = sprintf('P%d_trial%d', p, r);
        save(fullfile(output_folder, [varName '.mat']), 'out_eeg');
    end
end
