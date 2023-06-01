%% Neurocomputational Mechanisms Underlying Cross-Modal Associations And Their Influence on Perceptual Decisions:
%% 

% Bolam, J., Boyle, S. C., Ince, R. A. A., & Delis (2022).
% Neurocomputational mechanisms underlying cross-modal associations and
% their influence on perceptual decisions.
% NeuroImage, 247, 118841. 

% Linear Discriminant Analysis Outline: 
% Locked to Stimulus-Onset. 
% Window Length = 50ms
% Sliding Window Increments = 10ms from -100ms pre-stimulus to 1500ms post-stimulus
% Logistic Regression
% Permutations - 1000. Significance Level p = 0.05

tic % start stopwatch (just to observe how long it takes on laptop) 

for i = 1:20

    % Loading all participant's information (LDA Labels + EEG - Channels x Timepoints x Trials)
    eval(['load Prepro_all_S0', num2str(i), '_EOG_out.mat dataX'])    
    
    Num_trials = size(dataX.trialinfo,1); % Num_trials = Original number of trials.
    
    % Auditory Trials Only:
    a = 0; 
    for ii = 1:Num_trials
        if dataX.trialinfo(ii,2) < 3
            a = a + 1; 
            dataA.trial{a} = dataX.trial{ii};
        end
    end
    
    % Updating Num_trials (Now Auditory trial only)
    dataA.trialinfo=dataX.trialinfo(dataX.trialinfo(:,2)<3,:); 
    Num_trials = size(dataA.trialinfo,1)
    
    % Exclude trials <300ms & >1200ms: 
    b = 0; 
    for jj = 1:Num_trials
        if (dataA.trialinfo(jj,7)>0.3) & (dataA.trialinfo(jj,7)<1.2)
            b = b+1;
            dataAud.trial{b} = dataA.trial{jj};
        end
    end

    % Updating Num_trials (Exclusion Criteria Applied): 
    dataAud.trialinfo = dataA.trialinfo((dataA.trialinfo(:,7)>0.3) & (dataA.trialinfo(:,7)<1.2),:);
    Num_trials = size(dataAud.trialinfo,1);

    % Resizing EEG Data to pass single-trial LDA analysis: 
    Num_EEG_electrodes = 128; % Number of EEG Electrodes/Channels. 
    Num_timepoints = 401; % Number of timepoints (milliseconds; -0.5-1.5)
    Num_stimfeat = 2; % Stimulus Features (High Pitch Tone/Low Pitch Tone)
    Num_congruency = 2; % Congruency (Congruent/Incongruent).

    % EEG matrix = EEG Channels/Electrodes x Timepoints x Trials:
    EEGaud = zeros(Num_EEG_electrodes,Num_timepoints,Num_trials);
    % Reshape EEG data to fit EEGaud: 
    for kk = 1:Num_trials
        EEGaud(:,:,kk) = dataAud.trial{kk}(1:Num_EEG_electrodes,:); 
    end
    [N_Electrodes, N_Timepoints, N_Trials] = size(EEGaud); 
    
    % Specifying timepoints (from -0.5 to +1.5 seconds):
    timepoints = -0.5:0.005:1.5; % 1 x 401 double. 

    % Classifier's 'Discriminant' Labels:
    trialinfo = dataAud.trialinfo; 
    StimulusFeature_Labels = trialinfo(:,2); % 1 = High Pitch Tone, 2 = Low Pitch Tone
    Congruency_Labels = trialinfo(:,3); % 1 = Congruent, 2 = Incongruent
    
    % LOGISTIC REGRESSION - Sliding Window Width = 50ms, Increments = 10ms:
    EEG_LR = EEGaud; 
    % Counters: 
    ee = 0;
    ff = 0; 
    
    % LDA (LR) - Stimulus Feature:
    for e = 1:1:401-9
        ee = ee + 1; 
        EEGwindow = reshape(EEG_LR(:,[e:e+9],:),128,10*N_Trials);
        [Azloo_LR_sf(:,ee,i),Azsig_LR_sf(:,ee,i),Y_LR_sf(:,ee,i),a_LR_sf(:,ee,i),v_LR_sf(:,ee,i),D_LR_sf] = single_trial_analysis(EEGwindow,StimulusFeature_Labels-1,10,0,[1 1000 0.05],[],0,0)
        e
    end
    
    % LDA (LR) - Congruency: 
    for f = 1:1:401-9
        ff = ff + 1; 
        EEGwindow = reshape(EEG_LR(:,[f:f+9],:),128,10*N_Trials); 
        [Azloo_LR_con(:,ff,i),Azsig_LR_con(:,ff,i),Y_LR_con(:,ff,i),a_LR_con(:,ff,i),v_LR_con(:,ff,i),D_LR_con] = single_trial_analysis(EEGwindow,Congruency_Labels-1,10,0,[1 1000 0.05],[],0,0)
        f
    end
    
end

toc

% Due to timescales, LDA was ran separately for each participant. 
% Saved datasets were then reduced to -100ms pre-stimulus onset to 800ms
% post-stimulus onset.
