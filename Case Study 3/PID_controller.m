%% short description
% script used to implement PID control for the PSE-MFS loop of the grinding 
% circuit case study.
% Code used previously for SARSA is modified to select actions according to
% the PID algorithm.
% Recommended form of PID algorithm used.
% Date: 2021/04/07
%% NOTE:  The second disturbance, ore hardness, cannot be measured reliably.
%% it may be possible if there were some correlation between density and hardness, 
%% but this will likely be to dependent on the local ore composition.  Therefore, 
%% ore hardness is not considered as part of the agent state definition (unmeasurable).
% Date: 2021/03/15(par.SP_weighting*par.PSE_SP - PSE)
%%
%% A below is referring to the first simulation in the list of MIMO simulations
% Name: A.m
%%
clc;clear
%% 
% seed random number generator to ensure exploration-exploitation strategy 
% is repeatable
rng(5)

par.TV = 0; % initialize total variation statistic

par.lowerSAT = 33;
par.upperSAT = 59.4;
%% stiction parameters
par.J = 2;
par.S = 4;

%% specify whether phif parameter should be updated during validation phases
changePhif = 1;%1;
%% start parallel pool
% p = parpool('SU HPC1', 16);

%% hyperparameters for sarsa agent
par.alpha = 0.7;           % step size parameter
par.decay_for_epsilon = 1; % decay factor to multiply with epsilon
par.gamma = 0.99;          % discount factor
par.bandwidth = 0.02;      % bandwidth for binary reward function

%% training settings and SP
training.nmberOfEps = 1;         % number of episodes to use in training
training.nmberOfSteps = 10000;   % number of steps allowed per episode 
setPoint = (66.84)/100;          % initial SP
training.firstPSEVec = [(66.84/100)]'; % first CV value
training.firstStateVec = setPoint - training.firstPSEVec; % first control error
training.targetRunningReward = 5000; % minimum reward in window before stopping training (large because not intended stopping criterion)
training.decayInterval = 1;  % number of episodes before decaying probability of taking a random action

%% settings for learning curve construction
windowSize = 10; % size of windows for summing rewards
setStep_Num = 20; % number of steps per episode for curve construction

%% SP and DV_2 (rock fraction of ore fed to circuit) sampling settings
%% number of SP and DV changes per episode
rvTwo.numSP = 10;
rvTwo.numDV = 10;

%% lower and upper bounds for SP sampling (PSE)
rvTwo.lower_SP = 0.58;
rvTwo.upper_SP = 0.66;

%% lower and upper bounds for DV sampling (rock fraction fed to circuit)
rvTwo.lower_DV = 0.465 - 0.15;
rvTwo.upper_DV = 0.465 + 0.10;

%% simulation time used as range across which sampling must occur
rvTwo.simTime = training.nmberOfSteps;

%% DV_1 sampling settings (hardness of incoming ore) - SP only provided for
%% function consistency
%% number of SP and DV changes
rvOne.numSP = rvTwo.numSP;
rvOne.numDV = 10;

%% lower and upper bounds for SP sampling (PSE)
rvOne.lower_SP = rvTwo.lower_SP;
rvOne.upper_SP = rvTwo.upper_SP;

%% lower and upper bounds for DV sampling (rock hardness)
rvOne.lower_DV = 5;
rvOne.upper_DV = 7;

%% simulation time used as range across which sampling must occur
rvOne.simTime = training.nmberOfSteps;

%% controller testing settings (functionality not included in this script)
%% number of SP and DV changes per episode
rv_test_Two.numSP = 1;
rv_test_Two.numDV = 1;

%% lower and upper bounds for SP sampling (PSE)
rv_test_Two.lower_SP = 0.68;
rv_test_Two.upper_SP = 0.68;

%% lower and upper bounds for DV sampling (rock fraction fed to circuit)
rv_test_Two.lower_DV = 0.465;
rv_test_Two.upper_DV = 0.465;

%% simulation time used as range across which sampling must occur
rv_test_Two.simTime = training.nmberOfSteps;

%% number of SP and DV changes
rv_test_One.numSP = rv_test_Two.numSP;
rv_test_One.numDV = 10;

%% lower and upper bounds for SP sampling (PSE)
rv_test_One.lower_SP = rv_test_Two.lower_SP;
rv_test_One.upper_SP = rv_test_Two.upper_SP;

%% lower and upper bounds for DV sampling (rock hardness)
rv_test_One.lower_DV = 6.03;
rv_test_One.upper_DV = 6.03;

%% simulation time used as range across which sampling must occur
rv_test_One.simTime = training.nmberOfSteps;

%% filter parameters
filter.sampletime = 1;
filter.tauf = 5;

%% discretize possible states and actions (can lead to error if not sufficient to cover all encountered states)
% state bounds 
MDPstateVecLow = -0.1;           % vector for lower bounds of all state components
MDPstateVecHigh = 0.1;           % vector for upper bounds of all state components
statesResAvailable = 15;         % number of discrete states = (statesRes - 1)
% action bounds
MDPactionVecLow = 0;             % vector for lower bounds of all action components
MDPactionVecHigh = 0.15;         % vector for upper bounds of all action components
actionResAvailable = 6;          % actions available = (actionResAvailable - 1)

%% create training environment
myEnvironment = createMDP(actionResAvailable,statesResAvailable,par.bandwidth); % create RL environment SPECS

%% system parameters
%% sump control parameters
par.DB = 7.85; % ball density (t/m^3)
par.DS = 3.2; % feed ore density (t/m^3)

deltaXs = 6;   % maximum sump volume change (m^3)
deltaTot = 350; % maximum inlet flow change to sump (m^3/h)
par.damping = 8; % "estimated" damping coefficient for sump(-)

par.Kc = -0.736*(deltaTot)/deltaXs; % controller gain (1/h)
par.tau = (4*par.damping^2)/(-1*par.Kc); % integral time (h)

%% TEMPORARY TUNING PARAMETERS
par.Kc = -100;
par.tau = 20;

%% CFF starting value
par.CFF = 370.2; % CFF value obtained during phase one of model implementation

%% mill and feed parameters
par.alphaf = 0.055; % fraction fines in ore
par.alphar = 0.465; % fraction rock in ore
par.alphap = 1; % fractional power reduction per fractional reduction from maximum mill speed
par.alphaspeed = 0.712; % fraction of the critical mill speed
par.alphaphi = 0.01; % fractional change in kW/fines produced per change in fractional mill filling
par.Ps = 0.5; % parameter incorporating power-change for fraction solids in mill
par.Pv = 0.5; % parameter incorporating power-change for volume of mill filled
par.epsilonSV = 0.6; % max fraction solids by volume for static slurry
par.phib = 90; % steel abrasion factor (kWh/t)
par.phif = 29.6; % power needed per tonned of produced fines (kWh/t)
par.phir = 6.03; % rock abrasion factor (kWh/t)
par.rheaPmax = 0.57; % rheology factor associated with maximum power draw of mill
par.Pmax = 1662; % maximum mill motor power draw (kW)
par.vmill = 59.12; % volume of mill (m^3)
par.vPmax = 0.34; % fraction of mill volume filled for maximum power draw
par.Vv = 84; % driving force parameter (-)
par.XP = 0; % cross-term for maximum power draw

%% parameters for cyclone
par.alphasu = 0.87; % parameter for fraction solids in underflow
par.C1 = 0.6;   % constant for sensitivity of underflow split to total feed
par.C2 = 0.7;   % constant for sensitivity of underflow split to fraction solids in cyclone feed
par.C3 = 4;     % constant for sensitivity of underflow split to fraction solids in cyclone feed
par.C4 = 4;     % constant for sensitivity of underflow split to fraction of fines in cyclone feed solids
par.epsilonc = 129; % parameter for to coarse split (m^3/h) (helps describe the fraction of solids in the underflow)

%% mixed sump dimensions
par.L = 3.2; % length of sump (m)
par.B = 1.1; % width of sump (m)
par.center_line = 0.7; % vertical position of center line (m)
par.H_SP = 1; % SP of height of material in sump above center line (m) 
par.H = par.center_line + par.H_SP; % total height of sump contents (m)

if changePhif == 1
   %% step 
   par.phif = @(t) par.phif + (t > 10)*(37.6 - 29.6) + (t > 30)*(35.7 - 37.6) + (t > 50)*(31.5 - 35.7) + (t > 70)*(36.9 - 31.5);
   %par.phif = @(t) par.phif + (t > 1)*(37.6 - 29.6);

elseif changePhif ~= 1
    par.phif = @(t) par.phif + 0*t;
end

%% initial SS values of mill state variables (m^3)
par.XmwSS = 4.789;
par.XmsSS = 4.844;
par.XmfSS = 1.002;
par.XmrSS = 1.797;
par.XmbSS = 8.488;

%% initial SS values of sump state variables (m^3)
par.XswSS = 4.118;
par.XssSS = 1.866;
par.XsfSS = 0.3864;
par.IntErrSS = 0;

par.Int_PSE_Err_SS = 0;
%% states for control statistics
par.IAESS = 0;
par.ITAESS = 0;
%% initial values of externally changed variables
par.MIW = 4.64; % step in water to circuit (m^3/h)
par.MFS = 65.2; % step in ore to circuit (t/h)
%% old MFB
par.MFB = @(t) 5.69; % feed balls to circuit (t/h)
%% new MFB
par.SFW = 140.5; % sump feed water (m^3/h)
%% pulse function with lower peaks (2021-04-13)
maxMFB = 8;
par.MFB = @(t) (-maxMFB + maxMFB*(t < 5) + maxMFB*(t > 0)) + ...
          (-maxMFB + maxMFB*(t < 25) + maxMFB*(t > 20)) + ...
          (-maxMFB + maxMFB*(t < 45) + maxMFB*(t > 40)) + ...
          (-maxMFB + maxMFB*(t < 65) + maxMFB*(t > 60)) + ...
          (-maxMFB + maxMFB*(t < 85) + maxMFB*(t > 80));
%% specify continuous or binary reward structure
binaryReward = 1;

%% initialize variable for recording rewards within "averaging" window
learning.rewardsInWindow = 0;

%% definition of dStates
%% number of intervals + 1:
numError_intervals = 3;%11; % feedback error
numrf_intervals = 11; % DV
numPSE_intervals = 11; % SP
%% define lower bounds for discretizations
errorLow = MDPstateVecLow;
rfLow = rvTwo.lower_DV;
PSELow = rvTwo.lower_SP;
%% define "padding" for discretization
Error_padding = errorLow.*ones( (statesResAvailable - (2 + numError_intervals) ), 1)';
rf_padding = rfLow.*ones( (statesResAvailable - (2 + numrf_intervals) ), 1)';
PSE_padding = PSELow.*ones( (statesResAvailable - (2 + numPSE_intervals) ), 1)';
%% discretize the states
%% error
fineRes = linspace(MDPstateVecLow,MDPstateVecHigh,numError_intervals); % initialize discrete states for agent
dStates = [-15,Error_padding,fineRes,15]';
%% rock fraction
finerfRes = linspace(rvTwo.lower_DV,rvTwo.upper_DV,numrf_intervals);
dStates(1:1:end,2) = [0,rf_padding,finerfRes,1]';
%% PSE
finePSERes = linspace(rvTwo.lower_SP,rvTwo.upper_SP,numPSE_intervals);
dStates(1:1:end,3) = [0,PSE_padding,finePSERes,1]';

%% discrete action selections available to agent
dActions = linspace(33,66,actionResAvailable)';%linspace(35, 66, actionResAvailable)';

%% matrix of epsilon hyperparameters (num of rows in dStates X num of ff states)
par.epsilonVec = 0.1*ones( size(dStates,1),( size(dStates,2)-1 ) ); % columns for rock fraction and PSE SP ff components

%% create Q-table
Reps.action_value = zeros([(size(dStates,1)-1),(size(dStates,1)-1),...
                  (size(dStates,1)-1),(size(dActions,1)-1)],'double');
action_value_initial = Reps.action_value;

%% initialize first model coordinate
startingCoordinate = [par.XmwSS,par.XmsSS,par.XmfSS,par.XmrSS,par.XmbSS,...
                   par.XswSS,par.XssSS,par.XsfSS,par.IntErrSS,par.Int_PSE_Err_SS,par.IAESS,par.ITAESS]';
%%
par.manualDVOne_steps = [4.5,5,6,7,4];%4.5 + (7-4.5)*rand(1,5);%[6.03,6.03];%%% rock hardness manual steps
par.manualDVOne_times = 100*[10,20,30,40,50];%linspace(10,100,5);%[20,40];%% % times for rock hardness manual steps
par.manualDVTwo_steps = (0.465-0.15) + (0.465-(0.465-0.15))*rand(1,5);%[0.465,0.465];%% % ore rock fraction manual steps
par.manualDVTwo_times = 100*linspace(0,100,5);%[30,50]; %  % times for rock fraction manual steps

%%
par.manualSP_steps = 0.67:-0.01:0.58;%0.59:0.01:0.68;%0.68:-0.01:0.59;%0.70:0.01:0.75;%0.70:0.01:0.75;%linspace(0.70,0.76,5); % PSE manual SP steps
par.manualSP_times = 100*(10:10:100);%linspace(10,60,5); % times for PSE SP steps

par.manualStepFlag = 1; % flag for selecting manual steps in DVs and SP
%% PID control parameters
par.Kc_PSE = -10;%-15.05;
par.TI_PSE = 19.60;
par.TD_PSE = 4.97;
par.MFS_initial = 65.2;

par.TimeDiff = ( 1/100 );
               
%% train sarsa-LPF controller in process-based parallel environment
for episodeCntr = 1:training.nmberOfEps
        [learning_temp,temp_Reps] = train_agent(par,training,rvTwo,rvOne,learning,...
                                    dStates,dActions,setPoint,myEnvironment,...
                                    Reps,startingCoordinate,episodeCntr,filter);
        out_Reps(episodeCntr) = temp_Reps;
        out_learning(episodeCntr) = learning_temp;
end % end of episode loop

%% report control statistics
STATS.IAE = cell2mat(out_learning.agentExp{1, 1}(end,10));
STATS.ITAE = cell2mat(out_learning.agentExp{1, 1}(end,11));
STATS.TV = cell2mat(out_learning.agentExp{1, 1}(end,12));
%% approximate overall action-value table with simplified boundary conditions
for i = 1:1:training.nmberOfEps
    Reps.action_value = Reps.action_value + out_Reps(i).action_value;
end

subplot(2,1,1)
plot(linspace(1,100,training.nmberOfSteps-1),cell2mat(out_learning.agentExp{1, 1}(:,1)))
title('errors'); xlabel('steps (1/100 of an hour per step)');
ylabel('SP(T) - CV(T)')
subplot(2,1,2)
plot(cell2mat(out_learning.agentExp{1, 1}(:,4)))
title('MV(T)'); xlabel('steps (1/100 of an hour per step)');
ylabel('MFS (t/h)')
 
figure
%yyaxis left
% PSE (-)
% plot((1:1:size(cell2mat(out_learning.agentExp{1, 1}(:,1))))/1,(cell2mat(out_learning.agentExp{1, 1}(:,1))/-1) + ...
%     cell2mat(out_learning.agentExp{1, 1}(:,3)),'-.b', 'LineWidth', 1);
plot(linspace(1,100,training.nmberOfSteps-1),(cell2mat(out_learning.agentExp{1, 1}(:,1))/-1) + ...
    cell2mat(out_learning.agentExp{1, 1}(:,3)),'-.b', 'LineWidth', 1);

% title('Control error for PSE'); 
xlabel('time (h)');
ylabel('PSE (-)'); grid on; hold on

%% PSE SP
% plot((1:1:size(cell2mat(out_learning.agentExp{1, 1}(:,3))))/1,cell2mat(out_learning.agentExp{1, 1}(:,3)),'k','LineWidth', 2); hold on
plot(linspace(1,100,training.nmberOfSteps-1),cell2mat(out_learning.agentExp{1, 1}(:,3)),'k','LineWidth', 2); hold on
legend('PSE','PSE SP')

%% functions
function [learning,Reps] = train_agent(par,training,rvTwo,rvOne,...
                         learning,dStates,dActions,setPoint,myEnvironment,...
                         Reps,startingCoordinate,episodeCntr,filter)

        if learning.rewardsInWindow < training.targetRunningReward
        clear State
        clear Action
        currentTimeStamp = 1; % initialize MDP time step
        %% initialize first error state component as per initial SS
        State(currentTimeStamp) = training.firstStateVec(randi(length(training.firstStateVec))); % error already calculate earlier
        learning.prevPSE = setPoint - State(currentTimeStamp); % E(T-1) = SP(T-1) - CV(T-1) -> CV(T-1) = SP(T-1) - E(T-1)
        
        %% sample SP, DV_1 (ore hardness) and DV_2 (rock fraction in ore)
        [SP_times,SP_steps,DVTwo_times,DVTwo_steps] = generateEpRVSamples(rvTwo,training.nmberOfSteps);
        [~,~,DVOne_times,DVOne_steps] = generateEpRVSamples(rvOne,training.nmberOfSteps);
        if par.manualStepFlag == 1
            SP_times = par.manualSP_times;
            SP_steps = par.manualSP_steps;
            DVTwo_times = par.manualDVTwo_times;
            DVTwo_steps = par.manualDVTwo_steps;
            DVOne_times = par.manualDVOne_times;
            DVOne_steps = par.manualDVOne_steps;
            rvTwo.numSP = size(SP_steps,2);
            rvTwo.numDV = size(DVTwo_steps,2);
            rvOne.numDV = size(DVOne_steps,2);
        end
        for stepCntr = 1:1:training.nmberOfSteps
            %  perform training in episode if terminal state has not been
            %  achieved
            %% adjust setPoint and disturbanceValue_2 (rock fraction) as per sampling
            for SP_cntr = 1:1:rvTwo.numSP
                if SP_times(SP_cntr) == currentTimeStamp
                    setPoint = SP_steps(SP_cntr);

                end
                
            end
             %% adjust disturbanceValue_2 (ore rock fraction)
            for DV_cntr = 1:1:rvTwo.numDV
                if DVTwo_times(DV_cntr) == currentTimeStamp
                    disturbanceValue_2 = DVTwo_steps(DV_cntr); 
                    
                end
            end
            %% initialize DV 2 if it does not exist
            if exist('disturbanceValue_2') ~= 1
                disturbanceValue_2 = par.alphar;
            end
            %% adjust disturbanceValue_1 (ore hardness)
            for DV_cntr = 1:1:rvOne.numDV
                if DVOne_times(DV_cntr) == currentTimeStamp
                    disturbanceValue_1 = DVOne_steps(DV_cntr); 
                    
                end
            end
            %% initialize DV 1 if it does not exist
            if exist('disturbanceValue_1') ~= 1
                disturbanceValue_1 = par.phir;
            end
            %% 
            %% initialize states of MDP
            State_1(currentTimeStamp) = State(currentTimeStamp); % control error
            State_2(currentTimeStamp) = disturbanceValue_2; % rock fraction
            State_3(currentTimeStamp) = setPoint;   % PSE SP
            
            if State(currentTimeStamp) ~= myEnvironment.Terminal &&...
                                   currentTimeStamp < training.nmberOfSteps
                % map current MDP state to corresponding state number for
                % the agent
                %% old version
                %learning.crntAgentState = mapStatesToAgent(State(currentTimeStamp),dStates);
                %% new version
                [learning.crntAgentState_1,learning.crntAgentState_2,...
                    learning.crntAgentState_3] = mapStatesToAgent(State_1(currentTimeStamp),...
                    State_2(currentTimeStamp),State_3(currentTimeStamp),dStates);
                % let agent select the next action 
                learning.crntAgentAction = selectAction(Reps,par,learning.crntAgentState_1,...
                    learning.crntAgentState_2,learning.crntAgentState_3,...
                    myEnvironment);    
                % action (true MV control action to take)
                Action(currentTimeStamp) = mapToMDP(learning.crntAgentAction,dActions,Reps);
                % filter agent's selected action (initialize with first
                % true agent output)
                if currentTimeStamp > 1
                    previousSmoothed = Action(currentTimeStamp - 1);
                    crntModelCoordinate = nxtModelCoordinate;
                elseif currentTimeStamp == 1
                    previousSmoothed = Action(currentTimeStamp);
                    crntModelCoordinate = startingCoordinate;
                end
                %% store unfiltered actions
                learning.discreteSelections{1,1}{stepCntr,1} = Action(currentTimeStamp);
                
                %% PID control calculation
                if currentTimeStamp == 1
                    Action(currentTimeStamp) = par.MFS_initial;
                    par.dCVdt = 0;
                elseif currentTimeStamp ~= 1
                    Action(currentTimeStamp) = par.Kc_PSE*(100*State_1(currentTimeStamp) + (1/par.TI_PSE)*crntModelCoordinate(10) - par.TD_PSE*par.dCVdt) + par.MFS_initial;
                    
                    %% include saturation 
                    if Action(currentTimeStamp) <= par.lowerSAT
                        Action(currentTimeStamp) = par.lowerSAT;
                    elseif Action(currentTimeStamp) >= par.upperSAT
                        Action(currentTimeStamp) = par.upperSAT;
                    end
                end
               
                if currentTimeStamp ~= 1
                    crntPSE = learning.nxtPSE;
                end
                %% settings for stiction model of Choudhury et. al.
                MV_low_bound = 0;   % lower bound of MV
                MV_high_bound = 100;%1000; % upper bound of MV
                S = par.S;%0; % deadband + stickband
                J = par.J;%0; % slip jump
                %% stiction calculation starts
                %% initialize actions in MV units
                if currentTimeStamp == 1 
                    crntAction = Action(currentTimeStamp); 
                    nxtAction = crntAction; 

                elseif currentTimeStamp > 1
                    crntAction = Action(currentTimeStamp - 1);
                    nxtAction = Action(currentTimeStamp);

                end
                %% stiction model applied to valve output
                if currentTimeStamp == 1
                    xss = 0;    % initialize memory variable for output signal when valve becomes stuck
                    % previous output as a % of the final element range
                    MV_output_previous = ( (crntAction - MV_low_bound)/(MV_high_bound - MV_low_bound) )*100;
                    MV_incoming_present = crntAction; % initialize present MV signal
                    MV_incoming_previous = crntAction;% initialize previous MV signal
                    vnew_prev = 0; % initialize previous local gradient of control signal
                    I = 0; % initialize flag for valve stuck during trajectory
                else

                    MV_output_previous = MV_output; % previous output as a % of the final element range
                    MV_incoming_present = nxtAction; % present MV signal
                    MV_incoming_previous = crntAction; % previous MV signal
                    vnew_prev = vnew; % previous local gradient of control signal

                end
                %% stiction calculation, MV_output is a % of MV range
                [MV_output,I,xss,vnew,~] = stictionModel(J,S,xss,...
                                           MV_output_previous,I,...
                                           MV_incoming_present,...
                                           MV_incoming_previous,...
                                           MV_low_bound,MV_high_bound,...
                                           vnew_prev);
                % convert MV_output at current timestamp to actual MV value
                crntAction = (MV_output/100)*(MV_high_bound - MV_low_bound)+ MV_low_bound;
                % apply stiction model output as current action
                Action(currentTimeStamp) = crntAction;
                %% end of stiction calculation
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % update total variation statistic
                if currentTimeStamp > 1
                    par.TV = par.TV + abs(Action(currentTimeStamp) - Action(currentTimeStamp - 1));
                end
                % simulate model to obtain next CV 
                [nxtModelCoordinate,learning.nxtPSE] = simulateMDP(currentTimeStamp,...
                                                       crntModelCoordinate,...
                                                       Action(currentTimeStamp),...
                                                       par,disturbanceValue_1,...
                                                       disturbanceValue_2,setPoint);
                if currentTimeStamp == 1
                    % nxtPSE = learning.nxtPSE;
                    crntPSE = learning.nxtPSE;
                
                end
                
                par.dCVdt = (learning.nxtPSE - crntPSE)/( par.TimeDiff );
              
                
                learning.nxtState_1 = setPoint - learning.nxtPSE;   % error component
                % only next state components RESULTING from interaction
                % between agent and RL environment taken into account
                learning.nxtState_2 = State_2(currentTimeStamp);
                learning.nxtState_3 = State_3(currentTimeStamp);
                
                [learning.nxtAgentState_1,learning.nxtAgentState_2,...
                    learning.nxtAgentState_3] = mapStatesToAgent(learning.nxtState_1,...
                                                learning.nxtState_2,...
                                                learning.nxtState_3,dStates); 
                     
                % store training information
                learning.agentExp{1,1}{stepCntr,1} = State_1(currentTimeStamp);
                learning.agentExp{1,1}{stepCntr,2} = State_2(currentTimeStamp);
                learning.agentExp{1,1}{stepCntr,3} = State_3(currentTimeStamp);
                learning.agentExp{1,1}{stepCntr,4} = Action(currentTimeStamp);
                learning.agentExp{1,1}{stepCntr,9} = crntModelCoordinate(10);
                
                learning.agentExp{1,1}{stepCntr,10} = crntModelCoordinate(11); %IAE for PSE-MFS
                learning.agentExp{1,1}{stepCntr,11} = crntModelCoordinate(12); %ITAE for PSE-MFS
                learning.agentExp{1,1}{stepCntr,12} = par.TV;
                %% store ore hardness values (DV not part of states)
                learning.hardness_Data{1,1}{stepCntr,1} = disturbanceValue_1;
                         
            % shift time step T <- (T+1)
            currentTimeStamp = currentTimeStamp + 1;
            % shift state S(T) <- S(T+1)
            State(currentTimeStamp) = learning.nxtState_1;
            learning.prevPSE = learning.nxtPSE;
            
            end % end loop for non-terminal state
        display(stepCntr)
        end % end loop for steps
       
        
        end % end rewards in window if-statement
    % Store number of non-terminal entries of each training episode
    optimalityCntr(episodeCntr,1) = currentTimeStamp;
    fprintf('%d\n',episodeCntr) % display episode number to give indication of training progress
    
    if episodeCntr < training.nmberOfEps && mod(episodeCntr,training.decayInterval) == 0
       % par.epsilon = par.epsilon*par.decay_for_epsilon; % decay probability for taking random action 
       %% decay probability of a random action for rock fraction state component
       par.epsilonVec(learning.crntAgentState_2,1) = par.epsilonVec(learning.crntAgentState_2,1)*par.decay_for_epsilon;
       %% decay probability of a random action for PS SP state component
       par.epsilonVec(learning.crntAgentState_3,2) = par.epsilonVec(learning.crntAgentState_3,2)*par.decay_for_epsilon;
    end

end

function [nxtModelCoordinate,nxtPSE] = simulateMDP(currentTimeStamp,...
                                       crntModelCoordinate,...
                                       a,par,disturbanceValue_1,disturbanceValue_2,setPoint)
    %% solve Hulbert model equations
    % Xmw, Xms, Xmf, Xmr, Xmb, Xsw, Xss, Xsf, IE(t), IE_PSE(t), IAE_PSE,
    % ITAE_PSE
    % sump delays larger to accommodate numerical behaviour arising when changing SFW
    par.modLags = [30/3600,30/3600,30/3600,30/3600,30/3600,0.1,0.1,0.1,0.1]; 
    %% old
    % start = currentTimeStamp;
    % stop = currentTimeStamp + 1;
    %% new
    start = currentTimeStamp*par.TimeDiff;%(1/100);%*(1/60);
    stop = ( currentTimeStamp + 1 )*par.TimeDiff;%(1/100);%*(1/60);
    %% pass starting state to history function through "par" structure
    par.XmwSS = crntModelCoordinate(1);
    par.XmsSS = crntModelCoordinate(2);
    par.XmfSS = crntModelCoordinate(3);
    par.XmrSS = crntModelCoordinate(4);
    par.XmbSS = crntModelCoordinate(5);
    par.XswSS = crntModelCoordinate(6);
    par.XssSS = crntModelCoordinate(7);
    par.XsfSS = crntModelCoordinate(8);
    par.IntErrSS = crntModelCoordinate(9);
    par.IntErr_PSE_SS = crntModelCoordinate(10);
    
    par.IAESS = crntModelCoordinate(11);
    par.ITAESS = crntModelCoordinate(12);
    %% pass action to model function through "par" structure
    par.MFS = a;
    %% pass DV value(-s) to model function through "par" structure
    par.phir = disturbanceValue_1;
    par.alphar = disturbanceValue_2;
    %% simulate grinding circuit model
    par.PSE_SP = setPoint;
    model_with_delays = dde23( @circuit_model, par.modLags, @history, [start stop], [], par );
    time = linspace(start,stop,100);
    model_high_res = deval( model_with_delays, time );
    nxtXmw = model_high_res(1,end);
    nxtXms = model_high_res(2,end);
    nxtXmf = model_high_res(3,end);
    nxtXmr = model_high_res(4,end);
    nxtXmb = model_high_res(5,end);
    nxtXsw = model_high_res(6,end);
    nxtXss = model_high_res(7,end);
    nxtXsf = model_high_res(8,end);
    nxtIntErr = model_high_res(9,end);
    nxt_PSE_Int_Err = model_high_res(10,end);
    
    nxt_IAE = model_high_res(11,end);
    nxt_ITAE = model_high_res(12,end);
    %% store next model coordinate in a vector
    nxtModelCoordinate = [nxtXmw,nxtXms,nxtXmf,nxtXmr,nxtXmb,nxtXsw,nxtXss,nxtXsf,nxtIntErr,nxt_PSE_Int_Err,nxt_IAE,nxt_ITAE]';
    
    %% additional circuit variables
    for cntr = 1:1:length(time)
        Xmw(cntr) = model_high_res(1,cntr); Xms(cntr) = model_high_res(2,cntr); Xmf(cntr) = model_high_res(3,cntr);
        Xmr(cntr) = model_high_res(4,cntr); Xmb(cntr) = model_high_res(5,cntr);
    
        Xsw(cntr) = model_high_res(6,cntr); Xss(cntr) = model_high_res(7,cntr); 
        Xsf(cntr) = model_high_res(8,cntr); IE(cntr) = model_high_res(9,cntr);
    
        phi(cntr) = ( max(min(1 - ( (1/par.epsilonSV) - 1)*( Xms(cntr)/Xmw(cntr) ), 1), 0) )^0.5; % rheology factor

        Vmwo(cntr) = par.Vv*phi(cntr)*Xmw(cntr)*( Xmw(cntr)/( Xms(cntr) + Xmw(cntr) ) );    % outlet flowrate of water
        Vmso(cntr) = par.Vv*phi(cntr)*Xmw(cntr)*( Xms(cntr)/( Xms(cntr) + Xmw(cntr) ) );    % outlet flowrate of solids
        Vmfo(cntr) = par.Vv*phi(cntr)*Xmw(cntr)*( Xmf(cntr)/( Xms(cntr) + Xmw(cntr) ) );   % outlet flowrate of fines
    
        LOAD(cntr) = Xmw(cntr) + Xmr(cntr) + Xms(cntr) + Xmb(cntr);
        fraction_filled(cntr) = LOAD(cntr)/par.vmill;
        %% old
        % Zr(cntr) = (LOAD(cntr)/(par.vmill*par.vPmax)) - 1;
        % Zx(cntr) = (phi(cntr)/par.rheaPmax) - 1;
        %% new
        Zx(cntr) = (LOAD(cntr)/(par.vmill*par.vPmax)) - 1;
        Zr(cntr) = (phi(cntr)/par.rheaPmax) - 1;
        Pmill(cntr)  = par.Pmax*( 1 - par.Pv*Zx(cntr)^2 - par.Ps*Zr(cntr)^2 )*par.alphaspeed^par.alphap;
    
        RC(cntr) = ( (Pmill(cntr)*phi(cntr))/(par.DS*par.phir) )*( Xmr(cntr)/( Xmr(cntr) + Xms(cntr) ) );
        FP(cntr) = Pmill(cntr)/( par.DS*par.phif(0)*(1 + par.alphaphi*((LOAD(cntr)/par.vmill) - par.vPmax ) ) );
        BC(cntr) = ( ( Pmill(cntr)*phi(cntr) )/par.phib)*( Xmb(cntr)/( par.DS*( Xmr(cntr) + Xms(cntr) ) + par.DB*Xmb(cntr) ) );
   
        SVOL(cntr) = Xsw(cntr) + Xss(cntr);
        err(cntr) = par.H*par.L*par.B - SVOL(cntr); % control error for sump volume
        CFF(cntr) = par.CFF + par.Kc*( err(cntr) + (1/par.tau)*IE(cntr) );
        Vswo(cntr) = CFF(cntr)*( Xsw(cntr)/SVOL(cntr) );
        Vsso(cntr) = CFF(cntr)*( Xss(cntr)/SVOL(cntr) );
        Vsfo(cntr) = CFF(cntr)*( Xsf(cntr)/SVOL(cntr) );
    
        CFD(cntr) = ( Xsw(cntr) + par.DS*Xss(cntr) )/( Xsw(cntr) + Xss(cntr) );
    
        %% cyclone model equations
        % inlet flowrates
        Vcsi(cntr) = Vsso(cntr);
        Vcfi(cntr) = Vsfo(cntr);
        Vcwi(cntr) = Vswo(cntr);
        Vcci(cntr) = Vcsi(cntr) - Vcfi(cntr);
   
        % cyclone outlets
        %Fi = Vcsi/par.CFF;
        Fi(cntr) = Vcsi(cntr)/CFF(cntr);
        Pi(cntr) = Vcfi(cntr)/Vcsi(cntr);
        Vccu(cntr) = Vcci(cntr)*( 1 - par.C1*exp(-1*CFF(cntr)/par.epsilonc) )*( 1 - (Fi(cntr)/par.C2).^par.C3 )*...
                   ( 1 - Pi(cntr)^par.C4 );
        Fu(cntr) = 0.6 - ( 0.6 - Fi(cntr) )*exp( -1*(Vccu(cntr))/(par.alphasu*par.epsilonc) );
        Vcwu(cntr) = Vcwi(cntr)*( Vccu(cntr) - Fu(cntr)*Vccu(cntr) )/( Fu(cntr)*Vcwi(cntr) + Fu(cntr)*Vcfi(cntr) - Vcfi(cntr) );
        Vcfu(cntr) = Vcfi(cntr)*( Vccu(cntr) - Fu(cntr)*Vccu(cntr) )/( Fu(cntr)*Vcwi(cntr) + Fu(cntr)*Vcfi(cntr) - Vcfi(cntr) );
        Vcsu(cntr) = Vccu(cntr) + Vcfu(cntr);
        Vcfo(cntr) = Vcfi(cntr) - Vcfu(cntr);
        Vcco(cntr) = Vcci(cntr) - Vccu(cntr);
        Vcso(cntr) = Vcsi(cntr) - Vcsu(cntr);
        Vcwo(cntr) = Vcwi(cntr) - Vcwu(cntr);
        PSE(cntr) = Vcfo(cntr)/( Vcco(cntr) + Vcfo(cntr) );
   
    end
    
    nxtPSE = PSE(cntr);
end

% history function for dde solver
function y = history(~,par)
    y = [par.XmwSS,par.XmsSS,par.XmfSS,par.XmrSS,par.XmbSS,par.XswSS,...
        par.XssSS,par.XsfSS,par.IntErrSS,par.Int_PSE_Err_SS,par.IAESS,par.ITAESS];
end

% function to model grinding circuit dynamics
function dydt = circuit_model(t,y,y_lagged,par)
    % y(1) -> Xmw
    % y(2) -> Xms
    % y(3) -> Xmf
    % y(4) -> Xmr
    % y(5) -> Xmb
    
    % y(6) -> Xsw
    % y(7) -> Xss
    % y(8) -> Xsf
    % y(9) -> Integral error for sump volume
    
    % y(10) -> Integral error for PSE loop
    % y(11) -> IAE for PSE loop
    % y(12) -> ITAE for PSE loop
    %% initialize lags
    yl1 = y_lagged(:,1);
    yl2 = y_lagged(:,2);
    yl3 = y_lagged(:,3);
    yl6 = y_lagged(:,6);
    yl7 = y_lagged(:,7);
    yl8 = y_lagged(:,8);
    
    %% feeder calculations
    Vfwo = par.MIW;
    Vfso = (par.MFS/par.DS)*(1 - par.alphar);
    Vffo = (par.MFS/par.DS)*par.alphaf;
    Vfro = (par.MFS/par.DS)*par.alphar;
    Vfbo = (par.MFB(t)/par.DB);
    
    %% mill model equations
    phi = ( max(min(1 - ( (1/par.epsilonSV) - 1)*( y(2)/y(1) ), 1), 0) )^0.5; % rheology factor
    Vmwo = par.Vv*phi*y(1)*( y(1)/( y(2) + y(1) ) );    % outlet flowrate of water
    Vmso = par.Vv*phi*y(1)*( y(2)/( y(2) + y(1) ) );    % outlet flowrate of solids
    Vmfo = par.Vv*phi*y(1)*( y(3)/( y(2) + y(1) ) );    % outlet flowrate of fines
    
    LOAD = y(1) + y(4) + y(2) + y(5);          % volume of material in mill
    
    % calculations for mill power draw
    %% old
    % Zr = (LOAD/(par.vmill*par.vPmax)) - 1;
    % Zx = (phi/par.rheaPmax) - 1;
    %% new
    Zx = (LOAD/(par.vmill*par.vPmax)) - 1;
    Zr = (phi/par.rheaPmax) - 1;
    Pmill  = par.Pmax*( 1 - par.Pv*Zx^2 - par.Ps*Zr^2 )*par.alphaspeed^par.alphap;
    
    % breakage functions
    RC = ( (Pmill*phi)/(par.DS*par.phir) )*( y(4)/( y(4) + y(2) ) );
    FP = Pmill/( par.DS*par.phif(t)*(1 + par.alphaphi*((LOAD/par.vmill) - par.vPmax ) ) );
    BC = ( ( Pmill*phi )/par.phib)*( y(5)/( par.DS*( y(4) + y(2) ) + par.DB*y(5) ) );
    
    %% calculate delayed flowrates from SAG mill 
    phi_delay = ( max(min(1 - ( (1/par.epsilonSV) - 1)*( yl2(2)/yl1(1) ), 1), 0) )^0.5;
    Vmwo_delay = par.Vv*phi_delay*y(1)*( yl1(1)/( yl2(2) + yl1(1) ) );
    Vmso_delay = par.Vv*phi_delay*yl1(1)*( yl2(2)/( yl2(2) + yl1(1) ) );
    Vmfo_delay = par.Vv*phi_delay*yl1(1)*( yl3(3)/( yl2(2) + yl1(1) ) );
    Cms_delay = Vmso_delay/( Vmso_delay + Vmwo_delay ); % delayed solids concentration from mill
    Cmf_delay = Vmfo_delay/( Vmso_delay + Vmwo_delay ); % delayed fines concentration from mill
    Vmfo_adjusted = Cmf_delay*( Vmso + Vmwo );
    Vmso_adjusted = Cms_delay*( Vmso + Vmwo );
    Vmwo_adjusted = ( Vmso + Vmwo )*( 1 - Cms_delay );
    
    % adjust mill outlet flowrates to incorporate time delays 
    Vmfo = Vmfo_adjusted;
    Vmso = Vmso_adjusted;
    Vmwo = Vmwo_adjusted;
    
    %% sump model equations
    % volume calculations and cyclone feed flowrate
    SVOL = y(6) + y(7);
    err = par.H*par.L*par.B - SVOL; % control error for sump volume
    dIntERRdt = err;                % error must be integrated for PI control
    CFF = par.CFF + par.Kc*( err + (1/par.tau)*y(9) ); % PI control for cyclone feed
    
    % outlet flowrates
    Vswo = CFF*( y(6)/SVOL );
    Vsso = CFF*( y(7)/SVOL );
    Vsfo = CFF*( y(8)/SVOL );
    
    % differential equations
    dXswdt = Vmwo - Vswo + par.SFW;
    dXssdt = Vmso - Vsso;
    dXsfdt = Vmfo - Vsfo;
    
    %% calculate delayed flowrates from sump
    SVOL_delay = yl6(6) + yl7(7);
    Vswo_delay = CFF*( yl6(6)/SVOL_delay );
    Vsso_delay = CFF*( yl7(7)/SVOL_delay );
    Vsfo_delay = CFF*( yl8(8)/SVOL_delay );
    Css_delay = Vsso_delay/( Vsso_delay + Vswo_delay );
    Csf_delay = Vsfo_delay/( Vsso_delay + Vswo_delay );
    Vsso_adjusted = Css_delay*( Vsso + Vswo );
    Vsfo_adjusted = Csf_delay*( Vsso + Vswo );
    Vswo_adjusted = ( Vsso + Vswo )*( 1 - Css_delay );
    
    % adjust mill outlet flowrates to incorporate time delays 
    Vsso = Vsso_adjusted;
    Vsfo = Vsfo_adjusted;
    Vswo = Vswo_adjusted; 
    
    %% cyclone model equations
    % inlet flowrates
    Vcsi = Vsso;
    Vcfi = Vsfo;
    Vcwi = Vswo;
    Vcci = Vcsi - Vcfi;
   
    % cyclone outlets
    %Fi = Vcsi/par.CFF;
    Fi = Vcsi/CFF;
    Pi = Vcfi/Vcsi;
    Vccu = Vcci*( 1 - par.C1*exp(-1*CFF/par.epsilonc) )*( 1 - (Fi/par.C2).^par.C3 )*...
           ( 1 - Pi^par.C4 );
    Fu = 0.6 - ( 0.6 - Fi )*exp( -1*(Vccu)/(par.alphasu*par.epsilonc) );
    Vcwu = Vcwi*( Vccu - Fu*Vccu )/( Fu*Vcwi + Fu*Vcfi - Vcfi );
    Vcfu = Vcfi*( Vccu - Fu*Vccu )/( Fu*Vcwi + Fu*Vcfi - Vcfi );
    Vcsu = Vccu + Vcfu;
    Vcfo = Vcfi - Vcfu;
    Vcco = Vcci - Vccu;
    Vcso = Vcsi - Vcsu;
    Vcwo = Vcwi - Vcwu;
    PSE = Vcfo/( Vcco + Vcfo );
    
    %% solve mill material balances with delayed cyclone underflow
    dXmwdt = ( Vfwo + Vcwu ) - Vmwo;
    dXmsdt = ( Vfso + Vcsu ) - Vmso + RC;
    dXmfdt = ( Vffo + Vcfu ) - Vmfo + FP;
    dXmrdt = Vfro - RC;
    dXmbdt = Vfbo - BC;
    
    %display(t) % display time of solver to track solution progress
    
    %% PSE control error
    dInt_PSE_Err_dt = 100*(par.PSE_SP - PSE);
    error = dInt_PSE_Err_dt;
    % - par.TD_PSE*( dCVdt )
    par.MFS = par.Kc_PSE*(error + (1/par.TI_PSE)*y(10) - par.TD_PSE*( par.dCVdt )) + par.MFS_initial;
    if par.MFS <= par.lowerSAT
        par.MFS = par.lowerSAT;
        dInt_PSE_Err_dt = 0; % prevent reset windup
    elseif par.MFS >= par.upperSAT
        par.MFS = par.upperSAT;
        dInt_PSE_Err_dt = 0; % prevent reset windup
    end
    %% calculate integral control statistics
    dIAEdt = abs(par.PSE_SP - PSE);
    dITAEdt = t*abs(par.PSE_SP - PSE);
    %% vector of derivatives
    dydt = [dXmwdt,dXmsdt,dXmfdt,dXmrdt,dXmbdt,dXswdt,dXssdt,dXsfdt,dIntERRdt,dInt_PSE_Err_dt,dIAEdt,dITAEdt]';
    
end

% function to create the parameters for the RL environment's model
function rlEnv = createMDP(actionRes,statesRes,bandwidth)

    rlEnv.binReward = @(controlError) -1 + 1*(controlError < bandwidth) + 1*(controlError > -1*bandwidth);
    rlEnv.numberOfActions = actionRes - 1;
    rlEnv.numberOfStates = statesRes - 1;
    rlEnv.Terminal = -2; 

end

% function to map agent's action to the MV output of the MDP
function MDPAction = mapToMDP(agentActionIndex,dActions,Reps)

   MDPAction = dActions(agentActionIndex);

end

%  digital low-pass filter (chapter 12 of MARLIN)
function smoothedAction = digitalFilter(sampleTime,tauf,previousSmoothed,currentAction)
    A = exp(-1*sampleTime/tauf);
    % time domain digital filter calculation
    smoothedAction = A*previousSmoothed + (1-A)*currentAction;
    
end

% function to create vector of sampled DVs, SPs and times for these
% sampled changes
function [SP_times,SP_steps,DV_times,DV_steps] = generateEpRVSamples(rv,upperTimeBound)
    %% generate SP sampling data
    SP_times = sort( ceil(1 + upperTimeBound*rand(1,rv.numSP)) );%sort(randperm(rv.simTime,rv.numSP));
    for SP_cntr = 1:1:(rv.numSP)
        SP_steps(SP_cntr) = rv.lower_SP + (rv.upper_SP - rv.lower_SP)*rand(1,1);
        
    end

    %% generate DV sampling data
    DV_times = sort( ceil(1 + upperTimeBound*rand(1,rv.numDV)) );%sort(randperm(rv.simTime,rv.numDV));
    for DV_cntr = 1:1:(rv.numDV)
        DV_steps(DV_cntr) = rv.lower_DV + (rv.upper_DV - rv.lower_DV)*rand(1,1);
        
    end
    
end

%% mapping of states to agent
function [agentState_1,agentState_2,agentState_3] = mapStatesToAgent(state_1,state_2,state_3,dStates)
    % assign coded state for PSE error
    for cntr = 1:1:(size(dStates,1)-1)
        if state_1 > dStates(cntr,1) && state_1 <= dStates(cntr+1,1)
            agentState_1 = cntr;
        else
        end
    end
    % assign coded state for rock fraction
    for cntr = 1:1:(size(dStates,1)-1)
        if state_2 > dStates(cntr,2) && state_2 <= dStates(cntr+1,2)
            agentState_2 = cntr;
        else
        end
    end
    % assign coded state for PSE SP
    for cntr = 1:1:(size(dStates,1)-1)
        if state_3 > dStates(cntr,3) && state_3 <= dStates(cntr+1,3)
            agentState_3 = cntr;
        else
        end
    end
    
    if exist('agentState_1') ~= 1
        display(state_1)
    end
    
end

%% select action as per epsilon-greedy exploration strategy
function Action = selectAction(Reps,par,state_1,state_2,state_3,myEnvironment)
   t = rand(1);
   %% use state component three for probability of a random action (epsilon)
   %% seeing as epsilon is kept constant any state component could be used during
   %% the parallel training episodes
   if t <= par.epsilonVec(state_3,2)
       % take random action
       Action = randi(myEnvironment.numberOfActions);
   elseif t > par.epsilonVec(state_3,2)
       % take greedy action
       vec = Reps.action_value(state_1,state_2,state_3,:);
       index = find( ismember( vec(:),max( vec(:) ) ) );
       [~,~,~,Action] = ind2sub( size( vec(:,:,:,:) ), index );
       
   end
   
   % tie breaking
   if size(Action,1) > 1
       Action = randi(myEnvironment.numberOfActions);
   end
   
end

%% function to model valve stiction
function[MV_output,I,xss,vnew,MV_Percentage_present] = stictionModel(J,S,...
                                                       xss,MV_output_previous,...
                                                       I,MV_incoming_present,...
                                                       MV_incoming_previous,...
                                                       MV_low_bound,...
                                                       MV_high_bound,vnew_prev)

%% saturated condition and scaling of present MV input signal
if MV_incoming_present <= MV_low_bound
    MV_Percentage_present = 0;
    MV_output = 0;

elseif MV_incoming_present >= MV_high_bound
    MV_Percentage_present = 100;
    MV_output = 100;

elseif ( MV_incoming_present < MV_high_bound ) && ( MV_low_bound < MV_incoming_present )
    MV_Percentage_present = ( (MV_incoming_present - MV_low_bound)/(MV_high_bound - MV_low_bound) )*100;

end

%% scaling of previous MV input signal
if MV_incoming_previous <= MV_low_bound
    MV_Percentage_previous = 0;

elseif MV_incoming_previous >= MV_high_bound
    MV_Percentage_previous = 100;

elseif ( MV_incoming_previous < MV_high_bound ) && ( MV_low_bound < MV_incoming_previous )
    MV_Percentage_previous = ( (MV_incoming_previous - MV_low_bound)/(MV_high_bound - MV_low_bound) )*100;

end

vnew = (MV_Percentage_present - MV_Percentage_previous)/(1); % gradient of incoming control signal (consistently i.t.o. MDP time)
%% 
if MV_Percentage_present > 0 && MV_Percentage_present < 100
    vold = vnew_prev;
    vnew = (MV_Percentage_present - MV_Percentage_previous)/(1); % gradient of incoming control signal (consistently i.t.o. MDP time)
    if sign(vnew) == sign(vold)
        if I ~= 1 % if not stuck during valve transient behaviour
            % absolute difference between current signal and previous stuck signal
            DIFF = abs(MV_Percentage_present - xss); 
            if DIFF > S
                % adjust output from valve if DFF is greater than
                % (dead-band and stick band), i.e. valve is at
                % beginning of trajectory
                MV_output = MV_Percentage_present - sign(vnew)*( (S - J)/2 );
            else
                % keep output at stuck value
                MV_output = MV_output_previous;
            end
        elseif I == 1 % if in stuck condition during moving phase
            DIFF = abs(MV_Percentage_present - xss);
            if DIFF > J % if DIFF is greater than slip jump
                I = 0; % remove flag for being stuck during moving phase
                % adjust MV_output
                MV_output = MV_Percentage_present - sign(vnew)*( (S - J)/2 );
            else
                % keep input at the same value
                MV_output = MV_output_previous;
            end
        end
    elseif sign(vnew) ~= sign(vold) % if direction of valve movement changes
        if sign(vnew) == 0 % if valve reaches a stop position during a transient
            I = 1; % indicate this stop condition with flag
            xss = MV_Percentage_previous;%MV_incoming_previous; % update memory variable for previous stuck position
            MV_output = MV_output_previous; % keep MV output signal constant
        else
            % do the same for a stuck position not reached during the
            % valve's moving phase
            xss = MV_Percentage_previous;%MV_incoming_previous;
            MV_output = MV_output_previous;
        end
    end

end


end