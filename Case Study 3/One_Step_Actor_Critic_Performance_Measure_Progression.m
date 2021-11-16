instances_per_update = 100; % number of instances added for each point on graph
for outer_Cntr = 1:1:(20000/100)
    
%% short description
% One-step AC applied to PSE-MFS loop
% State:  A vector consisting of the following components that may have
% different values:
% S = [PSE error, inlet rock fraction, PSE SP]
% Action: In this script, the sump level is controlled by a separate PI
% controller. Therefore, the agent's action is represented by a single scalar that
% may have different values and represents MFS.
%% Note: Need to load a predefined set of parameter vectors:
%  load structure named "out_Reps" which contains vector 
%  instances named "Weights" and "Theta"
% Date: 2021/08/14
%%

%% 
%tic
% seed random number generator to ensure exploration-exploitation strategy 
% is repeatable
rng(5)

par.TV = 0; % initialize total variation statistic
%% stiction parameters
par.J = 0;%2;
par.S = 0;%4;
%% specify whether phif parameter should be updated during validation phases
changePhif = 1;%1;

par.bandwidth = 0.02;      % bandwidth for binary reward function

%% Populate centers and variances to use for actor and critic models
numS = 5; % number of state components in each direction
par.numA = 5; % number of action components in each direction
error_low_bound = -1; % lower bound for error RBF placement
error_high_bound = 1; % upper bound for error RBF placement
rockfrac_low_bound = -1; % lower bound for rock fraction
rockfrack_high_bound = 1;% upper bound for rock fraction
PSE_SP_low_bound = -1;   % lower bound for PSE SP
PSE_SP_high_bound = 1;   % upper bound for PSE SP
[policyState_1_Centers,policyState_2_Centers,policyState_3_Centers,...
    policy_MV_Centers] = Policy_RBF_placement(numS,par.numA,error_low_bound,...
    error_high_bound,rockfrac_low_bound,rockfrack_high_bound,...
    PSE_SP_low_bound,PSE_SP_high_bound);

%% 2021-05-30
% Initialize true lower bounds for mapping calculations
par.true_error_low_bound = -0.1;
par.true_error_high_bound = 0.1;
par.true_rockfrac_low_bound = 0.315;
par.true_rockfrac_high_bound = 0.565;
par.true_PSE_SP_low_bound = 0.58;
par.true_PSE_SP_high_bound = 0.66;

%% specify variances for policy
%policyVarianceVector = repmat([1,1,1],1,(numS*numS*numA));
policyVarianceVector = repmat(0.1*ones(1,numS),(numS*numS*par.numA));%repmat(0.2*ones(1,numS),(numS*numS*par.numA));%repmat(0.01*ones(1,numS),(numS*numS*par.numA));
% Set parameters for policy basis functions
[par.policyCenters,par.policyVariance] = createPolicyRBFSpecs(policyState_1_Centers,...
                               policyState_2_Centers,policyState_3_Centers,policy_MV_Centers,policyVarianceVector);
                               

%% place RBFs for state-value function
Critic_numS = 4;
[Critic_State_1_Centers,Critic_State_2_Centers,Critic_State_3_Centers,~] = Critic_RBF_placement(Critic_numS,error_low_bound,...
    error_high_bound,rockfrac_low_bound,rockfrack_high_bound,...
    PSE_SP_low_bound,PSE_SP_high_bound);

uniformCriticVariance_critic = 0.1;%0.2;
% Set parameters for critic basis functions
[par.criticCenters,par.criticVariance] = createCriticRBFSpecs(Critic_State_1_Centers,...
                                   Critic_State_2_Centers,Critic_State_3_Centers,...
                                   uniformCriticVariance_critic);
%% Actor and critic initializations
%% changed alphas and gamma on 2021-04-20
par.alphaActor = 0.05;  % Learning rate for actor parameters
par.alpha = 0.7;        % Learning rate for critic parameters
%% Need to load a saved set of vectors
%Reps.Weights = 0.01*rand(size(par.criticCenters,1),1); % Initalize critic paramaters
%Reps.Theta = 0.01*rand(size(par.policyCenters,1),1); % Initialize actor parameters
%%
par.gamma = 0.99;           % Discount factor
% store initial weightings
% WInitial = Reps.Weights;
% ThetaInitial = Reps.Theta;

%%
%%
%% END UPDATE
%% training settings and SP
training.nmberOfEps = 1;%1500;%500;%1500; %2000;           % number of episodes to use in training
training.nmberOfSteps = 100;%500;      % number of steps allowed per episode 
setPoint = (66.84)/100;%(60)/100;      % initial SP
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

%% controller testing settings
%% number of SP and DV changes per episode
rv_test_Two.numSP = 10;
rv_test_Two.numDV = 10;

%% lower and upper bounds for SP sampling (PSE)
rv_test_Two.lower_SP = 0.58;
rv_test_Two.upper_SP = 0.66;

%% lower and upper bounds for DV sampling (rock fraction fed to circuit)
rv_test_Two.lower_DV = 0.465 - 0.15;
rv_test_Two.upper_DV = 0.465 + 0.10;

%% simulation time used as range across which sampling must occur
rv_test_Two.simTime = training.nmberOfSteps;

%% number of SP and DV changes
rv_test_One.numSP = rv_test_Two.numSP;
rv_test_One.numDV = 10;

%% lower and upper bounds for SP sampling (PSE)
rv_test_One.lower_SP = rv_test_Two.lower_SP;
rv_test_One.upper_SP = rv_test_Two.upper_SP;

%% lower and upper bounds for DV sampling (rock hardness)
rv_test_One.lower_DV = 5;
rv_test_One.upper_DV = 7;

%% simulation time used as range across which sampling must occur
rv_test_One.simTime = training.nmberOfSteps;

%% filter parameters
filter.sampletime = 1;
filter.tauf = 5;%1;%0.5; %5;

%% discretize possible states and actions (can lead to error if not sufficient to cover all encountered states)
% state bounds 
MDPstateVecLow = -0.1;           % vector for lower bounds of all state components
MDPstateVecHigh = 0.1;           % vector for upper bounds of all state components
statesResAvailable = 15;        % number of discrete states = (statesRes - 1)
% action bounds
MDPactionVecLow = 0;            % vector for lower bounds of all action components
MDPactionVecHigh = 0.15;        % vector for upper bounds of all action components
actionResAvailable = 11;%6;%5;         % actions available = (actionResAvailable - 1)

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

%% states for control statistics
par.IAESS = 0;
par.ITAESS = 0;

%% initial values of externally changed variables
par.MIW = 4.64; % step in water to circuit (m^3/h)
par.MFS = 65.2; % step in ore to circuit (t/h)
%% pulse function V1
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
numError_intervals = 5;%3;%11; % feedback error
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
dActions = linspace(33,60,10)';%[33,39.6,46.2,52.8,59.4]';%linspace(33,66,actionResAvailable)';%linspace(33,60,actionResAvailable)';%linspace(35, 66, actionResAvailable)';

%% initialize first model coordinate
startingCoordinate = [par.XmwSS,par.XmsSS,par.XmfSS,par.XmrSS,par.XmbSS,...
                   par.XswSS,par.XssSS,par.XsfSS,par.IntErrSS,par.IAESS,par.ITAESS]';
               
if outer_Cntr == 1

    %% Added 2021-08-12
    Reps.Weights = out_Reps(outer_Cntr*instances_per_update).Weights;
    Reps.Theta = out_Reps(outer_Cntr*instances_per_update).Theta;
else%if mod(instances_per_update,outer_Cntr) == 0
    Reps.Weights = out_Reps(outer_Cntr*instances_per_update).Weights;
    Reps.Theta = out_Reps(outer_Cntr*instances_per_update).Theta;
end

%%
par.manualDVOne_steps = [4.5,5,6,7,4]; % rock hardness manual steps
par.manualDVOne_times = [10,20,30,40,50]; % times for rock hardness manual steps
par.manualDVTwo_steps = (0.465-0.15) + (0.465-(0.465-0.15))*rand(1,5);  % ore rock fraction manual steps
par.manualDVTwo_times = linspace(0,100,5);% times for rock fraction manual steps

%%
par.manualSP_steps = 0.67:-0.01:0.58; % PSE manual SP steps
par.manualSP_times = 10:10:100;       % times for PSE SP steps

par.manualStepFlag = 1;%0;%1; % flag for selecting manual steps in DVs and SP

%% train controller in process-based parallel environment
for episodeCntr = 1:1:1%training.nmberOfEps

        [learning_temp,temp_Reps] = train_agent(par,training,rvTwo,rvOne,learning,...
                                    dStates,dActions,setPoint,myEnvironment,...
                                    Reps,startingCoordinate,episodeCntr,filter);
        sim_Reps(episodeCntr) = temp_Reps;
        sim_learning(episodeCntr) = learning_temp;
end % end of episode loop

%% report control statistics
STATS.IAE = cell2mat(learning_temp.agentExp{1, 1}(end,7));
STATS.ITAE = cell2mat(learning_temp.agentExp{1, 1}(end,8));
STATS.TV = cell2mat(learning_temp.agentExp{1, 1}(end,9));

% report reward as a statistic as well
STATS.Rewards = sum( cell2mat(learning_temp.agentExp{1, 1}(:,5)) );
sum_stats(outer_Cntr) = STATS;

fprintf('\ncurrent outer counter = %.0f\n', outer_Cntr);


end

subplot(3,1,1)
plot([sum_stats.IAE],'--ko')
title('IAE progression')
subplot(3,1,2)
plot([sum_stats.ITAE],'--ko')
title('ITAE progression')
subplot(3,1,3)
plot([sum_stats.TV],'--ko')
title('TV progression')
%% functions
function [learning,Reps] = train_agent(par,training,rvTwo,rvOne,...
                         learning,dStates,dActions,setPoint,myEnvironment,...
                         Reps,startingCoordinate,episodeCntr,filter)

        if learning.rewardsInWindow < training.targetRunningReward
        clear State
        clear Action
        currentTimeStamp = 1; % initialize MDP time step
        %% START UPDATE 2021-04-20
        %%
        %%
        discountI = 1; % Scalar to incorporate multiplication with discount^t
        %%
        %%
        %% END UPDATE 2021-04-20
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
                    State_2(currentTimeStamp),State_3(currentTimeStamp),dStates,par);
                %% START UPDATE 2021-04-20
                % obtain policy pmf vector at current state 
                policyVec = evaluatePolicy(learning.crntAgentState_1,learning.crntAgentState_2,learning.crntAgentState_3,Reps,...
                            myEnvironment,par.policyCenters,par.policyVariance,par.numA); % obtain pmf @ current state
                learning.policyVecs{1,1}{stepCntr,5} = policyVec;
                % sample action as RV from policy (rejection sampling)
                K = rand(1,1); % sample pseudorandom number from uniform distribution
                learning.samplingInfo{1,episodeCntr}{stepCntr,1} = K; % store pseudorandom sampling information
                likelyActions = zeros(myEnvironment.numberOfActions,1);
                for cntr = 1:1:myEnvironment.numberOfActions
                    if K <= policyVec(cntr)
                        likelyActions(cntr) = 1; % populate vector of likely actions
                    end
                end
                % select actions (includes tie-breaking):
                if sum(likelyActions) == 0  % no action likely when sampling from pmf
                    learning.crntAgentAction = randi(myEnvironment.numberOfActions);
                elseif sum(likelyActions) > 1 % more than one action likely 
                    [actionSubSet,~] = find(ismember(likelyActions,1));
                    learning.crntAgentAction = randi(length(actionSubSet));
                elseif sum(likelyActions) == 1 % one action likely
                    [actionToTake,~] = find(ismember(likelyActions,...
                                     max(likelyActions(:))));
                    learning.crntAgentAction = actionToTake;
                end % end sampling from policy pmf
                %% END UPDATE
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
                
                % perform digital filter calculation
                Action(currentTimeStamp) = digitalFilter(filter.sampletime,...
                                           filter.tauf,previousSmoothed,...
                                           Action(currentTimeStamp));
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
   
                learning.nxtState_1 = setPoint - learning.nxtPSE;   % error component
                % only next state components RESULTING from interaction
                % between agent and RL environment taken into account
                learning.nxtState_2 = State_2(currentTimeStamp);
                learning.nxtState_3 = State_3(currentTimeStamp);
                
                [learning.nxtAgentState_1,learning.nxtAgentState_2,...
                    learning.nxtAgentState_3] = mapStatesToAgent(learning.nxtState_1,...
                                                learning.nxtState_2,...
                                                learning.nxtState_3,dStates,par); 
                %% START UPDATE 2021-04-20
                % obtain policy pmf vector at current state 
                policyVec = evaluatePolicy(learning.nxtAgentState_1,learning.nxtAgentState_2,learning.nxtAgentState_3,Reps,...
                            myEnvironment,par.policyCenters,par.policyVariance,par.numA); % obtain pmf @ current state
                % sample action as RV from policy (rejection sampling)
                K = rand(1,1); % sample pseudorandom number from uniform distribution
                learning.samplingInfo{1,episodeCntr}{stepCntr,1} = K; % store pseudorandom sampling information
                likelyActions = zeros(myEnvironment.numberOfActions,1);
                for cntr = 1:1:myEnvironment.numberOfActions
                    if K <= policyVec(cntr)
                        likelyActions(cntr) = 1; % populate vector of likely actions
                    end
                end
                % select actions (includes tie-breaking):
                if sum(likelyActions) == 0  % no action likely when sampling from pmf
                    learning.nxtAgentAction = randi(myEnvironment.numberOfActions);
                elseif sum(likelyActions) > 1 % more than one action likely 
                    [actionSubSet,~] = find(ismember(likelyActions,1));
                    learning.nxtAgentAction = randi(length(actionSubSet));
                elseif sum(likelyActions) == 1 % one action likely
                    [actionToTake,~] = find(ismember(likelyActions,...
                                     max(likelyActions(:))));
                    learning.nxtAgentAction = actionToTake;
                end % end sampling from policy pmf
    
                %% END UPDATE
                controlError = learning.nxtState_1;
                     learning.Reward = myEnvironment.binReward(controlError);
                     
                % store training information
                learning.agentExp{1,1}{stepCntr,1} = State_1(currentTimeStamp);
                learning.agentExp{1,1}{stepCntr,2} = State_2(currentTimeStamp);
                learning.agentExp{1,1}{stepCntr,3} = State_3(currentTimeStamp);
                learning.agentExp{1,1}{stepCntr,4} = Action(currentTimeStamp);
                learning.agentExp{1,1}{stepCntr,5} = learning.Reward;
                
                learning.agentExp{1,1}{stepCntr,7} = crntModelCoordinate(10);
                learning.agentExp{1,1}{stepCntr,8} = crntModelCoordinate(11);
                learning.agentExp{1,1}{stepCntr,9} = par.TV; 
                %learning.Reward
                %% store ore hardness values (DV not part of states)
                learning.hardness_Data{1,1}{stepCntr,1} = disturbanceValue_1;
                %% START UPDATE 2021-04-20
                % critic basis function vector at current time step
                PHI = populatePhi(learning.crntAgentState_1,learning.crntAgentState_2,learning.crntAgentState_3,par.criticCenters,par.criticVariance); 
                % critic basis function vector at next time step
                PHInxt = populatePhi(learning.nxtAgentState_1,learning.nxtAgentState_2,learning.nxtAgentState_3,par.criticCenters,par.criticVariance); % PHI vector at next time step
                % calculate bootstrap target for gradient-based parameter
                % updates (one-step return)
                sigma = learning.Reward + par.gamma*(Reps.Weights'*PHInxt); 
                % update critic
                if learning.nxtState_1 ~= myEnvironment.Terminal
                    % non-terminal update to critic weights
                    Reps.Weights = Reps.Weights + par.alpha*(sigma - Reps.Weights'*PHI)*PHI;
                elseif learning.nxtState_1 == myEnvironment.Terminal
                    %% changed 2021-04-20
                    % terminal update to critic weights
                    %% Reps.Weights = Reps.Weights + par.alpha*sigma*PHI;
                    %%Reps.Weights = Reps.Weights + par.alpha*(learning.Reward - Reps.Weights'*PHI)*PHI;
                    %% NB NOTE 2021-04-20: Update is only done until (total steps - 1). 
                    % Therefore, the non-terminal update should be used throughout.
                    Reps.Weights = Reps.Weights + par.alpha*(sigma - Reps.Weights'*PHI)*PHI;
                end % end critic update
                % policy basis function vector at current time step 
                X_at_sa = populateX(learning.crntAgentState_1,learning.crntAgentState_2,learning.crntAgentState_3,...
                        Action(currentTimeStamp),par.policyCenters,par.policyVariance,par.numA);            
                % populate eligibility vector
                eligibilityVec = populateVec(learning.crntAgentState_1,learning.crntAgentState_2,learning.crntAgentState_3,...
                                myEnvironment,X_at_sa,Reps,par.policyCenters,par.policyVariance,par.numA);
                % record eligibility vector data
                eligibilityProg{1,episodeCntr}{stepCntr,1} = eligibilityVec;
                % update actor
                if learning.nxtState_1 ~= myEnvironment.Terminal
                    % non-terminal update to policy parameters
                    Reps.Theta = Reps.Theta + par.alphaActor*discountI*...
                                (sigma-Reps.Weights'*PHI)*eligibilityVec;
                elseif learning.nxtState_1 == myEnvironment.Terminal
                    %% changed 2021-04-20
                    % terminal update to policy parameters
                    %%Reps.Theta = Reps.Theta + par.alphaActor*discountI*(sigma)*eligibilityVec;
                    %% NB NOTE 2021-04-20: Update is only done until (total steps - 1). 
                    % Therefore, the non-terminal update should be used throughout.
                    %%Reps.Theta = Reps.Theta + par.alphaActor*discountI*(learning.Reward - Reps.Weights'*PHI)*eligibilityVec;
                    Reps.Theta = Reps.Theta + par.alphaActor*discountI*...
                                (sigma-Reps.Weights'*PHI)*eligibilityVec;
                            
                end
                % update decaying discounting term
                discountI = par.gamma*discountI;
                %%
                %%
                %% END UPDATE
            % shift time step T <- (T+1)
            currentTimeStamp = currentTimeStamp + 1;
            % shift state S(T) <- S(T+1)
            State(currentTimeStamp) = learning.nxtState_1;
            learning.prevPSE = learning.nxtPSE;
            
            end % end loop for non-terminal state
    
        end % end loop for steps
       
        
        end % end rewards in window if-statement
    % Store number of non-terminal entries of each training episode
    optimalityCntr(episodeCntr,1) = currentTimeStamp;
    %fprintf('%d\n',episodeCntr) % display episode number to give indication of training progress

end

function [nxtModelCoordinate,nxtPSE] = simulateMDP(currentTimeStamp,...
                                       crntModelCoordinate,...
                                       a,par,disturbanceValue_1,disturbanceValue_2,setPoint)
    %% solve Hulbert model equations
    % Xmw, Xms, Xmf, Xmr, Xmb, Xsw, Xss, Xsf, IE(t)
    % sump delays larger to accommodate numerical behaviour arising when changing SFW
    par.modLags = [30/3600,30/3600,30/3600,30/3600,30/3600,0.1,0.1,0.1,0.1]; 
    %% old
    % start = currentTimeStamp;
    % stop = currentTimeStamp + 1;
    %% new
    start = currentTimeStamp;%*(1/60);
    stop = ( currentTimeStamp + 1 );%*(1/60);
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
    
    par.IAESS = crntModelCoordinate(10);
    par.ITAESS = crntModelCoordinate(11);
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
    
    nxtIAE = model_high_res(10,end);
    nxtITAE = model_high_res(11,end);
    
    %% store next model coordinate in a vector
    nxtModelCoordinate = [nxtXmw,nxtXms,nxtXmf,nxtXmr,nxtXmb,nxtXsw,nxtXss,nxtXsf,nxtIntErr,nxtIAE,nxtITAE]';
    
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
        par.XssSS,par.XsfSS,par.IntErrSS,par.IAESS,par.ITAESS];
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
    
    % y(10) -> IAE for PSE-MFS loop
    % y(11) -> ITAE for PSE-MFS loop
    
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
    %% calculate integral control statistics
    dIAEdt = abs(par.PSE_SP - PSE);
    dITAEdt = t*abs(par.PSE_SP - PSE);
    
    %% vector of derivatives
    dydt = [dXmwdt,dXmsdt,dXmfdt,dXmrdt,dXmbdt,dXswdt,dXssdt,dXsfdt,dIntERRdt,dIAEdt,dITAEdt]';
    
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
function [agentState_1,agentState_2,agentState_3] = mapStatesToAgent(state_1,state_2,state_3,~,par)

    agentState_1 = -1 + 2*( (state_1 - par.true_error_low_bound)/(par.true_error_high_bound - par.true_error_low_bound) );
    agentState_2 = -1 + 2*((state_2 - par.true_rockfrac_low_bound)/(par.true_rockfrac_high_bound - par.true_rockfrac_low_bound));
    agentState_3 = -1 + 2*( (state_3 - par.true_PSE_SP_low_bound)/(par.true_PSE_SP_high_bound - par.true_PSE_SP_low_bound) );
    
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

%% START UPDATE 2021-04-20

% function to pre-process parameters for policy parameterization (limited to X- and Y-data)
function [centerMatrix,varianceVec] = createPolicyRBFSpecs(X_1_Centers,X_2_Centers,X_3_Centers,YCenters,varianceVec)
% Define parameters for RBF features 
% For GW MDP form is Ci = [center for state, center for action] where i is 
% an integer between 1 and the number of features selected for modelling.
for cntr = 1:1:size(X_1_Centers,2)
    %% changed 2021-04-20
    %for innerCntr = 1:1:numberOfDimensions
        centerMatrix(cntr,1) = X_1_Centers(cntr);  % populate RBF placement for E(t)
        centerMatrix(cntr,2) = X_2_Centers(cntr);  % populate RBF placement for rock fraction
        centerMatrix(cntr,3) = X_3_Centers(cntr);  % populate RBF placement for PSE SP
        centerMatrix(cntr,4) = YCenters(cntr);     % populate RBF placement for MFS (MV)
    %end
    
end
 variance = varianceVec;%varianceVal*ones(1,size(centerMatrix,1)); 
end

% function to pre-process parameters for critic parameterization (limited to X- and Y-data)
function [centerMatrix, variance] = createCriticRBFSpecs(center_1,center_2,center_3,varianceVal)
% Define parameters for RBF features 
for cntr = 1:1:size(center_1,2)
    centerMatrix(cntr,1) = center_1(cntr); % state-value function E(t) critic centers
    centerMatrix(cntr,2) = center_2(cntr); % state-value function rock fraction critic centers
    centerMatrix(cntr,3) = center_3(cntr); % state-value function PSE SP critic centers
end
% RBF variance (does not have to be the same for all features)
variance = varianceVal*ones(1,size(centerMatrix,1)); 
end

% function to populate basis function vector for state-value approximation
% at provided state
function Phi = populatePhi(state_1,state_2,state_3,criticCenters,varianceVector)
% approximation scalar output at that coordinate
FeatureVec = zeros(size(criticCenters,1),1);
    % populate basis function vector
    for t = 1:1:size(criticCenters,1)
         
        %for stateCntr = 1:1:size(stateVec,1)
                % populate numerator state component
                FeatureVec(t,1) = FeatureVec(t,1) + ... 
                                (state_1 - ...
                                criticCenters(t,1)).^2;           
                FeatureVec(t,1) = FeatureVec(t,1) + ... 
                                (state_2 - ...
                                criticCenters(t,2)).^2;
                FeatureVec(t,1) = FeatureVec(t,1) + ... 
                                (state_3 - ...
                                criticCenters(t,3)).^2;
        %end
        % apply square root to obtain the Euclidean norm 
        %FeatureVec(t,1) = sqrt(FeatureVec(t,1));
        % divide by -2*variance
        FeatureVec(t,1) = -1*(FeatureVec(t,1)/(2*varianceVector(t)));
        % take the exponent
        FeatureVec(t,1) = exp(FeatureVec(t,1));
    end  
Phi = FeatureVec;    

end

% populate basis function vector for actor
function X = populateX(state_1,state_2,state_3,actionVec,actorCenters,varianceVector,numOfActions)
    % populate RBF vector as per specifications of vector of states and
    % vector of actions
    FeatureVec = zeros(size(actorCenters,1),1);
    
    %%actionVec = actionVec/numA; % scale action selections to be between 0 and 1
    %%2021-05-30 Scale action selections to be between -1 and 1
    %actionVec = -1 + 1*((actionVec - 1)/(numA - 1)); %% run 1 version
    actionVec = -1 + 2*((actionVec - 1)/(numOfActions - 1));
    % populate feature vector
    for t = 1:1:size(actorCenters,1)
        %% changed 2021-04-20
            %for stateCntr = 1:1:size(stateVec,1)
                % populate numerator state component
                FeatureVec(t,1) = FeatureVec(t,1) + ... 
                                (state_1 - ...
                                actorCenters(t,1)).^2;       
                FeatureVec(t,1) = FeatureVec(t,1) + ... 
                                  (state_2 - ...
                                  actorCenters(t,2)).^2; 
                FeatureVec(t,1) = FeatureVec(t,1) + ... 
                                  (state_3 - ...
                                  actorCenters(t,3)).^2; 
            %end
        %% changed 2021-04-20
            % for actionCntr = 1:1:size(actionVec,2)
                % populate numerator action component
                FeatureVec(t,1) = FeatureVec(t,1) + ...
                                (actionVec - ...
                                actorCenters(t,4)).^2;
            % end
            % apply square root to obtain the Euclidean norm 
            %FeatureVec(t,1) = sqrt(FeatureVec(t,1));
            % divide by -2*variance
            FeatureVec(t,1) = -1*(FeatureVec(t,1)/(2*varianceVector(t)));
            % varianceVector(t)
            % take the exponent
            FeatureVec(t,1) = exp(FeatureVec(t,1));
            
                    
    end     
    X = FeatureVec;
end

% function to populate the eligibility vector of softmax in action selection policy pmf
% at a (S,A)-coordinate.
% (works with vectors of states and actions as well owing to "populateX" function)
function eVec = populateVec(state_1,state_2,state_3,myEnvironment,Xat_sa,Reps,policyCenters,policyVariance,~)
    b = myEnvironment.numberOfActions;
    eVec = zeros(size(Reps.Theta,1),1); % initialize eligibility vector
    % populate eligibility vector
    for outerCntr = 1:1:size(Reps.Theta)
        numeratorSum = 0;
        denominatorSum = 0;
        for sumCntr = 1:1:b
            Xat_sb = populateX(state_1,state_2,state_3,sumCntr,policyCenters,policyVariance,myEnvironment.numberOfActions);
            toAddTop = exp(Reps.Theta'*Xat_sb)*Xat_sb(outerCntr);
            numeratorSum = numeratorSum + toAddTop;
            toAddBot = exp(Reps.Theta'*Xat_sb);
            denominatorSum = denominatorSum + toAddBot;
        end
        eVec(outerCntr,1) = Xat_sa(outerCntr) - (numeratorSum/...
                                                           denominatorSum);
        
    end
    
end

% function that returns pmf at current state for action sampling
% (works with vectors of states and actions as well owing to "populateX" function)
function policy = evaluatePolicy(state_1,state_2,state_3,Reps,myEnvironment,policyCenters,policyVariance,~)
    step = myEnvironment.numberOfActions;
    denominator = 0;
    for cntr = 1:1:step
        Xat_sb = populateX(state_1,state_2,state_3,cntr,policyCenters,policyVariance,myEnvironment.numberOfActions); % generate policy basis function vector
        termToAdd = exp(Reps.Theta'*Xat_sb); 
        denominator = denominator + termToAdd; % denominator of policy (normalization)
    end
        % numerator of policy at current (S,A)-coordinate
        numerator = zeros(step,1);
    for actionStep = 1:1:step
        % generate policy basis function vector
        Xat_s_aStep = populateX(state_1,state_2,state_3,actionStep,policyCenters,policyVariance,myEnvironment.numberOfActions ); 
        numerator(actionStep) = exp(Reps.Theta'*Xat_s_aStep);
        
    end
 
    policy = numerator./denominator; % pmf
    
    
end

% function for RBF placement for policy
function [policyState_1_Centers,policyState_2_Centers,policyState_3_Centers,policy_MV_Centers] = Policy_RBF_placement(numS,numA,error_low_bound,error_high_bound,rockfrac_low_bound,rockfrack_high_bound,PSE_SP_low_bound,PSE_SP_high_bound)
    policyState_1_Centers = repmat(linspace(error_low_bound,error_high_bound,numS),1,(numS*numS*numA));
    elementsForRepetition_2 = linspace(rockfrac_low_bound,rockfrack_high_bound,numS);
    repeatingUnit = zeros(size(elementsForRepetition_2,2),numS);
    for i = 1:1:size(elementsForRepetition_2,2)
        repeatingUnit(i,:) = elementsForRepetition_2(i).*ones(numS,1);
    end
    REPEAT_state_2 = 0;
    for j = 1:1:numS
        REPEAT_state_2 = [REPEAT_state_2,repeatingUnit(j,:)];
    end
    REPEAT_state_2(1) = [];
    elementsForRepetition_3 = linspace(PSE_SP_low_bound,PSE_SP_high_bound,numS);
    policyState_2_Centers = repmat(REPEAT_state_2,1,(numS*numA));%repmat(linspace(0,1,3),1,(3*3*5));
    for k = 1:1:size(elementsForRepetition_3,2)
        repeatingUnit_state_3(k,:) = elementsForRepetition_3(k).*ones(size(REPEAT_state_2,2),1)';
    end
    REPEAT_state_3 = 0;
    for j = 1:1:numS
        REPEAT_state_3 = [REPEAT_state_3,repeatingUnit_state_3(j,:)];
    end
    REPEAT_state_3(1) = [];
    policyState_3_Centers = repmat(REPEAT_state_3,1,numA);

    %% debugged MV grid values
    action_repetition = linspace(-1,1,numA);  %linspace((1/numA),1,numA);  %% scale between -1 and 1 2021-05-30
    row_of_actions = zeros(1,numS*numS*numS*numA);
    policy_MV_Centers = action_repetition(1);
    prevr = 1;
    for r = (numS*numS*numS):(numS*numS*numS):(numS*numS*numS*numA)
        policy_MV_Centers(1,(prevr+1):1:r) = action_repetition(1,r/(numS*numS*numS));
        prevr = r;
    end

end

function [policyState_1_Centers,policyState_2_Centers,policyState_3_Centers,policy_MV_Centers] = Critic_RBF_placement(numS,error_low_bound,error_high_bound,rockfrac_low_bound,rockfrack_high_bound,PSE_SP_low_bound,PSE_SP_high_bound)
    policyState_1_Centers = repmat(linspace(error_low_bound,error_high_bound,numS),1,(numS*numS*1));
    elementsForRepetition_2 = linspace(rockfrac_low_bound,rockfrack_high_bound,numS);
    repeatingUnit = zeros(size(elementsForRepetition_2,2),numS);
    for i = 1:1:size(elementsForRepetition_2,2)
        repeatingUnit(i,:) = elementsForRepetition_2(i).*ones(numS,1);
    end
    REPEAT_state_2 = 0;
    for j = 1:1:numS
        REPEAT_state_2 = [REPEAT_state_2,repeatingUnit(j,:)];
    end
    REPEAT_state_2(1) = [];
    elementsForRepetition_3 = linspace(PSE_SP_low_bound,PSE_SP_high_bound,numS);
    policyState_2_Centers = repmat(REPEAT_state_2,1,(numS*1));
    for k = 1:1:size(elementsForRepetition_3,2)
        repeatingUnit_state_3(k,:) = elementsForRepetition_3(k).*ones(size(REPEAT_state_2,2),1)';
    end
    REPEAT_state_3 = 0;
    for j = 1:1:numS
        REPEAT_state_3 = [REPEAT_state_3,repeatingUnit_state_3(j,:)];
    end
    REPEAT_state_3(1) = [];
    policyState_3_Centers = repmat(REPEAT_state_3,1,1);


    %% debugged MV grid values
    action_repetition = linspace((1/1),1,1);
    row_of_actions = zeros(1,numS*numS*numS*1);
    policy_MV_Centers = action_repetition(1);
    prevr = 1;
    for r = (numS*numS*numS):(numS*numS*numS):(numS*numS*numS*1)
        policy_MV_Centers(1,(prevr+1):1:r) = action_repetition(1,r/(numS*numS*numS));
        prevr = r;
    end

end
