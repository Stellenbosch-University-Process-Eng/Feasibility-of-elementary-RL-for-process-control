%% script that may be used simulate SARSA-based control of the Van de Vusse
%% reaction scheme
%% Different cases of parameter uncertainty as described in the 
%% relevant literature (see folder named "literature") may be selected, but
%% training on the nominal model was of interest in the study.
%% NOTE:  For "predefined == 1", 
%% the action-value function "BINARY002_2000_400_05_099_40second_sampling" 
%% must be present in the same folder as this script
clc
%clear

%%
%% 
% seed random number generator to ensure exploration-exploitation strategy 
% is repeatable
rng(1)
%% START OF USER INPUT

%% indicate whether pre-trained policy is used and time scale
predefined = 1;
SPRV = 0;   % specify whether SP should be treated as an RV or as a selection at the extreme

simSeconds = 1;

%% flags for uncertainty tests
Case_oneFlag = 0;
Case_twoFlag = 0;
%% Define control problem type
SPTracking = 1;
DVTracking = 0;


if SPTracking == 1 && DVTracking == 0 && predefined == 1
    simSeconds = 1;
    numberOfSteps = 200;
    
elseif SPTracking ~= 1 && DVTracking == 1 && predefined == 1
    simSeconds = 1;%0;
    numberOfSteps = 200;%300;
    
elseif SPTracking == 1 && DVTracking == 1 && predefined == 1
    simSeconds = 0;
    numberOfSteps = 200;
end
timeIfSeconds = 4000; 
if predefined ~= 1
    numberOfSteps = 200;
    %simSeconds = 0; NOT IN INITIAL VERSION
    
end

SP_tracking.stepsPerSec = (numberOfSteps/timeIfSeconds);
SP_tracking.SecondsPerStep = 1/SP_tracking.stepsPerSec;
SPstepTimeVec = [400*SP_tracking.stepsPerSec,1200*SP_tracking.stepsPerSec,...
    1500*SP_tracking.stepsPerSec,1700*SP_tracking.stepsPerSec,...
    1900*SP_tracking.stepsPerSec,2100*SP_tracking.stepsPerSec,...
    2300*SP_tracking.stepsPerSec,2500*SP_tracking.stepsPerSec,...
    2700*SP_tracking.stepsPerSec,2900*SP_tracking.stepsPerSec,...
    3100*SP_tracking.stepsPerSec,3300*SP_tracking.stepsPerSec,...
    3500*SP_tracking.stepsPerSec,3700*SP_tracking.stepsPerSec]';
%% 2021-07-22 added a second, alternative SPVec (the second SPVec below).  Also see other 2021-07-22 in code.
SPVec = [1.05,1.04,1.03,1.02,1.01,1.00,0.99,1,1.01,1.02,1.03,1.05,1.07,1.09]'; 
%SPVec = [1.05,1.04,1.03,1.02,1.02,1.02,1.02,1.02,1.03,1.03,1.03,1.05,1.07,1.09]';
%%

DVstepTimeVec = [10,40,70,90,120,150,170,220]';
DVvec = [-4.9,10,2,6,0,10.1,-4.9,10]';

%% parameters for sarsa agent
par.alpha = 0.5;
par.decay_for_epsilon = 0.99; % decay factor to multiply with epsilon
par.gamma = 0.99;%0.7; % discount factor

%% training settings and SP
training.nmberOfEps = 2000;                                % number of episodes to use in training
training.nmberOfSteps = numberOfSteps;                     % number of steps allowed per episode 
setPointVec = [0.95,1.09]';      % vector of SPs to select "randomly" at the start of each episode
training.windowLength = 10;      % window size for summing up rewards
training.targetRunningReward = 5000; % minimum reward in window before stopping training (large because not intended stopping criterion)
training.decayInterval = 1;  % number of episodes before decaying probability of taking a random action

%% details for maintaining MV constant
desiredTrueTime = 20;%40;
conversion_2 = training.nmberOfSteps/timeIfSeconds;
requiredMVSteprest = conversion_2*desiredTrueTime;
trueMVrestTime = (1/conversion_2)*ceil(requiredMVSteprest);


%% simulation settings
sim.nmberOfSteps = 300; % number of time steps (min) to simulate trained agent
sim.startingState = 0;  % initialize control error for simulation

%% filter parameters
filter.sampletime = 1;
filter.tauf = 20; 

%% reward bandwidth setting
rewardBandwidth = 1;

%% flag for binary reward
binReward = 1; % set to 0 to use squared exponential reward function

%% discretize possible states and actions (can lead to error if not sufficient to cover all encountered states)
% state bounds (note that only single state component is applicable in this application) 
MDPstateVecLow = [-0.6,110]';           % vector for lower bounds of all state components
MDPstateVecHigh = [0.6,150]';           % vector for upper bounds of all state components
statesResAvailable = 20;                % number of discrete states = (statesRes - 1)
% action bounds as vectors or scalars
%% NB -> ACTION BOUNDS ARE DEFINED LATER
%% THEREFORE, ONLY LINE 59 IS USED, ACTIONS ARE DEFINED IN LINE 82
MDPactionVecLow = [3,-1113.5]';         % vector for lower bounds of all action components
MDPactionVecHigh = [35,-1113.5]';       % vector for upper bounds of all action components
actionResAvailable = 4;                 % actions available = (actionResAvailable - 1)

%% create training environment (number of actions, states and reward shape)
myEnvironment = createRewardShape; % create RL environment SPECS
disturbanceValue = 5;
disturbanceVec = 5;

%% DEFINE RL ENVIRONMENT
%% model parameters
p.Vr = 0.01*1000;  %reactor volume (m^3)
p.Ar = 0.215; %surface area of cooling jacket (m^2)
p.mc = 5;     %mass of coolant in jacket (kg)
p.Cpc = 2;%2;    %coolant heat capacity (kJ/(kgK))
p.kw = 4032;  %cooling jacket heat transfer coefficient (kJ/(hm^2K))
p.Cp = 3.01;  %reactor content heat capacity (kJ/(kgK))
p.rho = 0.9342; %reactor content density (kg/L)
p.dHrab = 4.2;    %reaction enthalpy change for reaction 1 (kJ/molA)
p.dHrbc = -11;    %reaction enthalpy change for reaction 2 (kJ/mol B)
p.dHrad = -41.85; %reaction enthalpy change for reaction 3 (kJ/mol A)
p.E1 = -9758.3;   %activation energy reaction 1 (converted to K)
p.E2 = -9758.3;   %activation energy reaction 2 (converted to K)
p.E3 = -8560;     %activation energy reaction 3 (converted to K)
p.k10 = (1.287+0)*10^12; %Arrhenius constant for reaction 1 (h^-1)
p.k20 = (1.287+0)*10^12; %Arrhenius constant for reaction 2 (h^-1)
p.k30 = (9.043+0)*10^9;  %Arrhenius constant for reaction 3 (1/((mol A)h))

%% initial ss values and first guesses for model output ss coordinates
p.VdivVr = 14.19;  %flowrate to CSTR normalized to reactor volume (1/h)
p.VdivVrss = 14.19;%store ss value unaltered 
p.CA0 = 5.10; %Inlet concentration of A(mol/L)
p.Qc = -1113.5; %External cooling load to cw (kJ/h)
p.Ts = 114.2;   %Temperature in CSTR @ ss (degrees C)
p.Tcs = 112.9;  %Temperature of cw @ ss (degrees C)
p.T0s = 104.9;  %Intlet temperature @ ss (degrees C)
p.Cas = 2.14;    %Outlet concentration of component A @ ss (mol/L)

p.Cbs = 1.09;    %Outlet concentration of component B @ ss (mol/L)
p.CB0 = 0;       %Inlet concentration of component B.



%%
%Arrhenius equation set  (T [=] degC)
k1 = @(T) p.k10*exp(p.E1/(T+273.15)); 
k2 = @(T) p.k20*exp(p.E2/(T+273.15));
k3 = @(T) p.k30*exp(p.E3/(T+273.15));

%%
%solve SS input values to model -> pre-processing step to align values
%passed to model to the SS of the parameters hardcoded from the ranges
%provided in literature
%where x(1) -> Ca, x(2) -> Cb, x(3) -> T,
%x(4) -> Tcw

ssfun = @(x) [p.VdivVr*(p.CA0 - x(1)) - k1(x(3))*x(1) - k3(x(3))*x(1)^2;...
    
                  -1*p.VdivVr*(x(2)) + k1(x(3))*x(1) - k2(x(3))*x(2);...
                  
                  p.VdivVr*(p.T0s - x(3)) - (1/(p.rho*p.Cp))*(k1(x(3))*x(1)*p.dHrab...
                  + k2(x(3))*x(2)*p.dHrbc + k3(x(3))*x(1)^2*p.dHrad)...
                  + ((p.kw*p.Ar)/(p.rho*p.Cp*p.Vr))*(x(4) - x(3));...
                  
                  (1/(p.mc*p.Cpc))*(p.Qc + p.kw*p.Ar*(x(3) - x(4)));
                  
                  ];
initialguess = [p.Cas, p.Cbs, p.Ts, p.Tcs];
opt = optimoptions('fsolve'); 
opt.MaxIterations = 100;
opt.MaxFunctionEvaluations = 100;
ss = fsolve(ssfun, initialguess, opt); % solve numerically ss coordinate corresponding to user input
%% display true MDP sampling information
% if predefined == 0
%     trueSample = ceil(requiredMVSteprest)*(1/conversion_2);
%     fprintf('true sampling time = %.1f seconds\n', trueSample);
% elseif predefined == 1
%     trueSample = 1*(1/conversion_2);
%     fprintf('true sampling time = %.1f seconds\n', trueSample);
% end
%% Allow for the definition of parameter uncertainty cases
% CASE 1
if Case_oneFlag == 1 && predefined == 1 && Case_twoFlag == 0
    p.k10 = 1.327*10^12;
    p.k20 = 1.327*10^12;
    p.k30 = 8.773*10^9;
    p.dHrab = 6.56;
    p.dHrbc = -9.08;
    p.dHrad = -40.44;
elseif Case_twoFlag == 1 && predefined == 1 && Case_oneFlag == 0
    p.k10 = 1.247*10^12;
    p.k20 = 1.247*10^12;
    p.k30 = 9.313*10^9;
    p.dHrab = 1.84;
    p.dHrbc = -12.92;
    p.dHrad = -43.26;
end


%% solve dynamic model
%Arrhenius equation set  (T [=] degC)
Arr.k1 = @(T) p.k10*exp(p.E1/(T+273.15)); 
Arr.k2 = @(T) p.k20*exp(p.E2/(T+273.15));
Arr.k3 = @(T) p.k30*exp(p.E3/(T+273.15));

%% initialize variable for recording rewards within averaging window
learning.rewardsInWindow = 0;
%% settings for generator functions
SPgen.Tf = training.nmberOfSteps; % simulation period for random SPs
SPgen.x.mu = 1.02;                % mean for SP sampling
SPgen.x.sig = 0.08;               % standard deviation for SP sampling
SPgen.t.mu = 120;                 % mean for sampling times
SPgen.t.sig = 0.5;                % standard deviation for sampling times.

DVgen.Tf = training.nmberOfSteps; % simulation period for random DVs
DVgen.x.mu = 2;         % mean for DV sampling
DVgen.x.sig = 3;        % standard deviation for DV sampling
DVgen.t.mu = 120;       %training.nmberOfSteps/2;                   
DVgen.t.sig = SPgen.t.sig;

%% discretize states and actions and record extreme bounds of CV that may be instantiated
dActions = linspace(3,35,10)';
dActions(1:1:size(dActions,1),2) = -1113.5;

%% discretization 1
fineRes = linspace(-0.049,0.049,10);               
dStates = [-0.6,-0.05,fineRes,0.05,0.6]';
dStates(1:1:size(dStates,1)/2,2) = 0;
dStates((size(dStates,1)/2):1:end,2) = 300;
dStates(1:1:ceil(size(dStates)/3),3) = 50;
dStates(ceil(size(dStates)/3):1:2*ceil(size(dStates)/3),3) = (100+115)/2;
dStates(2*ceil(size(dStates)/3):1:end,3) = 150;

par.epsilonVec = 0.8*ones(size(dStates,1),1); % one epsilon for each ff state (# of intervals of inlet temperature...)

%% create Q-table
if predefined == 1


    load('BINARY002_2000_400_05_099_40second_sampling.mat')
    par.epsilonVec(:,1) = 0.0001; % vector containing initial probabilities of a random action
    
    training.nmberOfEps = 1;
    preSetPoint = ss(2);
    preDist = 0;
    if SPTracking == 1
        stepIndecesForSP = SPstepTimeVec;
        % SP values
        stepSPvalues = SPVec;
    elseif SPTracking ~= 1
        stepIndecesForSP = 0;
        stepSPvalues = 0;
    end
    
    if DVTracking == 1
        stepIndecesForDV = DVstepTimeVec;
        % DVs (as step magnitudes from initial SP)
        stepDVvalues = DVvec;
    elseif DVTracking ~= 1
        stepIndecesForDV = 0;
        stepDVvalues = 0;
    end

    stepID = 0;
    dvStepID = 0;
elseif predefined == 0
    % create action-value array (State components = <E(t),T>;Action components = <(V/Vr),Qk>)
    Reps.action_value = zeros([(size(dStates,1)-1),(size(dStates,1)-1),(size(dStates,1)-1)...
        (size(dActions,1)-1),(size(dActions,1)-1)],'double'); 

end

if simSeconds == 1
    trueTime = timeIfSeconds; % true process time to simulate during each episode (s)
    trueTime = trueTime/3600; % true process time to simulate during each episode (h)
    conversion = trueTime/training.nmberOfSteps; %(MDP sample time)*(conversion) = (timestamp for true process)
    correctedTimeScale = ((1:1:(training.nmberOfSteps-1))*conversion)'; % vector for conversion of results to true process time scale
elseif simSeconds == 0
    conversion = 1;
    correctedTimeScale = ((1:1:(training.nmberOfSteps-1))*conversion)';
end

%% train sarsa-LPF controller
for episodeCntr = 1:1:training.nmberOfEps
    if learning.rewardsInWindow < training.targetRunningReward
        clear State
        clear Action
        currentTimeStamp = 1;
        SPtrainFunc = funcGenerateRandomSteps(SPgen);   % create random SP trajectory
        DVtrainFunc = funcGenerateRandomSteps(DVgen);   % create random DV trajectory
        if predefined == 1
            setPoint = preSetPoint;
            disturbanceValue = preDist;
        elseif predefined == 0
            
            if SPRV == 1
                setPoint = SPtrainFunc.Values(currentTimeStamp);
            elseif SPRV ~= 1
                setPoint = setPointVec(randi(size(setPointVec,1)),1);
            end
            disturbanceValue = DVtrainFunc.Values(currentTimeStamp);
        end
        State_1(currentTimeStamp) = setPoint - ss(2); % initial control error w.r.t. CB
        State_2(currentTimeStamp) = ss(3); % initial reactor temperature
        
        if currentTimeStamp == 1 && predefined == 0 
            State_3(currentTimeStamp) = disturbanceValue + p.T0s; % inlet temperature to reactor
        elseif currentTimeStamp ~= 1 && predefined == 0
            State_3(currentTimeStamp) = DVtrainFunc.Values(currentTimeStamp - 1) + p.T0s;
        elseif predefined == 1
            State_3(currentTimeStamp) = disturbanceValue + p.T0s;
        end
        
        learning.prevCB = setPoint - State_1(currentTimeStamp);
        
        for stepCntr = 1:1:training.nmberOfSteps
            %  perform training in episode if terminal state has not been
            %  achieved
            
            if predefined == 1 && ismember(stepCntr,stepIndecesForSP)
                stepID = stepID + 1;
                setPoint = stepSPvalues(stepID,1);
            
            end
            
            if predefined == 1 && ismember(stepCntr,stepIndecesForDV)
                dvStepID = dvStepID + 1;
                disturbanceValue = stepDVvalues(dvStepID,1);

            end
           
            if  currentTimeStamp < training.nmberOfSteps
                % map current MDP state to corresponding state number for
                % the agent
                [learning.crntAgentState_1,learning.crntAgentState_2,...
                    learning.crntAgentState_3] = mapStatesToAgent(State_1(currentTimeStamp),...
                    State_2(currentTimeStamp),State_3(currentTimeStamp),dStates);
                
                % let agent select the next action 
                [learning.crntAgentAction_1,learning.crntAgentAction_2] = ...
                    selectAction(Reps,par,learning.crntAgentState_1,...
                    learning.crntAgentState_2,learning.crntAgentState_3,...
                    (size(dActions,1)-1));
            if simSeconds == 1 %&& predefined == 0 %% commented on 2021-07-22
                if mod(currentTimeStamp,ceil(requiredMVSteprest)) ~= 0 % if MV action not yet allowed 
                    if currentTimeStamp == 1 % initialize first action selection
                        constAction = learning.crntAgentAction_1;
                    else
                        learning.crntAgentAction_1 = constAction;
                    end
                elseif mod(currentTimeStamp,ceil(requiredMVSteprest)) == 0
                    constAction = learning.crntAgentAction_1;
                    
                end
            end
                % action (true MV control action to take)
                % Action(currentTimeStamp) = mapToMDP(learning.crntAgentAction,dActions,Reps);
                [Action_1(currentTimeStamp),Action_2(currentTimeStamp)] = mapToMDP(learning.crntAgentAction_1,learning.crntAgentAction_2,dActions);
                % filter agent's selected action (initialize with first
                % true agent output)
                if currentTimeStamp > 1
                    previousSmoothed_1 = Action_1(currentTimeStamp - 1);
                    previousSmoothed_2 = Action_2(currentTimeStamp - 1);
                elseif currentTimeStamp == 1
                    previousSmoothed_1 = Action_1(currentTimeStamp);
                    previousSmoothed_2 = Action_2(currentTimeStamp);
                end
    %% store unfiltered actions during training
    learning.discreteSelections{1,episodeCntr}{stepCntr,3} = Action_1(currentTimeStamp);
    learning.discreteSelections{1,episodeCntr}{stepCntr,4} = Action_2(currentTimeStamp);
    %% store SP and DV step info
    learning.SP{1,episodeCntr}{stepCntr,1} = setPoint;
    learning.DV{1,episodeCntr}{stepCntr,1} = disturbanceValue;
    %% for now do not filter action selections 
    % perform digital filter calculation
    Action_1(currentTimeStamp) = digitalFilter(filter.sampletime,filter.tauf,previousSmoothed_1,Action_1(currentTimeStamp));
    Action_2(currentTimeStamp) = digitalFilter(filter.sampletime,filter.tauf,previousSmoothed_2,Action_2(currentTimeStamp));

                if currentTimeStamp == 1
                    prevModelStates = ss;
                    
                else
                    prevModelStates = [nxtCa,nxtCb,nxtT,nxtTc];
                    
                end
                
            
                [nxtCa,nxtCb,nxtT,nxtTc] = ...
                    simulateMDP(currentTimeStamp,prevModelStates,...
                    Action_1(currentTimeStamp),Action_2(currentTimeStamp),...
                    disturbanceValue,p,Arr,conversion);
                learning.nxtState_1 = setPoint - nxtCb; % calculate next control error
                learning.nxtState_2 = nxtT;
                learning.nxtState_3 = disturbanceValue + p.T0s;

                % map next MDP state to corresponding state number for the
                % agent
                [learning.nxtAgentState_1,learning.nxtAgentState_2,...
                    learning.nxtAgentState_3] = mapStatesToAgent(learning.nxtState_1,...
                    learning.nxtState_2,learning.nxtState_3,dStates);
                % let agent select next action
                % learning.nxtAgentAction = selectAction(Reps,par,learning.nxtAgentState,myEnvironment);
                [learning.nxtAgentAction_1,learning.nxtAgentAction_2] = ...
                    selectAction(Reps,par,learning.nxtAgentState_1,...
                    learning.nxtAgentState_2,learning.nxtAgentState_3,(size(dActions,1)-1));
                % obtain reward
                %nxtAction = mapToMDP(learning.nxtAgentAction,dActions,Reps);
                [learning.nxtAction_1,learning.nxtAction_2] = mapToMDP(learning.nxtAgentAction_1,learning.nxtAgentAction_2,dActions);
              % maxMV = MDPactionVecHigh; % not used in reward function anymore (only distance from SP communicated)
                controlError = learning.nxtState_1;
                
                if binReward == 1
                    learning.Reward = myEnvironment.binReward(controlError);
                elseif binReward == 0
                    learning.Reward = myEnvironment.continuousReward(controlError);
                end
                    
                % store training information
                learning.agentExp{1,episodeCntr}{stepCntr,1} = State_1(currentTimeStamp);
                learning.agentExp{1,episodeCntr}{stepCntr,2} = State_2(currentTimeStamp);
                learning.agentExp{1,episodeCntr}{stepCntr,3} = State_3(currentTimeStamp);
                learning.agentExp{1,episodeCntr}{stepCntr,4} = Action_1(currentTimeStamp);
                learning.agentExp{1,episodeCntr}{stepCntr,5} = Action_2(currentTimeStamp);
                learning.agentExp{1,episodeCntr}{stepCntr,6} = learning.Reward;
                % update approximation of action-value function
                if learning.nxtState_1 ~= myEnvironment.Terminal
                    % non-terminal update to action-value array
                    Reps.action_value(learning.crntAgentState_1,...
                               learning.crntAgentState_2,learning.crntAgentState_3,learning.crntAgentAction_1,...
                               learning.crntAgentAction_2) = ...
                               Reps.action_value(learning.crntAgentState_1,...
                               learning.crntAgentState_2,learning.crntAgentState_3,learning.crntAgentAction_1,...
                               learning.crntAgentAction_2) +par.alpha*(learning.Reward + ...
                               par.gamma*Reps.action_value(learning.nxtAgentState_1,...
                               learning.nxtAgentState_2,learning.nxtAgentState_3,learning.nxtAgentAction_1,...
                               learning.nxtAgentAction_2)-...
                               Reps.action_value(learning.crntAgentState_1,...
                               learning.crntAgentState_2,learning.crntAgentState_3,...
                               learning.crntAgentAction_1,learning.crntAgentAction_2));
                        
                elseif learning.nxtState == myEnvironment.Terminal
                    % terminal update to action-value table (Q-table)
                    Reps.action_value(learning.crntAgentState_1,...
                               learning.crntAgentState_2,learning.crntAgentState_3,learning.crntAgentAction_1,...
                               learning.crntAgentAction_2) = ...
                               Reps.action_value(learning.crntAgentState_1,...
                               learning.crntAgentState_2,learning.crntAgentState_3,learning.crntAgentAction_1,...
                               learning.crntAgentAction_2) +par.alpha*(learning.Reward -...
                               Reps.action_value(learning.crntAgentState_1,...
                               learning.crntAgentState_2,learning.crntAgentState_3,...
                               learning.crntAgentAction_1,learning.crntAgentAction_2));
                           
                end % end action-value update
            
            % shift time step T <- (T+1)
            currentTimeStamp = currentTimeStamp + 1;
            % shift state S(T) <- S(T+1)
            State_1(currentTimeStamp) = learning.nxtState_1;
            State_2(currentTimeStamp) = learning.nxtState_2;
            State_3(currentTimeStamp) = learning.nxtState_3;
            % update SP and DV according to sampled trajectories for
            % episode
            if predefined == 0
                if SPRV == 1
                    setPoint = SPtrainFunc.Values(currentTimeStamp);
                end
                disturbanceValue = DVtrainFunc.Values(currentTimeStamp);
            end
            
            end % end loop for non-terminal state
    
        end % end loop for steps
        
        % Calculate reward and steps for window specified in training structure
        if mod(episodeCntr,training.windowLength) == 0
            rewardCount = 0;
            stepCounts = 0;
            residualSteps = 0; % step count not important for this task
        if (episodeCntr/training.windowLength) == 1
            startPoint = 1;
        end
        for cntr = startPoint:1:size(learning.agentExp,2)
            if size(learning.agentExp{1,cntr},1) == (training.nmberOfSteps - 1)
                    residualSteps = residualSteps + 1;
            end
            stepCounts = stepCounts + size(learning.agentExp{1,cntr},1);
            % assume that the episode's goal is achieved if number of steps is less than (maximum steps allowed per episode - 1).
            tempMAT = cell2mat(learning.agentExp{1,cntr});
            rewardCount = sum(tempMAT(:,6));
        end
        stepCounts = stepCounts + residualSteps;
        
        if predefined == 0

            windowData{(episodeCntr/training.windowLength),1} = rewardCount;
            windowData{(episodeCntr/training.windowLength),2} = stepCounts;
        
        end
        
        learning.rewardsInWindow = rewardCount;
        learning.stepsInWindow = stepCounts;
        startPoint = size(learning.agentExp,2)+1; % shift startpoint for calculating statistics of next window
        
        end % end calculation of window statistics
        
    end % end rewards in window if-statement
    % Store number of non-terminal entries of each training episode
    optimalityCntr(episodeCntr,1) = currentTimeStamp;
    fprintf('%d\n',episodeCntr) % display episode number to give indication of training progress
    
    if episodeCntr < training.nmberOfEps && mod(episodeCntr,training.decayInterval) == 0
       %par.epsilon = par.epsilon*par.decay_for_epsilon; % decay probability for taking random action
       par.epsilonVec(learning.crntAgentState_3,1) = par.epsilonVec(learning.crntAgentState_3,1)*par.decay_for_epsilon;
    end
    if episodeCntr == 1
        learning.cumulativeIAE = 0; % initialize cumulative IAE calculation across training episodes
    end
    
end % end of episode loop

%% display training results
if predefined == 0
    
subplot(2,4,1)
learning.averageRewardTrajec = cell2mat(windowData(:,1));
learning.averageStepsTrajec = cell2mat(windowData(:,2));
plot(cell2mat(windowData(:,1)),'ko','LineWidth',2,'MarkerSize',5) % plot moving average rewards
xlabel('averaging window number'); ylabel('Reward in window');
title('rewards in windows')
learning.windowData = windowData; % store moving average window data generated 
% Test for terminal state not being reached:
[optimalEps,~] = find(ismember(optimalityCntr,min(optimalityCntr(:))));
optimalEpisode = optimalEps(1);
if optimalEpisode == 1
    fprintf('\n Warning: number of steps between start state and terminal state the same for all episodes.\n');
end

end
%% first episode outputs
initialMAT = cell2mat(learning.agentExp{1,1}); % first episode's experiences of agent 
initialSelections = cell2mat(learning.discreteSelections{1,1}); % initial unfiltered actions

if predefined == 1
    % display SP data
    setPointData = cell2mat(learning.SP{1,1});
    CBData = setPointData - initialMAT(:,1);
    subplot(2,4,1)
    yyaxis left
    plot(correctedTimeScale,CBData(:),'gx','LineWidth',2)
    hold on
    yyaxis left
    plot(correctedTimeScale,setPointData(:),'k--','LineWidth',2)
    hold on
    xlabel('time (h)'); ylabel('CB and SP (mol/L)')
  
    
    % display DV data
    DVData = cell2mat(learning.DV{1,1});
    DVData = p.T0s + DVData;
    yyaxis right
    plot(correctedTimeScale,DVData(:),'k-','LineWidth',2)
    ylabel('inlet temperature (DV in degrees Celsius)')
    hold on
    plot(correctedTimeScale,p.T0s*ones(size(correctedTimeScale,1)),'m-','LineWidth',2)
    h = legend('CB(mol/L)','SP(mol/L)','DV (degrees Celsius)','Initial SS DV (degrees Celsius)');
    h.FontSize = 7;
    title('SP,DV and CV behaviour')
    
end

subplot(2,4,2)
plot(correctedTimeScale, initialMAT(:,1), 'bo', 'LineWidth', 2); % errors in tank in first episode
xlabel('time (h)'); ylabel('SP - CV (m)');
axis([0 correctedTimeScale(end) -inf inf])
title('State 1 (E(t))'); hold on
subplot(2,4,3)
plot(correctedTimeScale, initialMAT(:,2), 'ko', 'LineWidth', 2); % T values in reactor
xlabel('time (h)');ylabel('T (degrees C)');
axis([0 correctedTimeScale(end) -inf inf])
title('State 2 (T)');
subplot(2,4,4)
plot(correctedTimeScale, initialMAT(:,3), 'ko', 'LineWidth', 2); % inlet temperature (DV)
xlabel('time (h)'); ylabel('T0 (degrees C)');
axis([0 correctedTimeScale(end) -inf inf])
title('State 3 [T0] first episode'); 
subplot(2,4,5)
plot(correctedTimeScale, initialMAT(:,4), 'ko', 'LineWidth', 2); % VdivVr
xlabel('time (h)'); ylabel('VdivVr (1/h)');
axis([0 correctedTimeScale(end) -inf inf])
title('Action (VdivVr)'); 
hold on
plot(correctedTimeScale,initialSelections(:,1),'bx','LineWidth',2)
legend('filtered','unfiltered')
subplot(2,4,6)
plot(correctedTimeScale, initialMAT(:,5), 'ko', 'LineWidth', 2); % filtered Qc actions chosen
xlabel('time (h)'); ylabel('Qc (kJ/h)'); hold on
axis([0 correctedTimeScale(end) -inf inf])
title('filtered Action 2 (Qc)');
plot(correctedTimeScale,initialSelections(:,2),'bx','LineWidth',2)
legend('filtered','unfiltered')
subplot(2,4,7)
plot(correctedTimeScale, initialMAT(:,6), 'ro', 'LineWidth', 2); % rewards
xlabel('time (h)'); ylabel('rewards');
axis([0 correctedTimeScale(end) -inf inf])
title('rewards obtained');

%% last episode outputs
if predefined == 0
figure
subplot(2,3,1)
finalMAT = cell2mat(learning.agentExp{1,episodeCntr}); % final episode's experiences of agent
finalSelections = cell2mat(learning.discreteSelections{1,episodeCntr}); % final episode's discrete action selections
plot(correctedTimeScale,finalMAT(:,1), 'bo', 'LineWidth', 2); % errors in tank in last episode
xlabel('time (h)'); ylabel('SP - CV (m)');
axis([0 correctedTimeScale(end) -inf inf])
title('State 1 (E(t)) last episode'); hold on
subplot(2,3,2)
plot(correctedTimeScale, finalMAT(:,2), 'ko', 'LineWidth', 2); % T values in reactor
xlabel('time (h)');ylabel('T (degrees C)');
axis([0 correctedTimeScale(end) -inf inf])
title('State 2 (T) last episode');
subplot(2,3,3)
plot(correctedTimeScale, finalMAT(:,3), 'ko', 'LineWidth', 2); % inlet temperatures
xlabel('time (h)');ylabel('T0 (degrees C)');
axis([0 correctedTimeScale(end) -inf inf])
title('State 3 [T0] (last episode)');
subplot(2,3,4)
plot(correctedTimeScale, finalMAT(:,4), 'ko', 'LineWidth', 2); % filtered VdivVr actions chosen
xlabel('time (h)'); ylabel('VdivVr (1/h)'); hold on
axis([0 correctedTimeScale(end) -inf inf])
plot(correctedTimeScale,finalSelections(:,1),'bx','LineWidth',2)
title('Action 1 (VdivVr) last episode');
legend('filtered','unfiltered')
subplot(2,3,5)
plot(correctedTimeScale, finalMAT(:,5), 'ko', 'LineWidth', 2); % filtered Qc actions chosen
xlabel('time (h)'); ylabel('Qc (kJ/h)'); hold on
axis([0 correctedTimeScale(end) -inf inf])
title('Action 2 (Qc) last episode');
plot(correctedTimeScale,finalSelections(:,2),'bx','LineWidth',2)
legend('filtered','unfiltered')
subplot(2,3,6)
plot(correctedTimeScale, finalMAT(:,6), 'ro', 'LineWidth', 2); % rewards
xlabel('time (h)'); ylabel('rewards');
axis([0 correctedTimeScale(end) -inf inf])
title('rewards obtained during last episode');
end

%% save action-value table
% save('action_value_table_VdV_no_ff_part_10000_ep','Reps')

%% functions
% function to capture the dynamics of the Van der Vusse benchmark
function dVdVdt = VdVmodel(t,x, p, Arr)
%VdV model in column matrix form, where x(1) -> Ca, x(2) -> Cb, x(3) -> T,
%x(4) -> Tcw

dVdVdt =         [p.VdivVr*(p.CA0 - x(1)) - Arr.k1(x(3))*x(1) - Arr.k3(x(3))*x(1)^2;...
    
                  -1*p.VdivVr*(x(2)) + Arr.k1(x(3))*x(1) - Arr.k2(x(3))*x(2);...
                  
                  p.VdivVr*(p.T0 - x(3)) - (1/(p.rho*p.Cp))*(Arr.k1(x(3))*x(1)*p.dHrab...
                  + Arr.k2(x(3))*x(2)*p.dHrbc + Arr.k3(x(3))*x(1)^2*p.dHrad) ...
                  + ((p.kw*p.Ar)/(p.rho*p.Cp*p.Vr))*(x(4) - x(3));...
                  
                  (1/(p.mc*p.Cpc))*(p.Qc + p.kw*p.Ar*(x(3) - x(4)));
                  
                  ];
end

% function to map true states of MDP to coded states of agent
function [agentState_1,agentState_2,agentState_3] = mapStatesToAgent(state_1,state_2,state_3,dStates)
    % identify coded state component 1 (control error)
    for cntr = 1:1:(size(dStates,1)-1)
        if state_1 > dStates(cntr,1) && state_1 <= dStates(cntr+1,1)%< dStates(cntr+1,1)
            agentState_1 = cntr;
        else
        end
    end
    % identify coded state component 2 (temperature in reactor -> only one state available)
    for cntr = 1:1:(size(dStates,1)-1)
        if state_2 > dStates(cntr,2) && state_2 <= dStates(cntr+1,2)%< dStates(cntr+1,1)
            agentState_2 = cntr;
        else
        end
    end
    % identify coded state component 3 (inlet temperature)
    for cntr = 1:1:(size(dStates,1)-1)
        if state_3 > dStates(cntr,3) && state_3 <= dStates(cntr+1,3)%< dStates(cntr+1,1)
            agentState_3 = cntr;
        else
        end
    end
    
end

% function to select action
function [Action_1,Action_2] = selectAction(Reps,par,state_1,state_2,state_3,numberOfActions)
    t = rand(1);
    if t <= par.epsilonVec(state_3,1)
        % take random action
        Action_1 = randi(numberOfActions);
        Action_2 = randi(numberOfActions);
    elseif t > par.epsilonVec(state_3,1)
        % take greedy action
        vec = Reps.action_value(state_1,state_2,state_3,:,:);
        index = find(ismember(vec(:),max(vec(:))));
        [~,~,~,Action_1,Action_2] = ind2sub(size(vec(:,:,:,:,:)), index);
        
    end
    
    % tie breaking
    if size(Action_1,1) > 1
        Action_1 = randi(numberOfActions);
    end
    
    if size(Action_2,1) > 1
        Action_2 = randi(numberOfActions);
    end
    
end

% function to simulate the MDP
function [nxtCa,nxtCb,nxtT,nxtTc] = simulateMDP(currentTimeStamp,prevModelStates,action_1,action_2,disturbanceValue,p,Arr,conversion)
    MDPstart = currentTimeStamp;
    MDPstop = currentTimeStamp + 1;
    start = MDPstart*conversion;
    stop = MDPstop*conversion;
    tspan = linspace(start, stop, 10);
    inletTDisturbance = disturbanceValue; % disturbance in inlet temperature (degC)
    p.T0 = p.T0s + disturbanceValue;
    p.VdivVr = action_1;
    p.Qc = action_2;
    [~, Output] = ode23s(@ (t, x) VdVmodel(t, x, p, Arr), tspan, prevModelStates);
    nxtCa = Output(end,1);
    nxtCb = Output(end,2);
    nxtT = Output(end,3);
    nxtTc = Output(end,4);
   
end

% function to create the parameters for the RL environment's model
function rlEnv = createRewardShape(~)
    rlEnv.continuousReward = @(controlError) exp((-1*controlError^2)/0.01);
    rlEnv.binReward = @(controlError) -1 + 1*(controlError < 0.02) + 1*(controlError > -0.02);
    %rlEnv.binReward = @(controlError) -1 + 1*(controlError < 0.2) + 1*(controlError > -0.2);
    rlEnv.Terminal = -1000;
    
end

% function to discretize state and action spaces
function [dStates, dActions] = discretize(MDPstateVecLow,MDPstateVecHigh,...
                             MDPactionVecLow,MDPactionVecHigh,...
                             stateRes,actionRes)
    if size(MDPstateVecLow,1) == 1
        MDPstateBounds = [MDPstateVecLow,MDPstateVecHigh]';
    elseif size(MDPstateVecLow,1) > 1
        for cntr = 1:1:size(MDPstateVecLow,1)
            MDPstateBounds(1,cntr) = MDPstateVecLow(cntr);
            MDPstateBounds(2,cntr) = MDPstateVecHigh(cntr);
        end
    end

    if size(MDPactionVecLow,1) == 1
        MDPactionBounds = [MDPactionVecLow,MDPactionVecHigh]';
    elseif size(MDPactionVecLow,1) > 1
        for cntr = 1:1:size(MDPactionVecLow,1)
            MDPactionBounds(1,cntr) = MDPactionVecLow(cntr);
            MDPactionBounds(2,cntr) = MDPactionVecHigh(cntr);
        end
    end

    dStates = zeros(stateRes,size(MDPstateBounds,2)); % initialize variable for creating a discretized representation of possible states
    dActions = zeros(actionRes,size(MDPactionBounds,2)); % initialize variable for creating a discretized representation of possible actions
    % create discretized states for mapping
    for cntr = 1:1:size(MDPstateBounds,2)
        dStates(:,cntr) = linspace(MDPstateBounds(1,cntr), MDPstateBounds(2,cntr),stateRes);
    end
    % create discretized actions for mapping
    for cntr = 1:1:size(MDPactionBounds,2)
        dActions(:,cntr) = linspace(MDPactionBounds(1,cntr), MDPactionBounds(2,cntr),actionRes);
    end
    
end

% function to map agent's action to the MV output of the MDP (arithmetic
% mean of discretized action bounds of agent's selection)
function [MDPAction_1,MDPAction_2] = mapToMDP(agentActionIndex_1,agentActionIndex_2,dActions)
   
   MDPAction_1 = dActions(agentActionIndex_1,1);
   MDPAction_2 = dActions(agentActionIndex_2,2);
   
end

%  digital low-pass filter (chapter 12 of MARLIN)
function smoothedAction = digitalFilter(sampleTime,tauf,previousSmoothed,currentAction)
    A = exp(-1*sampleTime/tauf);
    % time domain digital filter calculation
    smoothedAction = A*previousSmoothed + (1-A)*currentAction;
    
end

%% Function to create the random steps

function fx = funcGenerateRandomSteps(p)


% Identify the time points at which steps will occur

i = 1;

tstep(i) = 0;

while tstep(i) < p.Tf     % while smaller than the simulation period

    i = i+1;              % increment the counter

    tstep(i) = tstep(i-1) + abs(p.t.mu + p.t.sig*randn); % Ensure no backwards steps are taken

end


% Generate a vector of equally spaced time points with high enough

% resolution to accurately capture the steps

t = linspace(0, p.Tf, p.Tf);%100*ceil(p.Tf/p.t.mu));


% Calculate the values of "x" during each step

xstep = p.x.mu + p.x.sig*randn(i,1);

x = 0*t;

for i = 1:length(tstep)

    x(t >= tstep(i)) = xstep(i);

end


% Create a function that acts as interpolant

fx = griddedInterpolant(t, x);


end
