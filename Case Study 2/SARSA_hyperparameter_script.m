%% Script used to run different instances of the SARSA hyperparameters for 
%% the Van de Vusse reaction scheme problem.
%% NOTES: Parallel computation used.  Parameters must be set manually for a run. 
% clc
% clear
tic
%%
%% 
% seed random number generator to ensure exploration-exploitation strategy 
% is repeatable
rng(1)

myPool = parpool(16);
%% START OF USER INPUT

%% indicate time scale
simSeconds = 1; % set to one for training with 40 second delays between action selections

%% specify true time to simulate
timeIfSeconds = 4000;

%% parameters for sarsa agent
par.alpha = 2.1;%2.5;%0.5;
par.decay_for_epsilon = 1;%0.99; % decay factor to multiply with epsilon
par.gamma = 0.99;%0.99;         % discount factor

%% training settings and SP
training.nmberOfEps = 11500;%23000;                 % number of episodes to use in training
training.nmberOfSteps = 400;     % number of steps allowed per episode 
training.windowLength = 10;      % window size for summing up rewards
training.targetRunningReward = 5000; % minimum reward in window before stopping training (large because not intended stopping criterion)
training.decayInterval = 1;  % number of episodes before decaying probability of taking a random action

%% details for maintaining MV constant
desiredTrueTime = 40;
conversion_2 = training.nmberOfSteps/timeIfSeconds;
requiredMVSteprest = conversion_2*desiredTrueTime;
trueMVrestTime = (1/conversion_2)*ceil(requiredMVSteprest);

%% filter parameters
filter.sampletime = 1;
filter.tauf = 20; 

%% flag for binary reward
binReward = 1; % set to 0 to use squared exponential reward function

%% create training environment (number of actions, states and reward shape)
myEnvironment = createRewardShape; % create RL environment SPECS

%% settings for learning curve construction
windowSize = 10;%100; % size of windows for summing rewards
setStep_Num = 20; % number of steps per episode for curve construction

%% DEFINE RL ENVIRONMENT
%% model parameters
p.Vr = 0.01*1000;  %reactor volume (m^3)
p.Ar = 0.215; %surface area of cooling jacket (m^2)
p.mc = 5;     %mass of coolant in jacket (kg)
p.Cpc = 2;    %coolant heat capacity (kJ/(kgK))
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

%% Beginning of sampling details
%% SP for component B and DV (reactor inlet temperature)
%% number of SP and DV changes per episode
rvOne.numSP = 10;
rvOne.numDV = 10;

%% lower and upper bounds for SP sampling (concentration of component B in the product stream)
rvOne.lower_SP = 0.95;
rvOne.upper_SP = 1.11;

%% lower and upper bounds for DV sampling (inlet temperature)
rvOne.lower_DV = 100 - p.T0s;
rvOne.upper_DV = 115 - p.T0s;

%% integer value for random permutations determining when SP and DV steps will occur
rvOne.simTime = training.nmberOfSteps;

%% end of sampling details

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
    trueSample = ceil(requiredMVSteprest)*(1/conversion_2);
    fprintf('true sampling time = %.1f seconds\n', trueSample);

%% solve dynamic model
%Arrhenius equation set  (T [=] degC)
Arr.k1 = @(T) p.k10*exp(p.E1/(T+273.15)); 
Arr.k2 = @(T) p.k20*exp(p.E2/(T+273.15));
Arr.k3 = @(T) p.k30*exp(p.E3/(T+273.15));

%% initialize variable for recording rewards within averaging window
learning.rewardsInWindow = 0;


%% discretize states and actions and record extreme bounds of CV that may be instantiated
%% max number of states in one dimension = 10; minimum 1
%% fill in (number of states + 1) and (number of actions + 1) below
numError_intervals = 4;
numInletT_intervals = 11;
numSP_intervals = 6;
numAction_intervals = 11; 

%% define lower bounds for discretizations
errorLow = -0.05;
inletTLow = 100;
CBSPLow = 0.95;

%% define "padding" for discretization
numSP_padding = CBSPLow.*ones((13 - (2 + numSP_intervals)),1)';
error_padding = errorLow.*ones((13 - (2 + numError_intervals)),1)';
inletT_padding = inletTLow.*ones((13 - (2 + numInletT_intervals)),1)';

dActions = linspace(3,35,numAction_intervals)';%10)';
dActions(1:1:size(dActions,1),2) = -1113.5;

%% discretize states
% control error
fineRes = linspace(-0.05,0.05,numError_intervals);               
dStates = [-1,error_padding,fineRes,1]';
% reactor temperature
% dStates(1:1:size(dStates,1)/2,2) = 0;
% dStates((size(dStates,1)/2):1:end,2) = 300;
dStates(1,2) = 0;
dStates(2:1:end,2) = 300;
% inlet temperature
% dStates(1:1:ceil(size(dStates)/3),3) = 50;
% dStates(ceil(size(dStates)/3):1:2*ceil(size(dStates)/3),3) = (100+115)/2;
% dStates(2*ceil(size(dStates)/3):1:end,3) = 150;
fineInletTRes = linspace(100,115,numInletT_intervals);
dStates(1:1:end,3) = [90,inletT_padding,fineInletTRes,120]';
% CB SP
fineCBSPRes = linspace(0.95,1.11,numSP_intervals);
dStates(1:1:end,4) = [0.90,numSP_padding,fineCBSPRes,1.20]';
% dStates(1:1:2,4) = 0.6;
% dStates(2:1:end,4) = 1.2;

par.epsilonVec = 0.1*ones(size(dStates,1),1);%0.8*ones(size(dStates,1),1); % one epsilon for each ff state (# of intervals of inlet temperature...)

%% create Q-table
% create action-value array (State components = <E(t),T,inletT,CB_SP>;Action components = <(V/Vr),Qk>)
%% old
% Reps.action_value = zeros([(size(dStates,1)-1),(size(dStates,1)-1),(size(dStates,1)-1), (size(dStates,1)-1)...
%                   (size(dActions,1)-1),(size(dActions,1)-1)],'double'); 
%% new
Reps.action_value = zeros([(size(dStates,1)-1),2,(size(dStates,1)-1), (size(dStates,1)-1)...
                  (size(dActions,1)-1),2],'double'); 
%% time conversions between MDP time steps and dynamic model
if simSeconds == 1
    trueTime = timeIfSeconds; % true process time to simulate during each episode (s)
    trueTime = trueTime/3600; % true process time to simulate during each episode (h)
    conversion = trueTime/training.nmberOfSteps; %(MDP sample time)*(conversion) = (timestamp for true process)
    correctedTimeScale = ((1:1:(training.nmberOfSteps-1))*conversion)'; % vector for conversion of results to true process time scale
elseif simSeconds == 0
    conversion = 1;
    correctedTimeScale = ((1:1:(training.nmberOfSteps-1))*conversion)';
end

setPoint = ss(2); % initialize set point

%% initialize out_Reps
%out_Reps = zeros(size(Reps.action_value));
%% train sarsa-LPF controller
parfor episodeCntr = 1:training.nmberOfEps
    initialize_disturbanceValue = 0;
    [learning_temp,temp_Reps] = train_agent(par,training,rvOne,learning,...
                                dStates,dActions,setPoint,myEnvironment,...
                                Reps,ss,episodeCntr,filter,initialize_disturbanceValue,p,simSeconds,requiredMVSteprest,Arr,conversion,binReward);
    %out_Reps{1,episodeCntr}
    out_Reps(episodeCntr) = temp_Reps;%cast(temp_Reps.action_value,'single');
    out_learning{1,episodeCntr} = learning_temp.agentExp{1,episodeCntr};

end % end of episode loop

% %% approximate overall action-value table with simplified boundary conditions
% for i = 1:1:training.nmberOfEps
%     Reps.action_value = Reps.action_value + out_Reps(i).action_value;%out_Reps{1,i}.action_value;
% end
%             
% %% run trained sarsa-LPF controller
% for sim_episodeCntr = 1:1
%     sim_disturbanceValue = 0;
%     [learning_sim,sim_Reps] = train_agent(par,training,rvOne,learning,...
%                                 dStates,dActions,setPoint,myEnvironment,...
%                                 Reps,ss,sim_episodeCntr,filter,sim_disturbanceValue,p,simSeconds,requiredMVSteprest,Arr,conversion,binReward);
%     sim_Reps = sim_Reps;
%     sim_learning = learning_sim.agentExp{1,sim_episodeCntr};
% end % end of episode loop
% 
% 
% % %% construct learning curve
% fprintf('NOTE:  Testing settings for DV and SP used when generating learning curve\n')
% N = training.nmberOfEps;  % # Eps for curve generation
% training.nmberOfSteps = setStep_Num;
% note_string = sprintf('NOTE: training.nmberOfSteps set to %.0f for learning curve generation', training.nmberOfSteps);
% fprintf(note_string)
% fprintf('\n')
% W = windowSize; % window size
% %% NUMBER OF STEPS 
% k = (N/W); % number of entries in learning curve
% Curve.action_value = zeros.*Reps.action_value;  % initialize Q-table
% learning_curve = zeros(k,1); % initialize learning curve variable
% index_cntr = 0;%zeros(k,1); % initialize counter to index learning curve entries
% for cntr = 1:1:N % loop episodes
%     initialize_windowDisturbance = 0;
%     % add layer of action-values
%     Curve.action_value = Curve.action_value + out_Reps(cntr).action_value;%out_Reps{1,i}.action_value;%out_Reps(cntr).action_value;
%     [Curve_learning(cntr),Curve] = train_agent(par,training,rvOne,learning,...
%                                 dStates,dActions,setPoint,myEnvironment,...
%                                 Reps,ss,sim_episodeCntr,filter,initialize_windowDisturbance,p,simSeconds,requiredMVSteprest,Arr,conversion,binReward);
%     if mod(cntr,W) == 0
%        index_cntr = index_cntr + 1;
%        
%          for window_Cntr = ( cntr - W + 1 ):1:cntr
%              learning_curve(index_cntr,1) = learning_curve(index_cntr,1) + sum( cell2mat(Curve_learning(window_Cntr).agentExp{1, 1}(:,7)) );
%          end
%        
%     end
%     
% end

delete(myPool)
toc
%% functions

% training function
function [learning,Reps] = train_agent(par,training,rvOne,learning,dStates,dActions,setPoint,myEnvironment,Reps,ss,episodeCntr,filter,disturbanceValue,p,simSeconds,requiredMVSteprest,Arr,conversion,binReward)
    if learning.rewardsInWindow < training.targetRunningReward
        clear State
        clear Action
        currentTimeStamp = 1;
        
        State(currentTimeStamp) = setPoint - ss(2); % initial control error w.r.t. CB
        
        learning.prevCB = setPoint - State(currentTimeStamp);
        
        %% sample SP and DV (reactor inlet temperature)
        [SP_times,SP_steps,DVOne_times,DVOne_steps] = generateEpRVSamples(rvOne,training.nmberOfSteps);
        
        for stepCntr = 1:1:training.nmberOfSteps

            %% adjust setPoint and DV (inlet temperature) as per sampling
            for SP_cntr = 1:1:rvOne.numSP
                if SP_times(SP_cntr) == currentTimeStamp%mod(SP_times(SP_cntr),currentTimeStamp) == 0
                    setPoint = SP_steps(SP_cntr);
                    
                end
                
            end
            
            
            %% adjust disturbanceValue (inlet temperature)
            for DV_cntr = 1:1:rvOne.numDV
                if DVOne_times(DV_cntr) == currentTimeStamp%mod(DVOne_times(DV_cntr),currentTimeStamp) == 0
                    disturbanceValue = DVOne_steps(DV_cntr); 
                    
                end
            end
            %% end adjustment of SP and DV as per sampling
            %% state component definitions
            State_1(currentTimeStamp) = State(currentTimeStamp);%setPoint - ss(2); % control error
            State_2(currentTimeStamp) = ss(3); % initial reactor temperature (kept at one level throughout
            State_3(currentTimeStamp) = disturbanceValue + p.T0s; % inlet temperature
            State_4(currentTimeStamp) = setPoint; % CB set point

            if  currentTimeStamp < training.nmberOfSteps
                % map current MDP state to corresponding state number for
                % the agent
                [learning.crntAgentState_1,learning.crntAgentState_2,...
                    learning.crntAgentState_3,learning.crntAgentState_4] = mapStatesToAgent(State_1(currentTimeStamp),...
                    State_2(currentTimeStamp),State_3(currentTimeStamp),State_4(currentTimeStamp),dStates);
                
                % let agent select the next action 
                [learning.crntAgentAction_1,learning.crntAgentAction_2] = ...
                    selectAction(Reps,par,learning.crntAgentState_1,...
                    learning.crntAgentState_2,learning.crntAgentState_3,learning.crntAgentState_4,...
                    (size(dActions,1)-1));
                if simSeconds == 1 % && predefined == 0
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
                learning.DV{1,episodeCntr}{stepCntr,1} = disturbanceValue + p.T0s;
                %% for now do not filter action selections 
                % perform digital filter calculation
                Action_1(currentTimeStamp) = digitalFilter(filter.sampletime,filter.tauf,previousSmoothed_1,Action_1(currentTimeStamp));
                Action_2(currentTimeStamp) = digitalFilter(filter.sampletime,filter.tauf,previousSmoothed_2,Action_2(currentTimeStamp));

                if currentTimeStamp == 1
                    prevModelStates = [ ss 0 ];
                    
                else
                    prevModelStates = [nxtCa,nxtCb,nxtT,nxtTc,nxtIAE];
                    
                end
                
                [nxtCa,nxtCb,nxtT,nxtTc,nxtIAE] = ...
                    simulateMDP(currentTimeStamp,prevModelStates,...
                    Action_1(currentTimeStamp),Action_2(currentTimeStamp),...
                    disturbanceValue,p,Arr,conversion,setPoint);
                learning.nxtState_1 = setPoint - nxtCb; % calculate next control error
                learning.nxtState_2 = nxtT;
                % Note that next timestep is only incorporated for outputs
                % resulting from interaction with the RL environment
                learning.nxtState_3 = State_3(currentTimeStamp);%disturbanceValue + p.T0s;
                learning.nxtState_4 = State_4(currentTimeStamp);%setPoint;
                % map next MDP state to corresponding state number for the
                % agent
                [learning.nxtAgentState_1,learning.nxtAgentState_2,...
                    learning.nxtAgentState_3,learning.nxtAgentState_4] = mapStatesToAgent(learning.nxtState_1,...
                    learning.nxtState_2,learning.nxtState_3,learning.nxtState_4,dStates);
                % let agent select next action
                % learning.nxtAgentAction = selectAction(Reps,par,learning.nxtAgentState,myEnvironment);
                [learning.nxtAgentAction_1,learning.nxtAgentAction_2] = ...
                    selectAction(Reps,par,learning.nxtAgentState_1,...
                    learning.nxtAgentState_2,learning.nxtAgentState_3,learning.nxtAgentState_4,(size(dActions,1)-1));
                % obtain reward
                %nxtAction = mapToMDP(learning.nxtAgentAction,dActions,Reps);
                [learning.nxtAction_1,learning.nxtAction_2] = mapToMDP(learning.nxtAgentAction_1,learning.nxtAgentAction_2,dActions);
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
                learning.agentExp{1,episodeCntr}{stepCntr,4} = State_4(currentTimeStamp);
                learning.agentExp{1,episodeCntr}{stepCntr,5} = Action_1(currentTimeStamp);
                learning.agentExp{1,episodeCntr}{stepCntr,6} = Action_2(currentTimeStamp);
                learning.agentExp{1,episodeCntr}{stepCntr,7} = learning.Reward;
                
                learning.agentExp{1,episodeCntr}{stepCntr,9} = nxtIAE;
                % update action-value function approximation
                if learning.nxtState_1 ~= myEnvironment.Terminal
                    % non-terminal update to action-value array
                    Reps.action_value(learning.crntAgentState_1,...
                               learning.crntAgentState_2,learning.crntAgentState_3,learning.crntAgentState_4,learning.crntAgentAction_1,...
                               learning.crntAgentAction_2) = ...
                               Reps.action_value(learning.crntAgentState_1,...
                               learning.crntAgentState_2,learning.crntAgentState_3,learning.crntAgentState_4,learning.crntAgentAction_1,...
                               learning.crntAgentAction_2) +par.alpha*(learning.Reward + ...
                               par.gamma*Reps.action_value(learning.nxtAgentState_1,...
                               learning.nxtAgentState_2,learning.nxtAgentState_3,learning.nxtAgentState_4,learning.nxtAgentAction_1,...
                               learning.nxtAgentAction_2)-...
                               Reps.action_value(learning.crntAgentState_1,...
                               learning.crntAgentState_2,learning.crntAgentState_3,learning.crntAgentState_4,...
                               learning.crntAgentAction_1,learning.crntAgentAction_2));
                        
                elseif learning.nxtState == myEnvironment.Terminal
                    % terminal update to action-value table (Q-table)
                    Reps.action_value(learning.crntAgentState_1,...
                               learning.crntAgentState_2,learning.crntAgentState_3,learning.crntAgentState_4,learning.crntAgentAction_1,...
                               learning.crntAgentAction_2) = ...
                               Reps.action_value(learning.crntAgentState_1,...
                               learning.crntAgentState_2,learning.crntAgentState_3,learning.crntAgentState_4,learning.crntAgentAction_1,...
                               learning.crntAgentAction_2) +par.alpha*(learning.Reward -...
                               Reps.action_value(learning.crntAgentState_1,...
                               learning.crntAgentState_2,learning.crntAgentState_3,learning.crntAgentState_4,...
                               learning.crntAgentAction_1,learning.crntAgentAction_2));
                           
                end % end action-value update
            
            % shift time step T <- (T+1)
            currentTimeStamp = currentTimeStamp + 1;
            % shift state S(T) <- S(T+1)
            %% the other states are updated when sampling DV and SP data
            State(currentTimeStamp) = learning.nxtState_1;
            
            end % end loop for non-terminal state
    
        end % end loop for steps

        
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

end
% function to capture the dynamics of the Van der Vusse reaction scheme
% model
function dVdVdt = VdVmodel(t,x, p, Arr,setPoint)
%VdV model in column matrix form, where x(1) -> Ca, x(2) -> Cb, x(3) -> T,
%x(4) -> Tcw
%x(5) -> abs(E(t))

dVdVdt =         [p.VdivVr*(p.CA0 - x(1)) - Arr.k1(x(3))*x(1) - Arr.k3(x(3))*x(1)^2;...
    
                  -1*p.VdivVr*(x(2)) + Arr.k1(x(3))*x(1) - Arr.k2(x(3))*x(2);...
                  
                  p.VdivVr*(p.T0 - x(3)) - (1/(p.rho*p.Cp))*(Arr.k1(x(3))*x(1)*p.dHrab...
                  + Arr.k2(x(3))*x(2)*p.dHrbc + Arr.k3(x(3))*x(1)^2*p.dHrad) ...
                  + ((p.kw*p.Ar)/(p.rho*p.Cp*p.Vr))*(x(4) - x(3));...
                  
                  (1/(p.mc*p.Cpc))*(p.Qc + p.kw*p.Ar*(x(3) - x(4)));...
                  
                  abs(setPoint - x(2))];
end

% function to map true states of MDP to coded states of agent
function [agentState_1,agentState_2,agentState_3,agentState_4] = mapStatesToAgent(state_1,state_2,state_3,state_4,dStates)
    % identify coded state component 1 (control error)
    for cntr = 1:1:(size(dStates,1)-1)
        if state_1 > dStates(cntr,1) && state_1 <= dStates(cntr+1,1)%< dStates(cntr+1,1)
            agentState_1 = cntr;
        else
        end
    end
    %% old
%     % identify coded state component 2 (temperature in reactor -> only one state available)
%     for cntr = 1:1:(size(dStates,1)-1)
%         if state_2 > dStates(cntr,2) && state_2 <= dStates(cntr+1,2)%< dStates(cntr+1,1)
%             agentState_2 = cntr;
%         else
%         end
%     end
%% new
agentState_2 = 1;
    % identify coded state component 3 (inlet temperature)
    for cntr = 1:1:(size(dStates,1)-1)
        if state_3 > dStates(cntr,3) && state_3 <= dStates(cntr+1,3)%< dStates(cntr+1,1)
            agentState_3 = cntr;
        else
        end
    end
    % identify coded state component 4 (CB setpoint)
    for cntr = 1:1:(size(dStates,1)-1)
        if state_4 > dStates(cntr,4) && state_4 <= dStates(cntr+1,4)%< dStates(cntr+1,1)
            agentState_4 = cntr;
        else
        end
    end
    
end

% function to select action
function [Action_1,Action_2] = selectAction(Reps,par,state_1,state_2,state_3,state_4,numberOfActions)
    t = rand(1);
    if t <= par.epsilonVec(state_3,1)%par.epsilon
        % take random action
        Action_1 = randi(numberOfActions);
        Action_2 = randi(numberOfActions);
    elseif t > par.epsilonVec(state_3,1)%par.epsilon
        % take greedy action
        vec = Reps.action_value(state_1,state_2,state_3,state_4,:,:);
        index = find(ismember(vec(:),max(vec(:))));
        [~,~,~,~,Action_1,Action_2] = ind2sub(size(vec(:,:,:,:,:,:)), index);
        
    end
    
    % tie breaking
    if size(Action_1,1) > 1
        Action_1 = randi(numberOfActions);
    end
    
    if size(Action_2,1) > 1
        Action_2 = randi(numberOfActions);
    end
    %% old is above
    %% new
    Action_2 = 1;
end

% function to simulate the MDP
function [nxtCa,nxtCb,nxtT,nxtTc,nxtIAE] = simulateMDP(currentTimeStamp,prevModelStates,action_1,action_2,disturbanceValue,p,Arr,conversion,setPoint)
    MDPstart = currentTimeStamp;
    MDPstop = currentTimeStamp + 1;
    start = MDPstart*conversion;
    stop = MDPstop*conversion;
    tspan = linspace(start, stop, 10);
    p.T0 = p.T0s + disturbanceValue;
    p.VdivVr = action_1;
    p.Qc = action_2;
    [~, Output] = ode23s(@ (t, x) VdVmodel(t, x, p, Arr, setPoint), tspan, prevModelStates);
    nxtCa = Output(end,1);
    nxtCb = Output(end,2);
    nxtT = Output(end,3);
    nxtTc = Output(end,4);
    nxtIAE = Output(end,5);
    
end

% function to create the parameters for the RL environment's model
function rlEnv = createRewardShape(~)
    rlEnv.continuousReward = @(controlError) exp((-1*controlError^2)/0.01);
    rlEnv.binReward = @(controlError) -1 + 1*(controlError < 0.02) + 1*(controlError > -0.02);
    rlEnv.Terminal = -1000;
    
end

% function to map agent's action to the MV output of the MDP
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

% function to create vector of sampled DVs, SPs and times for these changes
% SPs and DVs sampled using scaled uniform distributions.
% SP and DV sampling times sampled using a normal distri
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
