%% Example:  Training of a tabular SARSA agent for the control of liquid
%% height in the water tank model (eqn 49 in thesis).  
%% S = [E(T),DV,SP]'; A = x;
clc
clear
clf
tic
%%
%% 
% seed random number generator to ensure exploration-exploitation strategy 
% is repeatable
rng(1)

plotTest = 1;   % set equal to 1 if results of agent test need to be plotted (not recommended when using many episodes)

%% parameters for sarsa agent
par.alpha = 0.7;
par.decay_for_epsilon = 1;        % decay factor to multiply with epsilon
par.gamma = 0.99;                 % discount factor

%% training settings and SP
training.nmberOfEps = 1380;%3000;       % number of episodes to use in training
training.nmberOfSteps = 1200;     % number of steps allowed per episode 
training.windowLength = 10;       % window size for summing up rewards
training.targetRunningReward = 5000; % minimum reward in window before stopping training (large because not intended stopping criterion)
training.decayInterval = 1;  % number of episodes before decaying probability of taking a random action

%% filter parameters
filter.sampletime = 1;
filter.tauf = 20;

%% flag for binary reward
binReward = 1; % set to 0 to use squared exponential reward function

%% create training environment (number of actions, states and reward shape)
beta = 0.5; % tolerance either side of SP for binary reward function
myEnvironment = createRewardShape(beta); % create RL environment SPECS

%% DEFINE RL ENVIRONMENT
%% model parameters
p.Atank = 1.8;  % cross-sectional area of the water tank (m^2)
p.g = 9.81;     % gravitational acceleration (m/s^2)
p.L = 0.5;      % pressure loss parameter (-)
p.cv = 0.01*60;    % valve discharge coefficient (m^2.5/min)

%% initial ss values
p.FinSS = (190/1000);         % inlet volumetric flow rate (m^3/s)
p.HSS = 1.5;           % height of the liquid in the tank (m)
p.k = 1.17*10^-3;      % lumped parameter for valve (m^2)
p.xSS = 0.259;         % fraction valve opening (-)

%% input and output variables to be assigned as DV and MV
p.Fin = p.FinSS;       % inlet flow rate (m^3/s)
p.x = p.xSS;           % fraction valve opening (-)

%% Beginning of sampling details
%% SP for component B and DV (reactor inlet temperature)
%% number of SP and DV changes per episode
rvOne.numSP = 10;
rvOne.numDV = 10;

%% lower and upper bounds for SP sampling (height of liquid in the tank (m))
rvOne.lower_SP = 0.5;
rvOne.upper_SP = 1.5;

%% lower and upper bounds for DV sampling (inlet flow rate (m^3/s))
rvOne.lower_DV = (100/1000);
rvOne.upper_DV = (200/1000);

%% integer value for random permutations determining when SP and DV steps will occur
rvOne.simTime = training.nmberOfSteps;

%% end of sampling details

%% initialize variable for recording rewards within averaging window
learning.rewardsInWindow = 0;

%% discretize states and actions and record extreme bounds of CV that may be instantiated
%% max number of states in one dimension = 10; minimum 1
%% fill in (number of states + 1) and (number of actions + 1) below
numError_intervals = 11;
numInletFin_intervals = 11;
numSP_intervals = 11;
numAction_intervals = 11; 

%% define lower bounds for discretizations
errorLow = -2;
inletFlowRateLow = 40*(1/1000);
SPLow = 0.4;

%% define "padding" for discretization
numSP_padding = SPLow.*ones((13 - (2 + numSP_intervals)),1)';
error_padding = errorLow.*ones((13 - (2 + numError_intervals)),1)';
inletFlowrate_padding = inletFlowRateLow.*ones((13 - (2 + numInletFin_intervals)),1)';

dActions = linspace(0.01,0.99,numAction_intervals)';

%% discretize states
% control error
fineRes = linspace(-2,2,numError_intervals);               
dStates = [-20,error_padding,fineRes,20]';

% inlet flow rate
fineInletFlowrateRes = linspace(100*(1/1000),200*(1/1000),numInletFin_intervals);
dStates(1:1:end,2) = [40*(1/1000),inletFlowrate_padding,fineInletFlowrateRes,210*(1/1000)]';

% height SP
fineHSPRes = linspace(0.50,1.5,numSP_intervals);
dStates(1:1:end,3) = [0.4,numSP_padding,fineHSPRes,2]';


par.epsilonVec = 0.1*ones(size(dStates,1),1);

%% create Q-table
% create action-value hypervolume
Reps.action_value = zeros([(size(dStates,1)-1),(size(dStates,1)-1), (size(dStates,1)-1),...
                  (size(dActions,1)-1)],'double'); 

setPoint = p.HSS;     % initialize height SP

p.testFlag = 0;     % indicate that agent is being trained

%% train sarsa-LPF controller
for episodeCntr = 1:training.nmberOfEps
    initialize_disturbanceValue = (190/1000);
    [learning_temp,temp_Reps] = train_agent(par,training,rvOne,learning,...
                                dStates,dActions,setPoint,myEnvironment,...
                                Reps,episodeCntr,filter,initialize_disturbanceValue,p,binReward);

    Reps = temp_Reps;
    out_learning{1,episodeCntr} = learning_temp.agentExp{1,episodeCntr};
    extreme_heights(episodeCntr) = max(-1*cell2mat(out_learning{1, episodeCntr}(:,1)) + cell2mat(out_learning{1, episodeCntr}(:,3)));
    
    fprintf('\n%d\n', episodeCntr);
    
end % end of episode loop

most_extreme_height = max(extreme_heights(:)); % record highest liquid level obtained during training

overflow_cntr = 0; % initialize variable used to calculate fraction of time spent in overflow condition
for i = 1:1:training.nmberOfEps
    for j = 1:1:(training.nmberOfSteps - 1)
        if (-1*cell2mat(out_learning{1, i}(j,1)) + ...
                cell2mat(out_learning{1, i}(j,3))) >= 3.0
            overflow_cntr = overflow_cntr + 1;
            
        end % end if statement
    end % end inner loop
end % end outer loop
total_MDP_steps = training.nmberOfEps*training.nmberOfSteps; % total number of discrete time steps
fraction_overflow = overflow_cntr/total_MDP_steps;           % fraction of time spent in overflow condition
% test trained agent
%%
% number of SP and DV changes per episode
rvTwo.numSP = 1;
rvTwo.numDV = 1;

% lower and upper bounds for SP sampling (height of liquid in the tank (m))
rvTwo.lower_SP = 1.5;
rvTwo.upper_SP = 1.5;

% lower and upper bounds for DV sampling (inlet flow rate (m^3/s))
rvTwo.lower_DV = 110/1000;
rvTwo.upper_DV = 110/1000;

% integer value for random permutations determining when SP and DV steps will occur
rvTwo.simTime = training.nmberOfSteps;

p.testFlag = 1;     % indicate that controller is being tested
p.SP_Time = 500;    % time for test SP change
p.DV_Time = 500;    % time for test DV change

for sim_episodeCntr = 1:1
    sim_disturbanceValue = (190/1000); % initialize Fin for controller test
    [learning_sim,sim_Reps] = train_agent(par,training,rvTwo,learning,...
                                dStates,dActions,setPoint,myEnvironment,...
                                Reps,sim_episodeCntr,filter,sim_disturbanceValue,p,binReward);
    sim_Reps = sim_Reps;
    sim_learning = learning_sim.agentExp{1,sim_episodeCntr};
end % end of episode loop

if plotTest == 1
    %% SP
    subplot(1,3,1)
    plot(cell2mat(learning_sim.agentExp{1, 1}(:,3)), 'ko')
    xlabel('Time (min)'); ylabel('H SP (m)')
    set(gca,'fontsize',20)
    %% DV
    subplot(1,3,2)
    plot(cell2mat(learning_sim.agentExp{1, 1}(:,2)), 'ko')
    xlabel('Time (min)'); ylabel('Fin (m^3/min)')
    set(gca,'fontsize',20)
    %% rewards
    subplot(1,3,3)
    plot(cell2mat(learning_sim.agentExp{1, 1}(:,end)), 'ko')
    xlabel('Time (min)'); ylabel('Reward')
    set(gca,'fontsize',20)

    
    %% general formatting
    set(gcf,'color', 'white')
    
    figure
    %% discrete action selections
    subplot(1,2,1)
    plot(cell2mat(learning_sim.discreteSelections{1, 1}(:,end)),'kx','LineWidth', 2, 'MarkerSize', 6)
    hold on 
    %% filtered action selections
    plot(cell2mat(sim_learning(:,4)), 'ko', 'LineWidth', 3, 'MarkerSize', 6)
    xlabel('Time (min)'); ylabel('x (-)')
    set(gca,'fontsize',20)
    hold on
    xline(p.DV_Time, 'k-.', 'LineWidth', 2)
    lgnd = legend('unfiltered', 'filtered', 'step change');
    lgnd.FontSize = 14;
    
    %% CV behaviour
    subplot(1,2,2)
    plot(rvTwo.lower_SP - cell2mat(learning_sim.agentExp{1, 1}(:,1)),'ko')
    hold on
    plot(cell2mat(learning_sim.agentExp{1, 1}(:,3)), 'k--', 'LineWidth', 3)
    hold on
    plot(cell2mat(learning_sim.agentExp{1, 1}(:,3)) + beta, 'k--', 'LineWidth', 1)
    hold on
    plot(cell2mat(learning_sim.agentExp{1, 1}(:,3)) - beta, 'k--', 'LineWidth', 1)
    xlabel('Time (min)'); ylabel('H (m)')
    set(gca,'fontsize',20)
    hold on
    xline(p.DV_Time, 'k-.', 'LineWidth', 2)
    legend('H (m)', 'SP (m)', 'SP + \beta', 'SP - \beta')
    
    %% general formatting
    set(gcf,'color', 'white')
    
end

summedTestRewards = sum(cell2mat(learning_sim.agentExp{1, 1}(:,end)),'all'); % total rewards received during test
% delete(myPool)
toc

%% functions
%%

% training function
function [learning,Reps] = train_agent(par,training,rvOne,learning,dStates,dActions,setPoint,myEnvironment,Reps,episodeCntr,filter,disturbanceValue,p,binReward)
    if learning.rewardsInWindow < training.targetRunningReward
        clear State
        clear Action
        currentTimeStamp = 1;
        
        State(currentTimeStamp) = setPoint - p.HSS; % initial control error w.r.t. H
        
        %learning.prevCB = setPoint - State(currentTimeStamp);
        
        %% sample SP for height and DV (inlet flow rate)
        [SP_times,SP_steps,DVOne_times,DVOne_steps] = generateEpRVSamples(rvOne,training.nmberOfSteps);
        
        if p.testFlag == 1
            SP_times(:) = p.SP_Time;       % manually set SP change time
            DVOne_times(:) = p.DV_Time;    % manually set DV change time
        end
        
        for stepCntr = 1:1:training.nmberOfSteps

            %% adjust setPoint and DV as per sampling
            for SP_cntr = 1:1:rvOne.numSP
                if SP_times(SP_cntr) == currentTimeStamp%mod(SP_times(SP_cntr),currentTimeStamp) == 0
                    setPoint = SP_steps(SP_cntr);
                    
                end
                
            end
            
            
            %% adjust disturbanceValue 
            for DV_cntr = 1:1:rvOne.numDV
                if DVOne_times(DV_cntr) == currentTimeStamp%mod(DVOne_times(DV_cntr),currentTimeStamp) == 0
                    disturbanceValue = DVOne_steps(DV_cntr); 
                    
                end
            end
            %% end adjustment of SP and DV as per sampling
            %% state component definitions
            State_1(currentTimeStamp) = State(currentTimeStamp);%setPoint - ss(2); % control error
            State_2(currentTimeStamp) = disturbanceValue;
            State_3(currentTimeStamp) = setPoint; % H set point

            if  currentTimeStamp < training.nmberOfSteps
                % map current MDP state to corresponding state number for
                % the agent
                [learning.crntAgentState_1,learning.crntAgentState_2,...
                    learning.crntAgentState_3] = mapStatesToAgent(State_1(currentTimeStamp),...
                    State_2(currentTimeStamp),State_3(currentTimeStamp),dStates);
                
                % let agent select the next action 
                learning.crntAgentAction_1 = ...
                    selectAction(Reps,par,learning.crntAgentState_1,...
                    learning.crntAgentState_2,learning.crntAgentState_3,...
                    (size(dActions,1)-1));
                
                % action (true MV control action to take)
                % Action(currentTimeStamp) = mapToMDP(learning.crntAgentAction,dActions,Reps);
                Action_1(currentTimeStamp) = mapToMDP(learning.crntAgentAction_1,dActions);
                % filter agent's selected action (initialize with first
                % true agent output)
                if currentTimeStamp > 1
                    previousSmoothed_1 = Action_1(currentTimeStamp - 1);

                elseif currentTimeStamp == 1
                    previousSmoothed_1 = Action_1(currentTimeStamp);

                end
                %% store unfiltered actions during training
                learning.discreteSelections{1,episodeCntr}{stepCntr,3} = Action_1(currentTimeStamp);

                %% store SP and DV step info
                learning.SP{1,episodeCntr}{stepCntr,1} = setPoint;
                learning.DV{1,episodeCntr}{stepCntr,1} = disturbanceValue;
                
                % perform digital filter calculation
                Action_1(currentTimeStamp) = digitalFilter(filter.sampletime,filter.tauf,previousSmoothed_1,Action_1(currentTimeStamp));

                if currentTimeStamp == 1
                    prevModelStates = p.HSS;
                    
                else
                    prevModelStates = nxtH;
                    
                end
                
                % obtain next height from water tank model
                nxtH = ...
                    simulateMDP(currentTimeStamp,prevModelStates,...
                    Action_1(currentTimeStamp),...
                    disturbanceValue,p);
                
                learning.nxtState_1 = setPoint - nxtH; % calculate next control error
                % Note that next timestep is only incorporated for outputs
                % resulting from interaction with the RL environment
                learning.nxtState_2 = State_2(currentTimeStamp);%disturbanceValue + p.T0s;
                learning.nxtState_3 = State_3(currentTimeStamp);%setPoint;
                % map next MDP state to corresponding state number for the
                % agent
                [learning.nxtAgentState_1,learning.nxtAgentState_2,...
                    learning.nxtAgentState_3] = mapStatesToAgent(learning.nxtState_1,...
                    learning.nxtState_2,learning.nxtState_3,dStates);
                % let agent select next action
                % learning.nxtAgentAction = selectAction(Reps,par,learning.nxtAgentState,myEnvironment);
                learning.nxtAgentAction_1 = ...
                    selectAction(Reps,par,learning.nxtAgentState_1,...
                    learning.nxtAgentState_2,learning.nxtAgentState_3,(size(dActions,1)-1));
                % obtain reward
                %nxtAction = mapToMDP(learning.nxtAgentAction,dActions,Reps);
                learning.nxtAction_1 = mapToMDP(learning.nxtAgentAction_1,dActions);
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
                learning.agentExp{1,episodeCntr}{stepCntr,5} = learning.Reward;
                
                % update coordinate in action-value hypervolume
                if learning.nxtState_1 ~= myEnvironment.Terminal
                    % non-terminal update
                    Reps.action_value(learning.crntAgentState_1,...
                               learning.crntAgentState_2,learning.crntAgentState_3,learning.crntAgentAction_1) = ...
                               Reps.action_value(learning.crntAgentState_1,...
                               learning.crntAgentState_2,learning.crntAgentState_3,learning.crntAgentAction_1) + ...
                               par.alpha*(learning.Reward + ...
                               par.gamma*Reps.action_value(learning.nxtAgentState_1,...
                               learning.nxtAgentState_2,learning.nxtAgentState_3,learning.nxtAgentAction_1) - ...
                               Reps.action_value(learning.crntAgentState_1,...
                               learning.crntAgentState_2,learning.crntAgentState_3,...
                               learning.crntAgentAction_1));
                        
                elseif learning.nxtState == myEnvironment.Terminal
                    % terminal update
                    Reps.action_value(learning.crntAgentState_1,...
                               learning.crntAgentState_2,learning.crntAgentState_3,learning.crntAgentAction_1) = ...
                               Reps.action_value(learning.crntAgentState_1,...
                               learning.crntAgentState_2,learning.crntAgentState_3,learning.crntAgentAction_1) + ...
                               par.alpha*(learning.Reward -...
                               Reps.action_value(learning.crntAgentState_1,...
                               learning.crntAgentState_2,learning.crntAgentState_3,...
                               learning.crntAgentAction_1));
                           
                end % end value function update
            
            % shift time step T <- (T+1)
            currentTimeStamp = currentTimeStamp + 1;
            % shift state S(T) <- S(T+1)
            %% the other states are updated when sampling DV and SP data
            State(currentTimeStamp) = learning.nxtState_1;
            
            end % end loop for non-terminal state
    
        end % end loop for steps

        
    end % end rewards in window if-statement
    % Store number of non-terminal entries of each training episode
%     optimalityCntr(episodeCntr,1) = currentTimeStamp;
%     fprintf('%d\n',episodeCntr) % display episode number to give indication of training progress
    
    if episodeCntr < training.nmberOfEps && mod(episodeCntr,training.decayInterval) == 0
       %par.epsilon = par.epsilon*par.decay_for_epsilon; % decay probability for taking random action
       par.epsilonVec(learning.crntAgentState_3,1) = par.epsilonVec(learning.crntAgentState_3,1)*par.decay_for_epsilon;
    end

end

%%

% function to map true states of MDP to coded states of agent
function [agentState_1,agentState_2,agentState_3] = mapStatesToAgent(state_1,state_2,state_3,dStates)
    % identify coded state component 1 (control error E(T))
    for cntr = 1:1:(size(dStates,1)-1)
        if state_1 > dStates(cntr,1) && state_1 <= dStates(cntr+1,1)%< dStates(cntr+1,1)
            agentState_1 = cntr;
        else
        end
    end
 
    % identify coded state component 2 (inlet flow rate)
    for cntr = 1:1:(size(dStates,1)-1)
        if state_2 > dStates(cntr,2) && state_2 <= dStates(cntr+1,2)%< dStates(cntr+1,1)
            agentState_2 = cntr;
        else
        end
    end

    % identify coded state component 3 (H setpoint)
    for cntr = 1:1:(size(dStates,1)-1)
        if state_3 > dStates(cntr,3) && state_3 <= dStates(cntr+1,3)%< dStates(cntr+1,1)
            agentState_3 = cntr;
        else
        end
    end
    
end


%%

% function to select action
function Action_1 = selectAction(Reps,par,state_1,state_2,state_3,numberOfActions)
    t = rand(1);
    if t <= par.epsilonVec(state_3,1)%par.epsilon
        % take random action
        Action_1 = randi(numberOfActions);

    elseif t > par.epsilonVec(state_3,1)%par.epsilon
        % take greedy action
        vec = Reps.action_value(state_1,state_2,state_3,:);
        index = find(ismember(vec(:),max(vec(:))));
        [~,~,~,Action_1] = ind2sub(size(vec(:,:,:,:)), index);
        
    end
    
    % tie breaking
    if size(Action_1,1) > 1
        Action_1 = randi(numberOfActions);
    end
    
end

%%

% function to simulate the MDP
function [nxtH] = simulateMDP(currentTimeStamp,prevModelStates,action_1,disturbanceValue,p)
    MDPstart = currentTimeStamp;
    MDPstop = currentTimeStamp + 1;
    tspan = linspace(MDPstart, MDPstop, 10);
    p.Fin = p.FinSS + disturbanceValue;
    p.x = action_1;
    [~, HOutput] = ode45(@ (t, H) tankModel(t,H,p), tspan, prevModelStates);
    nxtH = HOutput(end,1);
    
end


%%

% function to model the water tank
function dHdt = tankModel(~,H,p)
%water tank model
dHdt = p.Fin - p.cv*p.x*sqrt(H);

end

%%

% function to create the parameters for the RL environment's model
function rlEnv = createRewardShape(beta)
    rlEnv.continuousReward = @(controlError) exp((-1*controlError^2)/0.01);
    rlEnv.binReward = @(controlError) -1 + 1*(controlError < beta) + 1*(controlError > -beta);
    rlEnv.Terminal = -1000;
    
end

%%

% function to map agent's action to the MV output of the MDP
function MDPAction_1 = mapToMDP(agentActionIndex_1,dActions)
   
   MDPAction_1 = dActions(agentActionIndex_1,1);
   
end

%%

%  digital low-pass filter (chapter 12 of MARLIN)
function smoothedAction = digitalFilter(sampleTime,tauf,previousSmoothed,currentAction)
    A = exp(-1*sampleTime/tauf);
    % time domain digital filter calculation
    smoothedAction = A*previousSmoothed + (1-A)*currentAction;
    
end

%%

% function to create vector of sampled DVs, SPs and times for these changes
% SPs and DVs sampled using scaled uniform distributions.
% SP and DV sampling times sampled using a normal distri
function [SP_times,SP_steps,DV_times,DV_steps] = generateEpRVSamples(rv,upperTimeBound)
    %% generate SP sampling data
    SP_times = sort( ceil(1 + upperTimeBound*rand(1,rv.numSP)) );
    for SP_cntr = 1:1:(rv.numSP)
        SP_steps(SP_cntr) = rv.lower_SP + (rv.upper_SP - rv.lower_SP)*rand(1,1);
        
    end

    %% generate DV sampling data
    DV_times = sort( ceil(1 + upperTimeBound*rand(1,rv.numDV)) );
    for DV_cntr = 1:1:(rv.numDV)
        DV_steps(DV_cntr) = rv.lower_DV + (rv.upper_DV - rv.lower_DV)*rand(1,1);
        
    end
    
end