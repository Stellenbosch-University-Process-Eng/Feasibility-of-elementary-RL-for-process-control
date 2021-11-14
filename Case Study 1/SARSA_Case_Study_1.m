%% short description
% sarsa control applied to water tank model (SP-changes during training).
% Date: 2020/09/14
%%
% Revision: 1
%% clear command window and workspace
clc
%clear
%% 
% seed random number generator to ensure exploration-exploitation strategy 
% is repeatable

tic
rng(1)
%% START OF USER INPUT
%% process gain (used to generate data indicating what the minimum and maximum ss bounds of the agents action-selection could be.
process.Kp = -20.4;  % process gain (m/(sm^2/min)) from PRC 

%% parameters for sarsa agent
par.alpha = 0.7; % step size parameter
par.epsilon = 0.8; % initial probability of taking a random action
par.decay_for_epsilon = 0.99; % decay factor to multiply with epsilon
par.gamma = 0.7; % discount factor

%% training settings and SP
training.nmberOfEps = 3000;       % number of episodes to use in training
training.nmberOfSteps = 1200;     % number of steps allowed per episode 
setPoint = 1.4; % set point height in tank (m)
setPointVec = [1 2]';           % vector of SPs to select "randomly" at the start of each episode
simSetPoint = setPoint;           % setPoint used if only simulating
training.firstHeightVec = [1.4]'; % vector of starting heights for episodes
training.firstStateVec = simSetPoint - training.firstHeightVec; % first control error
training.windowLength = 10;      % window size for summing up rewards
training.targetRunningReward = 5000; % minimum reward in window before stopping training (large because not intended stopping criterion)
training.decayInterval = 1;  % number of episodes before decaying probability of taking a random action

%% simulation settings
sim.nmberOfSteps = 300; % number of time steps (min) to simulate trained agent
sim.startingState = 0;  % initialize control error for simulation

%% filter parameters
filter.sampletime = 1;
filter.tauf = 20; 
%% reward bandwidth setting
rewardBandwidth = 1;

%% discretize possible states and actions (can lead to error if not sufficient to cover all encountered states)
% state bounds (note that only single state component is applicable in this application) 
MDPstateVecLow = -15;           % vector for lower bounds of all state components
MDPstateVecHigh = 15;           % vector for upper bounds of all state components
statesResAvailable = 14;%110;%14;%15;        % number of discrete states = (statesRes - 1)
% action bounds as vectors or scalars
%% NB -> ACTION BOUNDS ARE DEFINED LATER
%% THEREFORE, ONLY LINE 59 IS USED, ACTIONS ARE DEFINED IN LINE 82
MDPactionVecLow = 0;            % vector for lower bounds of all action components
MDPactionVecHigh = 0.15;        % vector for upper bounds of all action components
actionResAvailable = 11;%4;         % actions available = (actionResAvailable - 1)

%% create training environment
A = 1.8; % tank area (m^2)
L = 0.5; % pressure loss parameter (dimensionless)
myEnvironment = createMDP(rewardBandwidth,A,L,actionResAvailable,statesResAvailable,setPoint); % create RL environment SPECS
disturbanceValue = 190;
simDisturbanceValue = 200;
%% specify or agent should be trained
doTraining = 1; % specify as 1 if agent should be trained from scratch, otherwise provide trained critic at end of script

%% specify continuous or binary reward structure
binaryReward = 1;
%% END OF USER INPUT

%%
if doTraining == 1  %% added if statement on 2021-07-17 (rather load variables)
    %% create Q-table
    Reps.action_value = createRep(myEnvironment);
    action_value_initial = Reps.action_value;
end

%% initialize variable for recording rewards within averaging window
learning.rewardsInWindow = 0;
%% discretize states and actions and record extreme bounds of CV that may be instantiated
% create discrete set of actions for agent to choose from
[dStates,~] = discretize(MDPstateVecLow,MDPstateVecHigh,...
                  MDPactionVecLow,MDPactionVecHigh,statesResAvailable,actionResAvailable);
               
%%
dActions = linspace(0.001,0.2,(actionResAvailable-1))'; %% use for L_A runs

%% Q-learning dActions and dStates
    % CASE 1
dActions = linspace(0.001,0.2,10)'; %% defined on 2021-07-18 (used for runs with Q-learning)
dStates = [-150,linspace(-2.14,2.14,11),150]';

    % CASE 2
dActions = linspace(0.001,0.2,10)'; %% defined on 2021-07-18
dStates = [150,linspace(-2.14,2.14,101),150]'; %% defined on 2021-07-18

%% process.smallestSSEffect = (dActions(2)-dActions(1))*process.Kp;
%% process.largestSSEffect = (dActions(end)-dActions(1))*process.Kp;

%% CODE ADDED ON 2021_07_16 to more directly define dStates and dActions (this is the fine case used for concept validation)
%% coarse case is the original dStates code
%% lines below are for first set of runs with beta = 0.5
%dStates = [-150,linspace(-2.14,2.14,11),150]';
%dActions = linspace(0.001,0.2,10)';

%% Beta = 0.20 runs
%% LS02HA run:
dStates = [-150,linspace(-2.14,2.14,11),150]';
dActions = linspace(0.001,0.2,10)'; %% defined on 2021-07-18
%% HS02HA run:
dStates = [-150,linspace(-2.14,2.14,101),150]';
dActions = linspace(0.001,0.2,10)'; %% defined on 2021-07-18

%% Beta = 0.01 run
dStates = [-150,linspace(-2.14,2.14,11),150]';
dActions = linspace(0.001,0.2,10)'; %% defined on 2021-07-25

%% train agent
if doTraining == 1
%% train sarsa-LPF controller
for episodeCntr = 1:1:training.nmberOfEps
    if learning.rewardsInWindow < training.targetRunningReward
        clear State
        clear Action
        currentTimeStamp = 1;
        State(currentTimeStamp) = training.firstStateVec(randi(length(training.firstStateVec)));
        learning.prevHeight = setPoint - State(currentTimeStamp);
        setPoint = setPointVec(randi(length(setPointVec))); % sample a SP from setPointVec
        for stepCntr = 1:1:training.nmberOfSteps
            %  perform training in episode if terminal state has not been
            %  achieved
            if State(currentTimeStamp) ~= myEnvironment.Terminal &&...
                                   currentTimeStamp < training.nmberOfSteps
                % map current MDP state to corresponding state number for
                % the agent
                learning.crntAgentState = mapStatesToAgent(State(currentTimeStamp),dStates);
                % let agent select the next action 
                learning.crntAgentAction = selectAction(Reps,par,learning.crntAgentState,myEnvironment);
                % action (true MV control action to take)
                Action(currentTimeStamp) = mapToMDP(learning.crntAgentAction,dActions,Reps);
                % filter agent's selected action (initialize with first
                % true agent output)
                if currentTimeStamp > 1
                    previousSmoothed = Action(currentTimeStamp - 1);
                elseif currentTimeStamp == 1
                    previousSmoothed = Action(currentTimeStamp);
                end
                % perform digital filter calculation
                Action(currentTimeStamp) = digitalFilter(filter.sampletime,filter.tauf,previousSmoothed,Action(currentTimeStamp));
                % simulate RL environment to obtain next height in the tank
                learning.nxtHeight = simulateMDP(currentTimeStamp,...
                                  learning.prevHeight,...
                                  Action(currentTimeStamp),...
                                  myEnvironment,setPoint,disturbanceValue);
                learning.nxtState = setPoint - learning.nxtHeight; % calculate next control error
                % map next MDP state to corresponding state number for the
                % agent
                learning.nxtAgentState = mapStatesToAgent(learning.nxtState,dStates);
                % let agent select next action
                learning.nxtAgentAction = selectAction(Reps,par,learning.nxtAgentState,myEnvironment);
                % obtain reward
                nxtAction = mapToMDP(learning.nxtAgentAction,dActions,Reps);
                maxMV = MDPactionVecHigh; % not used in reward function anymore (only distance from SP communicated)
                controlError = learning.nxtState;
                if binaryReward == 0
                    learning.Reward = myEnvironment.Reward(controlError,...
                                    setPoint,nxtAction,...
                                    Action(currentTimeStamp),maxMV);
                elseif binaryReward == 1
                    learning.Reward = myEnvironment.binReward(controlError);
                end
                % store training information
                learning.agentExp{1,episodeCntr}{stepCntr,1} = State(currentTimeStamp);
                learning.agentExp{1,episodeCntr}{stepCntr,2} = Action(currentTimeStamp);
                learning.agentExp{1,episodeCntr}{stepCntr,3} = learning.Reward;
                % update critic
                if learning.nxtState ~= myEnvironment.Terminal
                    % non-terminal update to action-value table (Q-table)
                    Reps.action_value(learning.crntAgentState,...
                            learning.crntAgentAction) = ...
                               Reps.action_value(learning.crntAgentState,...
                               learning.crntAgentAction) + ...
                               par.alpha*(learning.Reward + ...
                               par.gamma*Reps.action_value(learning.nxtAgentState,learning.nxtAgentAction)...
                               -Reps.action_value(learning.crntAgentState,learning.crntAgentAction));
                        
                elseif learning.nxtState == myEnvironment.Terminal
                    % terminal update to action-value table (Q-table)
                    Reps.action_value(learning.crntAgentState,...
                            learning.crntAgentAction) = ...
                               Reps.action_value(learning.crntAgentState,...
                               learning.crntAgentAction) + ...
                               par.alpha*(learning.Reward - ...
                               Reps.action_value(learning.crntAgentAction,learning.crntAgentAction));
                end % end critic update
          
            % shift time step T <- (T+1)
            currentTimeStamp = currentTimeStamp + 1;
            % shift state S(T) <- S(T+1)
            State(currentTimeStamp) = learning.nxtState;
            learning.prevHeight = learning.nxtHeight;
            
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
            rewardCount = sum(tempMAT(:,3));
        end
        stepCounts = stepCounts + residualSteps;
        windowData{(episodeCntr/training.windowLength),1} = rewardCount;
        windowData{(episodeCntr/training.windowLength),2} = stepCounts;
        learning.rewardsInWindow = rewardCount;
        learning.stepsInWindow = stepCounts;
        startPoint = size(learning.agentExp,2)+1; % shift startpoint for calculating statistics of next window
        
        end % end calculation of window statistics
        
    end % end rewards in window if-statement
    % Store number of non-terminal entries of each training episode
    optimalityCntr(episodeCntr,1) = currentTimeStamp;
    fprintf('%d\n',episodeCntr) % display episode number to give indication of training progress
    
    if episodeCntr < training.nmberOfEps && mod(episodeCntr,training.decayInterval) == 0
       par.epsilon = par.epsilon*par.decay_for_epsilon; % decay probability for taking random action 
    end
    if episodeCntr == 1
        learning.cumulativeIAE = 0; % initialize cumulative IAE calculation across training episodes
    end
    learning.cumulativeIAE = learning.cumulativeIAE + calculateIAE(learning,episodeCntr);
    
end % end of episode loop

%% display training results
learning.averageRewardTrajec = cell2mat(windowData(:,1));
learning.averageStepsTrajec = cell2mat(windowData(:,2));
subplot(2,4,1)
plot(cell2mat(windowData(:,1)),'ko','LineWidth',2,'MarkerSize',5) % plot moving average rewards
xlabel('averaging window number'); ylabel('Reward in window');
title('rewards in windows')
subplot(2,4,2)
plot(cell2mat(windowData(:,2)),'ko','LineWidth',2,'MarkerSize',5) % plot moving average steps
xlabel('averaging window number'); ylabel('Average number of steps');
title('steps in windows')
learning.windowData = windowData; % store moving average window data generated 
% Test for terminal state not being reached:
[optimalEps,~] = find(ismember(optimalityCntr,min(optimalityCntr(:))));
optimalEpisode = optimalEps(1);
if optimalEpisode == 1
    fprintf('\n Warning: number of steps between start state and terminal state the same for all episodes.\n');
end
initialMAT = cell2mat(learning.agentExp{1,1}); % first episode's experiences of agent 
subplot(2,5,3)
plot(initialMAT(:,1), '-.b', 'LineWidth', 2); % errors in tank in first episode
xlabel('time (min)'); ylabel('SP - CV (m)');
title('Errors: first episode'); hold on
subplot(2,4,4)
plot(initialMAT(:,2), 'ko', 'LineWidth', 2); % actions taken by agent in first episode
xlabel('time (min)');ylabel('filtered actions ((sm^2)/min)');
title('filtered actions: first episode');
subplot(2,4,5)
plot(initialMAT(:,3), '-.r', 'LineWidth', 2); % rewards obtained by agent in first episode
xlabel('time (min)'); ylabel('Rewards (first episode)');
title('rewards: first episode');
finalMAT = cell2mat(learning.agentExp{1,episodeCntr}); % final episode's experiences of agent
subplot(2,4,6)
plot(finalMAT(:,1), '-.b', 'LineWidth', 2); % errors in tank in final episode
xlabel('time (min)'); ylabel('SP - CV (m)');
title('Errors: last episode'); hold on
subplot(2,4,7)
plot(finalMAT(:,2), 'ko', 'LineWidth', 2); % actions taken by agent in final episode
xlabel('time (min)');ylabel('filtered actions ((sm^2)/min)');
title('filtered actions: last episode');
subplot(2,4,8)
plot(finalMAT(:,3), '-.r', 'LineWidth', 2); % rewards obtained in final episode
xlabel('time (min)'); ylabel('Rewards (last episode)');
title('rewards: last episode');
figure
subplot(2,1,1)
heatmap(dStates);
title('bounds for states (Control error in meters)');ylabel('Bounds');
subplot(2,1,2)
heatmap(dActions);
title('bounds for actions (valve variable (sm^2/min))');ylabel('Bounds');
figure
subplot(2,1,1)
heatmap(action_value_initial);
xlabel('actions');ylabel('states')
title('initial action-value table')
subplot(2,1,2)
heatmap(Reps.action_value);
xlabel('actions');ylabel('states')
title('trained action-value table')
figure
    
end

if doTraining ~= 1
    % load pre-trained action-value table and corresponding epsilon
    %% [Reps.action_value,par.epsilon] = preTrainedRep; % commented on 2021-07-17 (rather load variables)
    par.epsilon = 10^-4; % commented on 2021-07-17 (rather load variables)
end
setPoint = simSetPoint;
%% simulate sarsa-LPF controller
for episodeCntr = 1:1:1
    if learning.rewardsInWindow < training.targetRunningReward
        clear State
        clear Action
        currentTimeStamp = 1;
        State(currentTimeStamp) = sim.startingState;
        simulation.prevHeight = setPoint - sim.startingState;
        for stepCntr = 1:1:sim.nmberOfSteps
            %  perform training in episode if terminal state has not been
            %  achieved
            if State(currentTimeStamp) ~= myEnvironment.Terminal &&...
                                   currentTimeStamp < sim.nmberOfSteps 
                % map current MDP state to corresponding state number for
                % the agent
                simulation.crntAgentState = mapStatesToAgent(State(currentTimeStamp),dStates);
                % let agent select the next action 
                simulation.crntAgentAction = selectAction(Reps,par,simulation.crntAgentState,myEnvironment);
                % action (true MV control action to take)
                Action(currentTimeStamp) = mapToMDP(simulation.crntAgentAction,dActions,Reps);
                % filter agent's selected action (initialize with first
                % true agent output)
                if currentTimeStamp > 1
                    previousSmoothed = Action(currentTimeStamp - 1);
                elseif currentTimeStamp == 1
                    previousSmoothed = Action(currentTimeStamp);
                end
                Action(currentTimeStamp) = digitalFilter(filter.sampletime,...
                                         filter.tauf,previousSmoothed,...
                                         Action(currentTimeStamp));
                simulation.nxtHeight = simulateMDP(currentTimeStamp,...
                                  simulation.prevHeight,...
                                  Action(currentTimeStamp),...
                                  myEnvironment,setPoint,simDisturbanceValue);
                simulation.nxtState = setPoint - simulation.nxtHeight;
                % map next MDP state to corresponding state number for the
                % agent
                simulation.nxtAgentState = mapStatesToAgent(simulation.nxtState,dStates);
                % let agent select next action
                simulation.nxtAgentAction = selectAction(Reps,par,simulation.nxtAgentState,myEnvironment);
                nxtAction = mapToMDP(simulation.nxtAgentAction,dActions,Reps);
                maxMV = MDPactionVecHigh;
                % obtain reward
                controlError = simulation.nxtState;
                if binaryReward == 0
                    simulation.Reward = myEnvironment.Reward(controlError,...
                                    setPoint,nxtAction,...
                                    Action(currentTimeStamp),maxMV);
                elseif binaryReward == 1
                    simulation.Reward = myEnvironment.binReward(controlError);
                end
                
                % store training information
                simulation.agentExp{1,episodeCntr}{stepCntr,1} = State(currentTimeStamp);
                simulation.agentExp{1,episodeCntr}{stepCntr,2} = Action(currentTimeStamp);
                simulation.agentExp{1,episodeCntr}{stepCntr,3} = simulation.Reward;
                simulation.agentActions{1,episodeCntr}{stepCntr,1} = simulation.crntAgentAction;
                % shift time step T <- (T+1)
                currentTimeStamp = currentTimeStamp + 1;
                % shift state S(T) <- S(T+1)
                State(currentTimeStamp) = simulation.nxtState;
                simulation.prevHeight = simulation.nxtHeight;
                
            end % end loop for non-terminal state              
   
        end % end loop for steps
        
        % Calculate reward and steps for window specified in training structure
        if mod(episodeCntr,training.windowLength) == 0
            rewardCount = 0;
            stepCounts = 0;
            residualSteps = 0;
        if (episodeCntr/training.windowLength) == 1
            startPoint = 1;
        end
        for cntr = startPoint:1:size(simulation.agentExp,2)
            if size(simulation.agentExp{1,cntr},1) == (sim.nmberOfSteps - 1)
                    residualSteps = residualSteps + 1;
            end
            stepCounts = stepCounts + size(simulation.agentExp{1,cntr},1);
            % assume that the episode's goal is achieved if number of steps is less than (maximum steps allowed per episode - 1).
            tempMAT = cell2mat(simulation.agentExp{1,cntr});
            rewardCount = sum(tempMAT(:,3)); % calculate sum of rewards in window
        end
            stepCounts = stepCounts + residualSteps;
            simWindowData{(episodeCntr/training.windowLength),1} = rewardCount;
            simWindowData{(episodeCntr/training.windowLength),2} = stepCounts;
            simulation.rewardsInWindow = rewardCount;
            simulation.stepsInWindow = stepCounts;
            startPoint = size(simulation.agentExp,2)+1;
        
        end % end calculation of window statistics
        
    end % end rewards in window if-statement
    % Store number of non-terminal entries of each training episode
    optimalityCntr(episodeCntr,1) = currentTimeStamp;
    
end % end of episode loop
%% display simulation results
subplot(2,2,1)
plot(1:1:size(cell2mat(simulation.agentExp{1,1}(:,1))),cell2mat(simulation.agentExp{1,1}(:,1)),'ko');
xlabel('time (min)'); ylabel('SP - CV (m)'); title('control errors during simulation');
subplot(2,2,2)
plot(1:1:size(cell2mat(simulation.agentExp{1,1}(:,2))),cell2mat(simulation.agentExp{1,1}(:,2)),'ko');
xlabel('time (min)'); ylabel('filtered agent actions (sm^2/min)'); title('filtered agent actions')
subplot(2,2,3)
plot(1:1:size(cell2mat(simulation.agentExp{1,1}(:,3))),cell2mat(simulation.agentExp{1,1}(:,3)),'ko');
xlabel('time (min)'); ylabel('rewards'); title('rewards obtained')

%% Calculate IAE and max error
IAE = calculateIAEEp(simulation); % IAE calculation for simulation 
fprintf('IAE during simulation = %.2f\n', IAE) % Display IAE
temp = cell2mat(simulation.agentExp{1,1});
maximumDev = max(abs(temp(:,1))) % calculate maximum deviation in tank
finalError = temp(end,1)
% display liquid heights in tank during simulation
heights = setPoint - temp(:,1);
subplot(2,2,4)
plot(1:1:size(heights,1),heights,'ko')
xlabel('time (min)'); ylabel('heights (m)'); title('heights during simulation')

toc
%% functions

% function to create action-value table
function Qtable = createRep(myEnvironment)
%    Qtable = 0.01*rand(myEnvironment.numberOfStates,myEnvironment.numberOfActions);
Qtable = zeros(myEnvironment.numberOfStates,myEnvironment.numberOfActions);
end

% function to map states to coded state numbers for agent
function agentState = mapStatesToAgent(state,dStates)

    for cntr = 1:1:(size(dStates,1)-1) +1 %% added plus one on 2021-07-16 for easier dState and dAction
        if state > dStates(cntr,1) && state <= dStates(cntr+1,1)%< dStates(cntr+1,1)
            agentState = cntr;
        else
        end
    end
end

% function to select action
function Action = selectAction(Reps,par,state,myEnvironment)
    t = rand(1);
    if t <= par.epsilon
        % take random action
        Action = randi(myEnvironment.numberOfActions);
    elseif t > par.epsilon
        % take greedy action
        Action = find(ismember(Reps.action_value(state,:),max(Reps.action_value(state,:))));
    end
    
    % tie breaking
    if size(Action,2) > 1
        Action = randi(myEnvironment.numberOfActions);
    end
    
end

% function to simulate the MDP
function nextHeight = simulateMDP(currentTimeStamp,s,a,myEnvironment,setPoint,disturbanceValue)
    start = currentTimeStamp;
    stop = currentTimeStamp + 1;
    startingH = s;
    tspan = linspace(start, stop, 10);
    [~,H] = ode45(@(t,H) ODE_tank(t,H,myEnvironment,a,disturbanceValue),tspan,startingH);
    nextHeight = H(end);

end

% function that contains the model of the MDP
function dHdt = ODE_tank(~,H,myEnvironment,action,disturbanceValue)
    % tank model
    dHdt = ((disturbanceValue/1000) - ...
         action*sqrt(myEnvironment.g*H*(1-myEnvironment.L)))/myEnvironment.A;

end

% function to create the parameters for the RL environment's model
function rlEnv = createMDP(rewardBandwidth,tankArea,lossParameter,actionRes,statesRes,~)

    rlEnv.A = tankArea; % tank area in m^2
    rlEnv.L = lossParameter; % pressure loss parameter
    rlEnv.g = 9.81; % gravitational acceleration in m/s^2
    rlEnv.Reward = @(controlError,SP,action2,action1,MVmax) 1*...
                 exp(-1*(controlError)^2/(rewardBandwidth)) + ...
                 0*SP; %+ 10*(abs(controlError) < 0.08);
    %rlEnv.binReward = @(controlError) -1 + 1*(controlError < 0.5) + 1*(controlError > -0.5);%0 + 0.5*(controlError < 0.5) + 0.5*(controlError > -0.5);
    rlEnv.binReward = @(controlError) -1 + 1*(controlError < 0.2) + 1*(controlError > -0.2);%0 + 0.5*(controlError < 0.5) + 0.5*(controlError > -0.5);
    %rlEnv.binReward = @(controlError) -1 + 1*(controlError < 0.01) + 1*(controlError > -0.01);%0 + 0.5*(controlError < 0.5) + 0.5*(controlError > -0.5);

    rlEnv.numberOfActions = actionRes - 1;
    rlEnv.numberOfStates = statesRes - 1;
    rlEnv.Terminal = -2; 

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
function MDPAction = mapToMDP(agentActionIndex,dActions,Reps)
    % take average action between two adjacent discretized actions
   % MDPAction = (dActions(agentActionIndex) + dActions(agentActionIndex+1))/2;
   MDPAction = dActions(agentActionIndex);
end

%  digital low-pass filter (chapter 12 of MARLIN)
function smoothedAction = digitalFilter(sampleTime,tauf,previousSmoothed,currentAction)
    A = exp(-1*sampleTime/tauf);
    % time domain digital filter calculation
    smoothedAction = A*previousSmoothed + (1-A)*currentAction;
    
end

%  data for pre-trained policy should in future be rounded to sensible
%  significant figures (just copied from original variable data at the moment)
function [trainedQ,epsilon] = preTrainedRep(~)
    %% just an example, not to be used without first running training
    trainedQ = [0.00417022	0.001467559	0.003967675	0.002044522
1.312790904	1.309245783	3.333314212	1.310448049
3.333319418	0.873781457	0.872793113	0.874435708
0.003023326	0.003455607	0.006852195	0.006704675
];
    epsilon = 0.0396; 
    
end

% function for calculating IAEs of training episodes 
function IAE = calculateIAE(learning,episodeNr)
    dataMatrix = abs(cell2mat(learning.agentExp{1,episodeNr})); % store control errors during simulation
    IAE = trapz(dataMatrix(:,1));
end

% function for calculating simulation IAE
function IAE = calculateIAEEp(simulation)
    dataMatrix = abs(cell2mat(simulation.agentExp{1,1}));
    IAE = trapz(dataMatrix(:,1));
end