%% short description
% Script that trains a tabular Q-learning agent with PI control data
% generated on the revised watertank model.
% Set modelValve = 1 to model valve stiction 
%% NB:  Expressed all actions and PI tuning parameters as lumped parameter k*(60)
% Date: 2020/09/15
%%
%% clear command window and workspace
clc
clear
%% 
% seed random number generator to ensure exploration-exploitation strategy 
% is repeatable
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
training.nmberOfEps = 3000;%5000;     % number of episodes to use in training
training.nmberOfSteps = 1200;   % number of steps allowed per episode 
setPoint = 1.5;%% Changed 2021-07-20 %1.4; % set point height in tank (m)
simSetPoint = setPoint; % setPoint used if only simulating
training.firstHeightVec = [1.5]'; %% Changed 2021-07-20 %[1.4]'; % vector of starting heights for episodes
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

%% parameter and flag for modelling valve stiction 
valve.d = 0.0008;   % valve-stiction band parameter (sm^2/min - unit of MV(t))
modelValve = 0;     % flag to indicate whether valve dynamics should be incorporated

%% discretize possible states and actions (can lead to error if not sufficient to cover all encountered states)
% state bounds (note that only single state component is applicable in this application) 
MDPstateVecLow = -2;           % vector for lower bounds of all state components
MDPstateVecHigh = 1;           % vector for upper bounds of all state components
statesResAvailable = 110;%14;%110;%15;       % number of discrete states = (statesRes - 1)
% action bounds as vectors or scalars
%% NB -> Actions are defined later.  The following few lines help with 
%% consistency in functions used.  Make sure "actionResAvailable" corresponds
%% to the chosen definition of dActions.
MDPactionVecLow = 0;          % vector for lower bounds of all action components
MDPactionVecHigh = 0.15;      % vector for upper bounds of all action components
actionResAvailable = 11;%21;      % actions available = (actionResAvailable - 1)

%% create training environment
A = 1.8; % tank area (m^2)
L = 0.5; % pressure loss parameter (dimensionless)
myEnvironment = createMDP(rewardBandwidth,A,L,actionResAvailable,statesResAvailable,setPoint); % create RL environment SPECS
disturbanceValueVec = 50:10:500;%50:10:350;%50:10:1000;    %% changed 2021-07-20 % inlet flowrates for PI controller simulation
%% Agent does not have to be "frozen", the following line is just for illustrating behaviour for a shorter episode.
simDisturbanceValue = 200;  % inlet flowrate used for simulating agent (Lpm).  

%% specify or agent should be trained
% doTraining = 1; % specify as 1 

%% specify continuous or binary reward structure
% binaryReward = 1; % if this is set to one, a pulse (minR = 0; maxR = 1) with width +- 0.5m around SP is used as reward function.
%% END OF USER INPUT

%% create action-value table (for one flowrate)
Reps.action_value = createRep(myEnvironment);
action_value_initial = Reps.action_value; % store initial action-value table

%% initialize variable for recording rewards within averaging window
learning.rewardsInWindow = 0;
%% discretize states and actions and record extreme bounds of CV that may be instantiated
% create discrete set of actions for agent to choose from
[dStates,~] = discretize(MDPstateVecLow,MDPstateVecHigh,...
                   MDPactionVecLow,MDPactionVecHigh,statesResAvailable,actionResAvailable);
% define discrete actions available to the RL agent
%dActions = linspace(0.001,0.4,20)';

%% revisiting Q-learning 2021-07
    %% Case 1
% dActions = linspace(0.001,0.2,10)'; %% defined on 2021-07-18
% dStates = [-150,linspace(-2.14,2.14,11),150]'; %% defined on 2021-07-18
    %% Case 2
dActions = linspace(0.001,0.2,10)'; %% defined on 2021-07-18
dStates = [150,linspace(-2.14,2.14,101),150]'; %% defined on 2021-07-18

% record smallest and largest steady-state CV changes that may be affected
% between two time steps in the MDP.
process.smallestSSEffect = (dActions(2)-dActions(1))*process.Kp;
process.largestSSEffect = (dActions(end)-dActions(1))*process.Kp;

%% PI controller parameters (before any SP or DV changes)
PI.gain = -0.064;           % PI controller gain 
PI.integralTime = (1/0.1);  % PI integral time
PI.timeDomainBias = 0.07;   % control bias (initial action at SS)
PI.initialSSHeight = 1.5;   % initial SS height in meters
PI.sampleTime = 1;          % sampling time between control calculations (MUST be << process time constant)
%% PI CONTROLLER HERE:
%% simulate PI controller and update Q-learning action-value table
for episodeCntr = 1:1:training.nmberOfEps
        clear State
        clear Action
        currentTimeStamp = 1;
        State(currentTimeStamp) = training.firstStateVec(randi(length(training.firstStateVec)));
        learning.prevHeight = setPoint - State(currentTimeStamp);
        disturbanceValue = randi([disturbanceValueVec(1,1),disturbanceValueVec(1,end)],1);  % sample a DV from disturbanceValueVec 
        for stepCntr = 1:1:training.nmberOfSteps
            % initialize control error
            if stepCntr == 1
                prevError = State(currentTimeStamp);
                prevIntegralError = 0;
                
            elseif stepCntr ~= 1
                prevError = State(currentTimeStamp - 1);
            end
            Error = State(currentTimeStamp);
            % simulate PI control law
            [Action(currentTimeStamp),prevIntegralError] = ...
                                                         PIController(PI,...
                                                         Error,prevError,...
                                                         prevIntegralError);
            %% model valve stiction (Stenman model)
            signal2Valve = Action(currentTimeStamp);
            if modelValve == 1
                
                if currentTimeStamp == 1
                    u_t = Action(currentTimeStamp);
                    x_tminOne = Action(currentTimeStamp);
                else
                    u_t = Action(currentTimeStamp);
                    x_tminOne = Action(currentTimeStamp - 1);
              
                end
            
            if abs(u_t - x_tminOne) <= valve.d
                x_t = x_tminOne;
            else
                x_t = Action(currentTimeStamp);
            end
            
            Action(currentTimeStamp) = x_t;
                
            end
            %% end of stiction modeling
            
            % simulate RL environment to obtain next height in the tank
            learning.nxtHeight = simulateMDP(currentTimeStamp,...
                                  learning.prevHeight,...
                                  Action(currentTimeStamp),...
                                  myEnvironment,setPoint,disturbanceValue);
            learning.nxtState = setPoint - learning.nxtHeight; % calculate next control error
            
            
            learning.controlData{1,episodeCntr}{stepCntr,1} = State(currentTimeStamp); % record control error data
            learning.controlData{1,episodeCntr}{stepCntr,2} = Action(currentTimeStamp); % record MV(t) data
            learning.controlData{1,episodeCntr}{stepCntr,3} = signal2Valve; % record control signal to final element
            
            if currentTimeStamp > 1
                %% update Q-learning agent's table -> agent must "trail behind" 
                %% the PI controller so that it has access to the state at the next time stamp of the MDP
                learning.agentState = mapStatesToAgent(State(currentTimeStamp-1),dStates); % obtain coded agent state at previous time stamp
                learning.nxtAgentState = mapStatesToAgent(State(currentTimeStamp),dStates); % obtain coded agent state at current time stamp
                learning.agentAction = mapActionsToAgent(Action(currentTimeStamp-1),dActions); % obtain coded agent action
                learning.agentReward = myEnvironment.binReward(Error); % obtain reward from environment
                % calculate greedy action at current time stamp
                learning.greedyAction = find(ismember(Reps.action_value(learning.nxtAgentState,:),...  
                                    max(Reps.action_value(learning.nxtAgentState,:))));
                % tie breaking
                if size(learning.greedyAction,2) > 1
                    learning.greedyAction = randi(myEnvironment.numberOfActions);
                end
                % update the agent's action-value table
                Reps.action_value(learning.agentState,learning.agentAction) = ...
                            Reps.action_value(learning.agentState,learning.agentAction) +...
                            par.alpha*(learning.agentReward + par.gamma*learning.greedyAction -...
                            Reps.action_value(learning.agentState,learning.agentAction));
            end

                                   
            % shift time step T <- (T+1)
            currentTimeStamp = currentTimeStamp + 1;
            % shift state S(T) <- S(T+1)
            State(currentTimeStamp) = learning.nxtState;
            learning.prevHeight = learning.nxtHeight;
             
         end
      fprintf('%d\n',episodeCntr); % display episode counter to notify user of training progress

end % end of episode loop

%% functions

% function to create action-value table
function Qtable = createRep(myEnvironment)
    Qtable = zeros(myEnvironment.numberOfStates,myEnvironment.numberOfActions);

end

% function to map states to coded state numbers for agent
function agentState = mapStatesToAgent(state,dStates)

    for cntr = 1:1:(size(dStates,1)-1)
        if state > dStates(cntr,1) && state <= dStates(cntr+1,1)%< dStates(cntr+1,1)
            agentState = cntr;
        else
        end
    end
end

% function to map PI control actions to coded action numbers for agent
function agentAction = mapActionsToAgent(action,dActions)
    for cntr = 1:1:(size(dActions,1)-1)
        if action > dActions(cntr,1) && action <= dActions(cntr+1,1)
            agentAction = cntr;
        else
        end
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
    rlEnv.binReward = @(controlError) 0 + 0.5*(controlError < 0.2) + 0.5*(controlError > -0.2);
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
function MDPAction = mapToMDP(agentActionIndex,dActions,~)
   MDPAction = dActions(agentActionIndex);
   
end

% function to calculate PI controller's MV(t) selection.
% Make sure sampling time interval for control is suitably small (Shannon's
% theorem). For the watertank, 1 min sampling is sufficient, seeing as this
% is much smaller than the process time constant (verified with modeling in Simulink).
% Continuous form of control equation used.
function [MV_t,currentIntegral] = PIController(PI,Error,prevError,prevIntegralError)

    integralChange = prevError*(PI.sampleTime) + 0.5*(Error - prevError)*(PI.sampleTime);
    currentIntegral = integralChange + prevIntegralError;
    MV_t = PI.gain*(Error + (1/PI.integralTime)*currentIntegral) + PI.timeDomainBias;
    
end
