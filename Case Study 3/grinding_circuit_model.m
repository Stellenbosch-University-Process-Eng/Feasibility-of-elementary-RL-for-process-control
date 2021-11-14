%% script that solves the Hulbert model equations with MFB feed as pulse inputs
%% Date: 2021-03-16
%% Name: grinding_circuit_model.m
%% reference:
%  Le Roux, J. D. et al. (2013) 
% ‘Analysis and validation of a run-of-mine ore grinding mill circuit model for process control’, 
%  Minerals Engineering, 43–44, pp. 121–134. doi: 10.1016/j.mineng.2012.10.009.
clc; clear

%% specify whether phif parameter should be updated during validation phases
changePhif = 0;%1;

%% sump control parameters
par.DB = 7.85; % ball density (t/m^3)
par.DS = 3.2; % feed ore density (t/m^3)

deltaXs = 6;    % maximum sump volume change (m^3)
deltaTot = 350; % maximum inlet flow change to sump (m^3/h)
par.damping = 8;% "estimated" damping coefficient for sump(-)

par.Kc = -0.736*(deltaTot)/deltaXs; % controller gain (1/h)
par.tau = (4*par.damping^2)/(-1*par.Kc); % integral time (h)
% par.Kc = -200;
% % par.tau = 0.001;  % 1;
% par.tau = 1;

%% values used in CCA paper (not earlier in meeting presentation)
par.Kc = -50;
par.tau = 6;
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
% par.DB = 7.85; % ball density (t/m^3)
% par.DS = 3.2; % feed ore density (t/m^3)
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

%% define all external flowrates to the model
stepTime_MIW = 90;  % time for step change in water to the circuit (h)
stepTime_MFS = 110; % time for step change in solids to the circuit (h)
stepTime_MFB = 2;   % time for step change in solids to teh circuit (h)

MIW_step = 0;       % magnitude of step change in water to the circuit (m^3/h)
MFS_step = 0;       % magnitude of step change in feed solids to the circuit (t/h)
MFB_step = 0;       % magnitude of step change in balls to the circuit (t/h)

%% step changes
par.MIW = @(t) 4.64 + (t > 10)*(3.66 - 4.64) + (t > 30)*(4.22 - 3.66) + (t > 50)*(4.71 - 4.22) + (t > 70)*(4.45 - 4.71); % step in water to circuit (m^3/h)
par.MFS = @(t) 65.2 + (t > 10)*(46.7 - 65.2) + (t > 30)*(57.2 - 46.7) + (t > 50)*(66.9 - 57.2) + (t > 70)*(61.7 - 66.9); % step in ore to circuit (t/h)
%% old re MFB
par.MFB = @(t) 5.69 + (t > 10)*(6.77 - 5.69) + (t > 30)*(6.58 - 6.77) + (t > 50)*(6.43 - 6.58) + (t > 70)*(5.38 - 6.43); % step in feed balls to circuit (t/h)
%% new re MFB
maxMFB = 5.69;%mean([5.69,6.77,6.58,6.43,5.38]);
%ballFeedSteps = [0,20,25,45,50,70,75,95,100];
%par.MFB = @(t) (-maxMFB + maxMFB*(t < 10) + maxMFB*(t > 0)) + (-maxMFB + maxMFB*(t < 65) + maxMFB*(t > 35)) + (-maxMFB + maxMFB*(t < 100) + maxMFB*(t > 70));
par.SFW = @(t) 140.5 + (t > 10)*(69.3 - 140.5) + (t > 30)*(70.4 - 69.3) + (t > 50)*(67.1 - 70.4) + (t > 70)*(67.1 - 67.1); % sump feed water (m^3/h)
%% updated pulse input
% par.MFB = @(t) (-maxMFB + maxMFB*(t < 30) + maxMFB*(t > 0)) + ...
%         (-maxMFB + maxMFB*(t < 65) + maxMFB*(t > 35)) + ...
%         (-maxMFB + maxMFB*(t < 100) + maxMFB*(t > 70));
    
%% pulse input after meeting 25
% maxMFB = 20.484;%21.6;%100;%50;
% par.MFB = @(t) (-maxMFB + maxMFB*(t < 5) + maxMFB*(t > 0)) + ...
%           (-maxMFB + maxMFB*(t < 25) + maxMFB*(t > 20)) + ...
%           (-maxMFB + maxMFB*(t < 45) + maxMFB*(t > 40)) + ...
%           (-maxMFB + maxMFB*(t < 65) + maxMFB*(t > 60)) + ...
%           (-maxMFB + maxMFB*(t < 85) + maxMFB*(t > 80));
%     
%% ramps
% par.MIW = @(t) 4.64 + ( ( (3.66 - 4.64)/10 )*t - ...
%     ( (3.66 - 4.64)/10 )*(10) ).*(t > 10 & t < 20) + ...
%     (( (3.66 - 4.64)/10 )*(20) - ( (3.66 - 4.64)/10 )*10).*(t > 20); 
%     
% par.MFS = @(t) 65.2 + ( ( (46.7 - 65.2)/10 )*t - ...
%     ( (46.7 - 65.2)/10 )*(10) ).*(t > 10 & t < 20) + ...
%     (( (46.7 - 65.2)/10 )*(20) - ( (46.7 - 65.2)/10 )*10).*(t > 20); 
% 
% par.MFB = @(t) 5.69 + ( ( (6.77 - 5.69)/10 )*t - ...
%     ( (6.77 - 5.69)/10 )*(10) ).*(t > 10 & t < 20) + ...
%     (( (6.77 - 5.69)/10 )*(20) - ( (6.77 - 5.69)/10 )*10).*(t > 20); 
% 
% par.SFW = @(t) 140.5 + ( ( (69.3 - 140.5)/10 )*t - ...
%     ( (69.3 - 140.5)/10 )*(10) ).*(t > 10 & t < 20) + ...
%     (( (69.3 - 140.5)/10 )*(20) - ( (69.3 - 140.5)/10 )*10).*(t > 20); 
% 

if changePhif == 1
   %% step 
   par.phif = @(t) par.phif + (t > 10)*(37.6 - 29.6) + (t > 30)*(35.7 - 37.6) + (t > 50)*(31.5 - 35.7) + (t > 70)*(36.9 - 31.5);
   %% ramp 
%    par.phif = @(t) par.phif + ( ( (37.6 - par.phif)/10 )*t - ...
%     ( (37.6 - par.phif)/10 )*(10) ).*(t > 10 & t < 20) + ...
%     (( (37.6 - par.phif)/10 )*(20) - ( (37.6 - par.phif)/10 )*10).*(t > 20);
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

%% solve Hulbert model equations
% Xmw, Xms, Xmf, Xmr, Xmb, Xsw, Xss, Xsf, IE(t)
% sump delays larger to accommodate numerical behaviour arising when changing SFW
par.modLags = [30/3600,30/3600,30/3600,30/3600,30/3600,0.1,0.1,0.1,0.1]; 
stopTime = 100; % stop time for simulation (h)
model_with_delays = dde23( @circuit_model, par.modLags, @history, [0 stopTime], [], par );
time = linspace(0,stopTime,stopTime);
model_high_res = deval( model_with_delays, time );

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


%% plot results
% % Xmw
% subplot(3,3,1)
% plot( time, model_high_res(1,:), 'LineWidth', 2); xlabel('time (h)')
% ylabel('Xmw (m^3)');
% % Xms
% subplot(3,3,2)
% plot( time, model_high_res(2,:), 'LineWidth', 2); xlabel('time (h)')
% ylabel('Xms (m^3)');
% % Xmf
% subplot(3,3,3)
% plot( time, model_high_res(3,:), 'LineWidth', 2); xlabel('time (h)')
% ylabel('Xmf (m^3)');
% % Xmr
% subplot(3,3,4)
% plot( time, model_high_res(4,:), 'LineWidth', 2); xlabel('time (h)')
% ylabel('Xmr (m^3)');
% % Xmb
% subplot(3,3,5)
% plot( time, model_high_res(5,:), 'LineWidth', 2); xlabel('time (h)')
% ylabel('Xmb (m^3)');
% % Xsw
% subplot(3,3,6)
% plot( time, model_high_res(6,:), 'LineWidth', 2); xlabel('time (h)')
% ylabel('Xsw (m^3)');
% % Xss
% subplot(3,3,7)
% plot( time, model_high_res(7,:), 'LineWidth', 2); xlabel('time (h)')
% ylabel('Xss (m^3)');
% % Xsf
% subplot(3,3,8)
% plot( time, model_high_res(8,:), 'LineWidth', 2); xlabel('time (h)')
% ylabel('Xsf (m^3)');
% % integral error
% subplot(3,3,9)
% plot( time, model_high_res(9,:), 'LineWidth', 2); xlabel('time (h)')
% ylabel('IE(t) (m^3)');
% 
% 
% %% plot additional variables
% figure
% % rheology factor
% subplot(2,8,1)
% plot(1:1:length(time), phi, 'k', 'LineWidth', 2); xlabel('time (h)');
% ylabel('rheology factor (-)')
% % LOAD
% subplot(2,8,2)
% plot(1:1:length(time), LOAD, 'k', 'LineWidth', 2); xlabel('time (h)');
% ylabel('LOAD (m^3)')
% % Pmill
% subplot(2,8,3)
% plot(1:1:length(time), Pmill, 'k', 'LineWidth', 2); xlabel('time (h)');
% ylabel('Pmill (kW)')
% % rock consumption
% subplot(2,8,4)
% plot(1:1:length(time), RC, 'k', 'LineWidth', 2); xlabel('time (h)');
% ylabel('RC (m^3)')
% % fines production
% subplot(2,8,5)
% plot(1:1:length(time), FP, 'k', 'LineWidth', 2); xlabel('time (h)');
% ylabel('FP (m^3)')
% % ball consumption
% subplot(2,8,6)
% plot(1:1:length(time), BC, 'k', 'LineWidth', 2); xlabel('time (h)');
% ylabel('BC (m^3)')
% % water outlet flowrate
% subplot(2,8,7)
% plot(1:1:length(time), Vmwo, 'k', 'LineWidth', 2); xlabel('time (h)');
% ylabel('Vmwo (m^3/h)')
% % solids outlet flowrate
% subplot(2,8,8)
% plot(1:1:length(time), Vmso, 'k', 'LineWidth', 2); xlabel('time (h)');
% ylabel('Vmso (m^3/h)')
% % fines outlet flowrate
% subplot(2,8,9)
% plot(1:1:length(time), Vmfo, 'k', 'LineWidth', 2); xlabel('time (h)');
% ylabel('Vmfo (m^3/h)')
% 
% % error
% subplot(2,8,10)
% plot(1:1:length(time), err, 'k', 'LineWidth', 2); xlabel('time (h)');
% ylabel('control error (m^3)');
% % integral error
% subplot(2,8,11)
% plot(1:1:length(time), IE, 'k', 'LineWidth', 2); xlabel('time (h)');
% ylabel('IE (h*m^3)');
% % CFF
% subplot(2,8,12)
% plot(1:1:length(time), CFF, 'k', 'LineWidth', 2); xlabel('time (h)');
% ylabel('CFF (m^3/h)');
% % SVOL
% subplot(2,8,13)
% plot(1:1:length(time), SVOL, 'k', 'LineWidth', 2); xlabel('time (h)');
% ylabel('SVOL (m^3)');
% % Vswo
% subplot(2,8,14)
% plot(1:1:length(time), Vswo, 'k', 'LineWidth', 2); xlabel('time (h)');
% ylabel('Vswo (m^3/h)');
% % Vsso
% subplot(2,8,15)
% plot(1:1:length(time), Vsso, 'k', 'LineWidth', 2); xlabel('time (h)');
% ylabel('Vsso (m^3/h)');
% % Vsfo
% subplot(2,8,16)
% plot(1:1:length(time), Vsfo, 'k', 'LineWidth', 2); xlabel('time (h)');
% ylabel('Vsfo (m^3/h)');
% 
% figure
% % PSE (%)
% subplot(2,4,1)
% plot(1:1:length(time), 100.*PSE, 'k', 'LineWidth', 2); xlabel('time (h)')
% ylabel('PSE (%)')
% % volume % of mill filled
% subplot(2,4,2)
% plot(1:1:length(time), 100.*fraction_filled, 'k', 'LineWidth', 2); xlabel('time (h)');
% ylabel('% of mill filled')
% % MIW 
% subplot(2,4,3)
% plot(time, par.MIW(time), 'k', 'LineWidth', 2); xlabel('time (h)')
% ylabel('MIW (m^3/h)');
% % MFS
% subplot(2,4,4)
% plot(time, par.MFS(time), 'k', 'LineWidth', 2); xlabel('time (h)')
% ylabel('MFS (t/h)');
% % SFW
% subplot(2,4,5)
% plot(time, par.SFW(time), 'k', 'LineWidth', 2); xlabel('time (h)')
% ylabel('SFW (m^3/h)');
% % CFD
% subplot(2,4,6)
% plot(1:1:length(time), CFD, 'k', 'LineWidth', 2); xlabel('time (h)')
% ylabel('CFD (t/m^3)')
% % MFB
% subplot(2,4,7)
% plot(time, par.MFB(time), 'k', 'LineWidth', 2); xlabel('time (h)')
% ylabel('MFB (t/h)')
% % phif
% subplot(2,4,8)
% plot(time, par.phif(time), 'k', 'LineWidth', 2); xlabel('time (h)')
% ylabel('phif (kWh/t)')

%% new
figure
subplot(2,3,1)
% SVOL
plot(1:1:length(time), SVOL, 'k', 'LineWidth', 2); xlabel('time (h)');
ylabel('SVOL (m^3)'); grid on
subplot(2,3,2)
% volume % of mill filled
plot(1:1:length(time), 100.*fraction_filled, 'k', 'LineWidth', 2); xlabel('time (h)');
ylabel('% of mill filled'); grid on
subplot(2,3,3)
% Pmill
plot(1:1:length(time), Pmill, 'k', 'LineWidth', 2); xlabel('time (h)');
ylabel('Pmill (kW)'); grid on
subplot(2,3,4)
% CFF
plot(1:1:length(time), CFF, 'k', 'LineWidth', 2); xlabel('time (h)');
ylabel('CFF (m^3/h)'); grid on
subplot(2,3,5)
% PSE (%)
plot(1:1:length(time), 100.*PSE, 'k', 'LineWidth', 2); xlabel('time (h)')
ylabel('PSE (%)'); grid on
subplot(2,3,6)
% MFB (t/h)
MFB_timePlot = linspace(0,length(time),1000);
plot(MFB_timePlot, par.MFB(MFB_timePlot), 'k', 'LineWidth', 2); xlabel('time (h)')
ylabel('MFB (t/h)'); grid on

% figure
% plot(model_high_res(5,:),'k'); 
% xlabel('time (h)'); ylabel('Xmb (m^3)'); 

% figure
% subplot(2,1,1)
% % PSE (%)
% plot(1:1:length(time), 100.*PSE, 'k', 'LineWidth', 2); xlabel('time (h)')
% ylabel('PSE (%)'); grid on; hold on
% subplot(2,1,2)
% % MFB (t/h)
% MFB_timePlot = linspace(0,length(time),1000);
% plot(MFB_timePlot, par.MFB(MFB_timePlot), 'k', 'LineWidth', 2); xlabel('time (h)')
% ylabel('MFB (t/h)'); grid on; hold on

%% functions
% function to model mill dynamics
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
    
    %% initialize lags
    yl1 = y_lagged(:,1);
    yl2 = y_lagged(:,2);
    yl3 = y_lagged(:,3);
    yl6 = y_lagged(:,6);
    yl7 = y_lagged(:,7);
    yl8 = y_lagged(:,8);
    
    %% feeder calculations
    Vfwo = par.MIW(t);
    Vfso = (par.MFS(t)/par.DS)*(1 - par.alphar);
    Vffo = (par.MFS(t)/par.DS)*par.alphaf;
    Vfro = (par.MFS(t)/par.DS)*par.alphar;
    Vfbo = (par.MFB(t)/par.DB);
    
    %% mill model equations
    phi = ( max(min(1 - ( (1/par.epsilonSV) - 1)*( y(2)/y(1) ), 1), 0) )^0.5; % rheology factor
    Vmwo = par.Vv*phi*y(1)*( y(1)/( y(2) + y(1) ) );    % outlet flowrate of water
    Vmso = par.Vv*phi*y(1)*( y(2)/( y(2) + y(1) ) );    % outlet flowrate of solids
    Vmfo = par.Vv*phi*y(1)*( y(3)/( y(2) + y(1) ) );    % outlet flowrate of fines
    
    LOAD = y(1) + y(4) + y(2) + y(5);          % volume of material in mill
    
    % calculations for mill power draw
    %% old
    %Zr = (LOAD/(par.vmill*par.vPmax)) - 1;
    %Zx = (phi/par.rheaPmax) - 1;
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
    %(par.CFF)*rand(1,1)
    
    % outlet flowrates
    Vswo = CFF*( y(6)/SVOL );
    Vsso = CFF*( y(7)/SVOL );
    Vsfo = CFF*( y(8)/SVOL );
    
    % differential equations
    dXswdt = Vmwo - Vswo + par.SFW(t);
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
    
    % adjust sump outlet flowrates to incorporate time delays 
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
    
    display(t) % display time of solver to track solution progress
    
    %% vector of derivatives
    dydt = [dXmwdt,dXmsdt,dXmfdt,dXmrdt,dXmbdt,dXswdt,dXssdt,dXsfdt,dIntERRdt]';
    
end

% history function for dde solver
function y = history(~,par)
    y = [par.XmwSS,par.XmsSS,par.XmfSS,par.XmrSS,par.XmbSS,par.XswSS,...
        par.XssSS,par.XsfSS,par.IntErrSS];
end