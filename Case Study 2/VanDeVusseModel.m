%% short description
% script that may be used to simulate the Van de Vusse reaction scheme
% model
% Chen, H., Kremling, A. and Allgöwer, F. (1995) 
%‘Nonlinear Predictive Control of a Benchmark CSTR’, 
% in Proceedings of the 3rd European Control Conference ECC’95. Rome, Italy, pp. 3247–3252.
%%
clc
clear

%% model parameters
p.Vr = 0.01*1000;  %reactor volume (m^3)
p.Ar = 0.215;      %surface area of cooling jacket (m^2)
p.mc = 5;          %mass of coolant in jacket (kg)
p.Cpc = 2;         %coolant heat capacity (kJ/(kgK))
p.kw = 4032;       %cooling jacket heat transfer coefficient (kJ/(hm^2K))
p.Cp = 3.01;       %reactor content heat capacity (kJ/(kgK))
p.rho = 0.9342;    %reactor content density (kg/L)
p.dHrab = 4.2;     %reaction enthalpy change for reaction 1 (kJ/molA)
p.dHrbc = -11;     %reaction enthalpy change for reaction 2 (kJ/mol B)
p.dHrad = -41.85;  %reaction enthalpy change for reaction 3 (kJ/mol A)
p.E1 = -9758.3;    %activation energy reaction 1 (converted to K)
p.E2 = -9758.3;    %activation energy reaction 2 (converted to K)
p.E3 = -8560;      %activation energy reaction 3 (converted to K)
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
p.CB0 = 0;

%% inlet temperature disturbance and time simulated
inletTDisturbance = 9;%5; % disturbance in inlet temperature (degC)
QcStep = 0; % step size for heat removal rate
VdivVrStep = 10;%-1*(14.19-20);%15.8-14.19;%8.34;%30-p.VdivVr;%20-14.19; % step size for VdivVr
stepTime = 0.1; % time of step change (h)
p.T0 = @(t) p.T0s + inletTDisturbance*(t > stepTime);
p.Qcdyn = @(t) p.Qc + QcStep*(t>stepTime);
p.VdivVrdyn = @(t) p.VdivVr + VdivVrStep*(t>stepTime);
timeDuration = 0.7; % time duration simulated (h)

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
ss = fsolve(ssfun, initialguess, opt);


%% solve dynamic model
%solve VdV model and store results in T and Y matrices
%ss = [p.Cas, p.Cbs, p.Ts, p.Tcs];

%Arrhenius equation set  (T [=] degC)
Arr.k1 = @(T) p.k10*exp(p.E1/(T+273.15)); 
Arr.k2 = @(T) p.k20*exp(p.E2/(T+273.15));
Arr.k3 = @(T) p.k30*exp(p.E3/(T+273.15));

%% simulate Van der Vusse model
time = linspace(0, timeDuration, 500);
[T, Output] = ode23s(@ (t, x) VdVmodel(t, x, p, Arr), time, ss);%, options);

%% display results
subplot(2, 2, 1)
plot(T, Output(:, 1), 'LineWidth', 2)
hold on; xlabel('time (h)'); ylabel('CA (mol/L)')

subplot(2, 2, 2)
plot(T, Output(:, 2), 'LineWidth', 2)
hold on; xlabel('time (h)'); ylabel('CB (mol/L)')

subplot(2, 2, 3)
plot(T, Output(:, 3), 'LineWidth', 2)
hold on; xlabel('time(h)'); ylabel('Temperature in reactor (degrees C)')

subplot(2, 2, 4)
plot(T, Output(:, 4), 'LineWidth', 2)
hold on; xlabel('time(h)'); ylabel('Temperature of cooling stream (degrees C)')


%% functions
% function to capture the dynamics of the Van der Vusse benchmark
function dVdVdt = VdVmodel(t,x, p, Arr)
%VdV model in column matrix form, where x(1) -> Ca, x(2) -> Cb, x(3) -> T,
%x(4) -> Tcw

dVdVdt =         [p.VdivVrdyn(t)*(p.CA0 - x(1)) - Arr.k1(x(3))*x(1) - Arr.k3(x(3))*x(1)^2;...
    
                  -1*p.VdivVrdyn(t)*(x(2)) + Arr.k1(x(3))*x(1) - Arr.k2(x(3))*x(2);...
                  
                  p.VdivVrdyn(t)*(p.T0(t) - x(3)) - (1/(p.rho*p.Cp))*(Arr.k1(x(3))*x(1)*p.dHrab...
                  + Arr.k2(x(3))*x(2)*p.dHrbc + Arr.k3(x(3))*x(1)^2*p.dHrad) ...
                  + ((p.kw*p.Ar)/(p.rho*p.Cp*p.Vr))*(x(4) - x(3));...
                  
                  (1/(p.mc*p.Cpc))*(p.Qcdyn(t) + p.kw*p.Ar*(x(3) - x(4)));
                  
                  ];
end
