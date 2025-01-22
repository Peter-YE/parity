%% Parameters
% Define constants
epsilon_0 = 8.854187817e-12;  % Vacuum permittivity (F/m)
len = 2e-6;                   % Rename 'length' to 'len' to avoid conflict
width = 1.33e-7;
height = 7.3e-8;
A = len * width;              % Area of the plates (m^2)
Ac = len * height;        
g0 = 1.5e-7;                  % Initial gap (m)
d0 = 5e-8;                    % Distance for coupling (m)
VDC = 2;                      % DC voltage (V)
VAC1 = 0.0001;                % AC voltage 1 (V)
VAC2 = 0.0005;                % AC voltage 2 (V)
m = 1;                        % Effective mass
f0 = 117.0818e6;
omega0 = (2 * pi) * f0;        % Natural frequency in rad/s
f1 = 117.0818e6;
omega = (2 * pi)*f1;
Q = 330;                % Quality factor

% Time settings
t_min = 0;                    % Start time (s)
t_max = 5e-5;                % End time (s)
dt = 1e-12;                    % Time step (s)
time = t_min:dt:t_max;        % Time array

% Initial conditions
z1_0 = 0;                     % Initial displacement of resonator 1
v1_0 = 0;                     % Initial velocity of resonator 1
z2_0 = 0;                     % Initial displacement of resonator 2
v2_0 = 0;                     % Initial velocity of resonator 2

% Preallocate results
z1 = zeros(1, size(time,2));
z2 = zeros(1, size(time,2));
v1 = zeros(1, size(time,2));
v2 = zeros(1, size(time,2));

%initial values
z1(1) = z1_0;
z2(1) = z2_0;
v1(1) = v1_0;
v2(1) = v2_0;
kc=-((epsilon_0*A)/d0^3)*(VAC2-VAC1)^2;

%% Step input



steps = 50;
tau = (t_max - t_min)/steps;
step_val = randi([0 1], 1, steps+1);
step_val = [ones(1,steps/2), zeros(1,steps/2+1)];
%%step_val = 2*randi([0 1], 1, steps+1) - 1;
step_time = linspace(t_min, t_max, steps+1);
step_function = @(t) interp1(step_time, step_val, t, 'previous', 'extrap');


%% Elec force

F_elec1 = @(t,y) step_function(t)* (epsilon_0*A*(VDC+VAC1)^2/(g0-y(1))^2)*sin(omega0 * t);  % External force on resonator 1
F_elec2 = @(t,y) 0;  % External force on resonator 2


dydt = @(t, y) [
        y(2); 
        F_elec1(t,y)/m - (omega0 * y(2) / Q) - (omega0^2)/m * y(1) ...
        - (kc / m) * (y(1) - y(3)) ; % Resonator 1
        y(4);
        F_elec2(t,y)/m - (omega0 * y(4) / Q) - (omega0^2)/m * y(3) ...
        - (kc / m) * (y(3) - y(1))  % Resonator 2
    ];


%% Runge-Kutta
for i = 1:size(time,2)-1
    t = time(i);

    % Current state vector [z1; v1; z2; v2]
    y = [z1(i); v1(i); z2(i); v2(i)];

    % Runge-Kutta steps
    k1 = dydt(t, y);
    k2 = dydt(t + dt / 2., y + dt * k1 / 2.);
    k3 = dydt(t + dt / 2., y + dt * k2 / 2.);
    k4 = dydt(t + dt, y + dt * k3);

    % Update solution
    y_next = y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);

    % Store results
    z1(i+1) = y_next(1);
    v1(i+1) = y_next(2);
    z2(i+1) = y_next(3);
    v2(i+1) = y_next(4);
end

%% Plot the results
figure;
subplot(2,1,1);
plot(time, z1, 'b-', 'LineWidth', 2);
hold on;
plot(time, z2, 'r-', 'LineWidth', 1);
%hold on;
%stairs(step_time, step_val, 'LineWidth', 1.5);
hold off;

% Add labels and legend
xlabel('Time (s)', 'FontSize', 12);
ylabel('Displacement (m)', 'FontSize', 12);
title('Time-Domain Response of Coupled NEMS Resonators', 'FontSize', 14);
legend('z_1(t)', 'z_2(t)', 'Location', 'best');
grid on;


subplot(2,1,2);
stairs(step_time, step_val, 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Step Value');
title('Step Function');
grid on;