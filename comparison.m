clearvars;
clear all;
clc;
% dynamic parameters
A = -1;
B = 0;
C = 0;
D = 1;
rho = 1;%0;
M = 2;
N = 2;
R = 1;
P = 1;
Q = 2;
reward_func = @(x,a) -(0.5*M * x.^2 + R * x.*a + 0.5*N*a.^2 + P*x + Q * a);
lam = 0.1;

% solve theoretical solution
% J(x) = a_2 * x^2/2 + a_1 * x, value = a_0
myfunc = @(t) (t * (B + C * D) - R).^2 + ( (2 * A + C^2) * t - M - rho * t) .* (N - t * D^2);
a_2 = fzero(myfunc,[-100,0]);
a_1 = (P * (N - a_2 * D^2) - Q * R + Q * a_2* (B+C*D))/(a_2 * B * (B + C*D) + (A - rho) * (N - a_2 * D^2) - B * R);
a_0 = (a_1 * B - Q)^2/(2 * (N - a_2 * D^2)) + lam * log(2*pi*lam/(N - a_2*D^2))/2;
optimal_r = (a_1 * B - Q)^2/(2 * (N - a_2 * D^2));

theta_true = [a_2;a_1;a_0];
phi_true = [(a_2 * (B + C*D)-R)/(N - a_2*D^2); (a_1 * B - Q)/(N - a_2*D^2); log(lam/(N - a_2*D^2))];
psi_true = [(a_2 * (B + C*D)-R)/(N - a_2*D^2); (a_1 * B - Q)/(N - a_2*D^2); log(1/(N - a_2*D^2))];

% simulate dwt
rng('default')
%rng(1)
T = 10^6; 
dt = 0.1;
nt = T/dt;
x0 = 0;
dw = sqrt(dt) * randn(nt,1);

M = 10^5;
% create behavioral data
b_m = [1,1]; %[0.5,0.5]
b_v = 1;

x_path_b = x0 * ones(nt+1,1);
a_path_b = zeros(nt+1,1);
r_path_b = zeros(nt,1);
x_now = x0;
for t = 1:nt
    a_now = b_m(1) * x_now + b_m(2) + sqrt(b_v) * randn(1);
    x_new = x_now + (A * x_now + B * a_now) * dt + (C * x_now + D * a_now) * dw(t);
    
    r_path_b(t) = reward_func(x_now,a_now);
    a_path_b(t) = a_now;
    x_path_b(t+1) = x_new;
    x_now = x_new;
end
a_now = b_m(1) * x_now + b_m(2) + sqrt(b_v) * randn(1);
a_path_b(nt+1) = a_now;


tic

x_path_b = x_path_b(1:10:(nt+1));
a_path_b = a_path_b(1:10:(nt+1));
r_path_b = r_path_b(1:10:nt);

dt = 1;
nt = T/dt;


% CSAC-learn off-policy
phi_init = zeros(3,1);
phi_init(3) = 0;
% generate initial action
phi = phi_init; 
theta = zeros(3,1);

alpha_theta = 1;
alpha_phi = 0.1;
batch_m = 1000;
iter = 1;

theta_path = zeros(3, M+1);
phi_path = zeros(3, M+1);
while iter <= M
    index = randsample(nt,batch_m,'true');
    a_path = a_path_b(index);
    r_path = r_path_b(index);
    x_path = x_path_b(index);
    x_path_next = x_path_b(index+1);
    pdf_b=normpdf(a_path, b_m(1) * x_path + b_m(2),sqrt(b_v));

    [grad_theta,grad_phi] = test_learn_para(pdf_b,phi,theta, x_path, x_path_next,r_path,a_path, dt,lam);
    grad_theta = max(min(grad_theta, 10),-10);
    grad_phi = max(min(grad_phi, 5),-5);
    theta = theta + alpha_theta * grad_theta/iter^0.67;
    phi = phi + alpha_phi * grad_phi/iter^0.67;

    theta_path(:,iter+1) = theta;
    phi_path(:,iter+1) = phi;
    iter = iter + 1;
    if mod(iter, 1e4) == 0
        fprintf('iteration %d: phi_1=%.3f, phi_2=%.3f, exp^phi_3=%.3f\n',iter, phi_path(1,iter)-phi_true(1), phi_path(2,iter)-phi_true(2), phi_path(3,iter)-phi_true(3));
    end
end

toc
figure;
plot(0:M, phi_path(1,:),'k','LineWidth',2)
hold on
plot(0:M, phi_true(1) * ones(M+1,1),'k--','LineWidth',2)
hold on
plot(0:M, phi_path(2,:),'r','LineWidth',2)
hold on
plot(0:M, phi_true(2) * ones(M+1,1),'r--','LineWidth',2)
hold on
plot(0:M, exp(phi_path(3,:)),'b','LineWidth',2)
hold on
plot(0:M, exp(phi_true(3)) * ones(M+1,1),'b--','LineWidth',2)
ylim([-10,10])
legend('\phi_1 Path','True Value \phi_1','\phi_2 Path','True Value \phi_2','e^{\phi_3} Path','True Value e^{\phi_3}')
set(gca,'FontSize',12);
xlabel('Iteration','fontsize',14)
title('Continuou soft actor-critic learning');

% q-V-learn off-policy [Jia and zhou,2023]
theta_q_init = zeros(3,1);
theta_q_init(3) = -log(lam);
% generate initial action
theta_q = theta_q_init;
theta_v = zeros(3,1);

alpha_theta = 1;
batch_m = 1000;
iter = 1;

learned_para = zeros(M+1,6);
learned_para(1,:) = [theta_q',theta_v'];
while iter <= M
    index = randsample(nt,batch_m,'true');
    a_path = a_path_b(index);
    r_path = r_path_b(index);
    x_path = x_path_b(index);
    x_path_next = x_path_b(index+1);
    
    grad_theta = qv_learn_para(theta_q,theta_v, x_path, x_path_next,r_path,a_path, dt,lam)/dt;
    
    theta_q = theta_q + alpha_theta * grad_theta(1:3)/iter^0.67;
    theta_v = theta_v + alpha_theta * grad_theta(4:6)/iter^0.67;
    
    theta_v = max(min(theta_v, 10),-10);
    theta_q = max(min(theta_q, 5),-5);

    learned_para(iter+1,:) = [theta_q',theta_v'];
    iter = iter + 1;
    if mod(iter, 1e4) == 0
        fprintf('iteration %d: phi_1=%.3f, phi_2=%.3f, exp^phi_3=%.3f\n',iter, learned_para(iter,1)-phi_true(1), learned_para(iter,2)-phi_true(2), learned_para(iter,3)-phi_true(3));
    end
end

toc
figure;
plot(0:M, learned_para(:,1),'k','LineWidth',2)
hold on
plot(0:M, psi_true(1) * ones(M+1,1),'k--','LineWidth',2)
hold on
plot(0:M, learned_para(:,2),'r','LineWidth',2)
hold on
plot(0:M, psi_true(2) * ones(M+1,1),'r--','LineWidth',2)
hold on
plot(0:M, lam*exp(learned_para(:,3)),'b','LineWidth',2)
hold on
plot(0:M, lam*exp(psi_true(3)) * ones(M+1,1),'b--','LineWidth',2)
ylim([-10,10])
legend('\psi_1 Path','True Value \psi_1','\psi_2 Path','True Value \psi_2','\lambda*e^{\psi_3} Path','True Value \lambda*e^{\psi_3}')
set(gca,'FontSize',12);
xlabel('Iteration','fontsize',14)
title('q learning');
% 
% % TD(0) [Jia and zhou,2022b, Algorithms 1 and 2]
 alpha_theta = 0.01;
 alpha_phi = 0.01;
 theta = zeros(3,1);
 phi = zeros(3,1);
% 
% 
 batch_m = 1000;
 iter = 1;
% 
 theta_path = zeros(3, M+1);
 phi_path = zeros(3, M+1);
 tic
 while iter <= M
     index = randsample(nt,batch_m,'true');
     a_path = a_path_b(index);
     r_path = r_path_b(index);
     x_path = x_path_b(index);
     x_path_next = x_path_b(index+1);
%     
%     
     [grad_theta, grad_phi] = semigradient(theta,phi,x_path,x_path_next,r_path,a_path,dt,lam);
%     
% 
     theta = theta + alpha_theta * grad_theta/(dt * iter^0.67);
     phi = phi + alpha_phi * grad_phi/(dt * iter^0.67);
% 
     theta = max(min(theta, 10),-10);
     phi = max(min(phi, 5),-5);
% 
     theta_path(:,iter+1) = theta;
     phi_path(:,iter+1) = phi;
     iter = iter + 1;
     if mod(iter, 1e4) == 0
        fprintf('iteration %d: phi_1=%.3f, phi_2=%.3f, exp^phi_3=%.3f\n',iter, phi_path(1,iter)-phi_true(1), phi_path(2,iter)-phi_true(2), phi_path(3,iter)-phi_true(3));
     end
 end
 toc
 figure;
 plot(0:M, phi_path(1,:),'k','LineWidth',2)
 hold on
 plot(0:M, phi_true(1) * ones(M+1,1),'k--','LineWidth',2)
 hold on
 plot(0:M, phi_path(2,:),'r','LineWidth',2)
 hold on
 plot(0:M, phi_true(2) * ones(M+1,1),'r--','LineWidth',2)
 hold on
 plot(0:M, exp(phi_path(3,:)),'b','LineWidth',2)
 hold on
 plot(0:M, exp(phi_true(3)) * ones(M+1,1),'b--','LineWidth',2)
 ylim([-10,10])
 legend('\phi_1 Path','True Value \phi_1','\phi_2 Path','True Value \phi_2','e^{\phi_3} Path','True Value e^{\phi_3}')
 set(gca,'FontSize',12);
 xlabel('Iteration','fontsize',14)
title('Continuous TD(0) learning');

% % SARSA(0)
 theta_init = zeros(6,1);
 theta_init(3) = -log(lam*dt);
% generate initial action
 theta = theta_init;
 learned_para_big_q = zeros(M+1,6);
% 
 learned_para_big_q(1,:) = theta';
% 
 alpha_theta = 0.1;
% 
 batch_m = 1000;
 iter = 1;
 tic
 while iter <= M
     index = randsample(nt,batch_m,'true');
     a_path = a_path_b(index);
     r_path = r_path_b(index);
     x_path = x_path_b(index);
     x_path_next = x_path_b(index+1);
     a_path_next = a_path_b(index + 1);
% 
     a_mean = theta(1) * x_path_next + theta(2);
     temp_v = lam*dt*exp(theta(3));
     ent = -log(normpdf(a_path_next, a_mean, sqrt(temp_v)));
% 
     grad_theta = qlearn_para(theta, x_path,x_path_next,r_path,a_path,a_path_next, dt,rho,lam,ent)/dt^2;
%     
     theta = theta + alpha_theta * grad_theta/iter^0.67;
%     
%     
     theta = max(min(theta, 5),-5);
% 
     learned_para_big_q(iter+1,:) = theta';
     iter = iter + 1;
 end
% 
 toc
 figure;
 plot(0:M, learned_para_big_q(:,1),'k','LineWidth',2)
 hold on
 plot(0:M, phi_true(1) * ones(M+1,1),'k--','LineWidth',2)
 hold on
 plot(0:M, learned_para_big_q(:,2),'r','LineWidth',2)
 hold on
 plot(0:M, phi_true(2) * ones(M+1,1),'r--','LineWidth',2)
 hold on
 plot(0:M, dt*lam*exp(learned_para_big_q(:,3)),'b','LineWidth',2)
 hold on
 plot(0:M, exp(phi_true(3)) * ones(M+1,1),'b--','LineWidth',2)
 ylim([-10,10])
 legend('\phi_1 Path','True Value \phi_1','\phi_2 Path','True Value \phi_2','e^{\phi_3} Path','True Value e^{\phi_3}')
 set(gca,'FontSize',12);
 xlabel('Iteration','fontsize',14)
 title('SARSA learning');
%TD(0)
function [grad_theta, grad_phi] = semigradient(theta,phi,xt_path,xt_path_next,rt_path,at_path,dt,lam)
grad_theta = theta - theta;
grad_phi = phi - phi;
v = @(x) theta(1) * x.^2/2 + theta(2) * x;
test_func_1 = @(x) x.^2/2;
test_func_2 = @(x) x;
regularizer = @(x) x-x + (log(2*pi) + 1 + phi(3))/2;

dmt = v(xt_path_next) - v(xt_path) + dt * ( rt_path + lam * regularizer(xt_path) - theta(3));
xi_1 = test_func_1(xt_path);
xi_2 = test_func_2(xt_path);

grad_theta(1:end-1) = mean(dmt .* [xi_1,xi_2], 1)';
grad_theta(end) = mean(dmt);


g_loglike_1 = -exp(-phi(3)) * (phi(1) * xt_path + phi(2) - at_path) .* xt_path;
g_loglike_2 = -exp(-phi(3)) * (phi(1) * xt_path + phi(2) - at_path);
g_loglike_3 = 0.5*exp(-phi(3)) * (phi(1) * xt_path + phi(2) - at_path).^2 -0.5;

q_reward = [0;0;0.5*lam];

grad_phi = mean(dmt .* [g_loglike_1,g_loglike_2,g_loglike_3], 1) + q_reward' * dt;
grad_phi = grad_phi';

end
%SARSA
function [grad_theta] = qlearn_para(theta, xt_path,xt_path_next,rt_path,at_path,at_path_next, dt,rho,lam,ent)
grad_theta = theta - theta;
q = @(a,x) (-0.5 * exp(-theta(3)) * (a - theta(1) * x - theta(2)).^2 + theta(4) * x.^2 + theta(5)*x);

test_func_1 = @(a,x) exp(-theta(3)) * (a - theta(1) * x - theta(2)) .* x;
test_func_2 = @(a,x) exp(-theta(3)) * (a - theta(1) * x - theta(2));
test_func_3 = @(a,x) 0.5 * exp(-theta(3)) * (a - theta(1) * x - theta(2)).^2;
test_func_4 = @(a,x) x.^2;
test_func_5 = @(a,x) x;
test_func_6 = @(a,x) a - a + 1;

dmt = q(at_path_next, xt_path_next) - q(at_path, xt_path) + dt * ( rt_path - rho * q(at_path, xt_path) - theta(6) + lam * ent);
xi_1 = test_func_1(at_path, xt_path);
xi_2 = test_func_2(at_path, xt_path);
xi_3 = test_func_3(at_path, xt_path);
xi_4 = test_func_4(at_path, xt_path);
xi_5 = test_func_5(at_path, xt_path);
xi_6 = test_func_6(at_path, xt_path);


grad_theta = mean(dmt .* [xi_1, xi_2, xi_3, xi_4, xi_5, xi_6]);

grad_theta = grad_theta';
end
%qv-learn
function [grad_theta] = qv_learn_para(theta_q,theta_v, xt_path, xt_path_next,rt_path,at_path, dt,lam)
grad_theta = [theta_q - theta_q; theta_v - theta_v]; 
q = @(a,x) -0.5 * exp(-theta_q(3)) * (a - theta_q(1) * x - theta_q(2)).^2 - 0.5*lam*(log(2*pi*lam) + theta_q(3));
v = @(x) theta_v(1)*x.^2 + theta_v(2)*x;

test_func_1 = @(a,x) exp(-theta_q(3)) * (a - theta_q(1) * x - theta_q(2)) .* x;
test_func_2 = @(a,x) exp(-theta_q(3)) * (a - theta_q(1) * x - theta_q(2));
test_func_3 = @(a,x) 0.5 * exp(-theta_q(3)) * (a - theta_q(1) * x - theta_q(2)).^2  - 0.5*lam;
test_func_4 = @(a,x) x.^2;
test_func_5 = @(a,x) x;
test_func_6 = @(a,x) a - a + 1;

dmt = v(xt_path_next) - v(xt_path) + dt * ( rt_path - q(at_path, xt_path) - theta_v(3));

xi_1 = test_func_1(at_path, xt_path);
xi_2 = test_func_2(at_path, xt_path);
xi_3 = test_func_3(at_path, xt_path);
xi_4 = test_func_4(at_path, xt_path);
xi_5 = test_func_5(at_path, xt_path);
xi_6 = test_func_6(at_path, xt_path);

grad_theta = mean(dmt .* [xi_1, xi_2, xi_3, xi_4, xi_5, xi_6], 1);

grad_theta = grad_theta';
end
%csac
function [grad_theta,grad_phi] = test_learn_para(pdf_b,theta,phi, xt_path, xt_path_next,rt_path,at_path, dt,lam)
grad_theta = theta - theta;
grad_phi = phi - phi;
p= @(a,x) -0.5 * exp(-phi(3)) * (a - phi(1) * x - phi(2)).^2 - 0.5*(log(2*pi) + phi(3));
v = @(x) theta(1)*x.^2./2 + theta(2)*x;
dmt = v(xt_path_next) - v(xt_path) + dt * ( rt_path - lam*p(at_path, xt_path)-theta(3));

test_func_1 = @(a,x) exp(-phi(3)) * (a - phi(1) * x - phi(2)) .* x;
test_func_2 = @(a,x) exp(-phi(3)) * (a - phi(1) * x - phi(2));
test_func_3 = @(a,x) 0.5 * exp(-phi(3)) * (a - phi(1) * x - phi(2)).^2  - 0.5;
test_func_4 = @(a,x) x.^2./2;
test_func_5 = @(a,x) x;
test_func_6 = @(a,x) a - a + 1;
xi_4 = test_func_4(at_path, xt_path);
xi_5 = test_func_5(at_path, xt_path);
xi_6 = test_func_6(at_path, xt_path);
is = normpdf(at_path, phi(1) .* xt_path + phi(2), sqrt(exp(phi(3))))./pdf_b;
k=length(at_path);
for i=1:k
    at_path(i) = phi(1) * xt_path(i) + phi(2) + sqrt(exp(phi(3))) * randn(1);
end
xi_1 = test_func_1(at_path, xt_path);
xi_2 = test_func_2(at_path, xt_path);
xi_3 = test_func_3(at_path, xt_path);
grad_theta = mean(is.*([xi_4, xi_5, xi_6].*dmt), 1);
grad_theta = grad_theta';
grad_phi = mean([xi_1, xi_2, xi_3] .*dmt-lam.* [xi_1, xi_2, xi_3]*dt, 1);
grad_phi = grad_phi';
end



