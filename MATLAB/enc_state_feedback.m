clear
close all
format compact

% setting
Ts = 10e-3;
t = 0:Ts:10;

% plant
A = [1 -1; 0 2];
B = [0; 1];
C = [1 0; 0 1];
D = 0;
sys = c2d(ss(A,B,C,D),Ts);
[A,B,C,D] = ssdata(sys);

% dimension
n = size(A,1);
m = size(B,2);

% initial state
x0 = ones(n,1);

% controller
Q = diag(ones(1,n));
R = diag(ones(1,m));
F = dlqr(A,B,Q,R);
F = -F;

% cryptosystem
bitLength = 20;
csys = ElGamal(bitLength);

% scaling gain
gamma_c = 1e2;
gamma_p = 1e2;

%% simulation w/o encryption

% state
x = x0*ones(1,length(t)+1);
% input
u = zeros(m,length(t));

% simulation
for k = 1:length(t)
    u(:,k) = F*x(:,k);
    x(:,k+1) = A*x(:,k)+B*u(:,k);
end

%% simulation w/ encryption

% state
x_ = x0*ones(1,length(t)+1);
% input
u_ = zeros(m,length(t));
% controller input
x_enc = zeros(n,length(t),2);
% controller output
u_enc = zeros(m,n,length(t),2);

% controller encryption
F_enc = zeros(m,n,2);
for i = 1:m
    for j = 1:n
        F_enc(i,j,:) = csys.Enc(F(i,j),gamma_c);
    end
end

% simulation
for k = 1:length(t)

    for i = 1:n
        x_enc(i,k,:) = csys.Enc(x_(i,k),gamma_p);
    end

    for i = 1:m
        for j = 1:n
            u_enc(i,j,k,:) = csys.Mult(F_enc(i,j,:),x_enc(j,k,:));
        end
    end

    for i = 1:m
        u_(i,k) = csys.Dec(u_enc(i,1,k,:),gamma_c*gamma_p);
        for j = 2:n
            u_(i,k) = u_(i,k)+csys.Dec(u_enc(i,j,k,:),gamma_c*gamma_p);
        end
    end

    x_(:,k+1) = A*x_(:,k)+B*u_(:,k);
end

%% result
close all

% control input
figure('Units','centimeters','Position',[2 2 2+1.62*8 2+1*8])
hold on
p(1) = plot(t,u(1,:),'-','Color',[132/255 145/255 158/255],'LineWidth',6);
p(2) = plot(t,u_(1,:),'-','Color',[0 90/255 1],'LineWidth',3);
p(3) = plot(t,zeros(1,length(t)),'--','Color',[0 0 0],'LineWidth',1);
ylim([-4 1])
xlim([0 10])
yticks([-4 -3 -2 -1 0 1])
xticks([0 5 10])
legend([p(1) p(2)],{'w/o encryption','w/ encryption'},'FontName','Times New Roman','Location','southeast')
ylabel('$$u$$','FontName','Times New Roman','Interpreter','latex','FontSize',20)
xlabel('Time (s)','FontName','Times New Roman','FontSize',20)
set(gca,'FontName','Times New Roman','FontSize',20)
box on

% state
figure('Units','centimeters','Position',[2 2 2+1.62*8 2+1*8])
hold on
p(1) = plot(t,x(1,1:length(t)),'-','Color',[132/255 145/255 158/255],'LineWidth',6);
p(2) = plot(t,x(2,1:length(t)),'-','Color',[132/255 145/255 158/255],'LineWidth',6);
p(3) = plot(t,x_(1,1:length(t)),'-','Color',[0 90/255 1],'LineWidth',3);
p(4) = plot(t,x_(2,1:length(t)),'-','Color',[0 90/255 1],'LineWidth',3);
p(5) = plot(t,zeros(1,length(t)),'--','Color',[0 0 0],'LineWidth',1);
ylim([-0.15 1.5])
xlim([0 10])
yticks([0 0.5 1.0 1.5])
xticks([0 5 10])
legend([p(1) p(3)],{'w/o encryption','w/ encryption'},'FontName','Times New Roman','Location','northeast')
ylabel('$$x$$','FontName','Times New Roman','Interpreter','latex','FontSize',20)
xlabel('Time (s)','FontName','Times New Roman','FontSize',20)
set(gca,'FontName','Times New Roman','FontSize',20)
box on