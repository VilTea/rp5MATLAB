clear ; 
clc;
%T（地面以上以上2米处的大气温度）、Po（气象站水平的大气压）、P（平均海平面的大气压）、
%U（地面高度2米处的相对湿度）、N（总云量）、VV（水平能见度）、Td（地面高度2米处的露点温度）、RRR（降水量）
%% load
inputfile = '北京2009-2018.xls';
%日期
[~,datetxt] = xlsread(inputfile, 1, 'A8:A27492');
%date = datetime(datetxt, 'InputFormat', 'dd.MM.yyyy HH');
date = datenum(datetxt,'dd.MM.yyyy HH');
dateplot = datetime(datetxt, 'InputFormat', 'dd.MM.yyyy HH:mm');
%数据集
[num, txt, raw] = xlsread(inputfile);
index = [1,2,3,5,21,22];
data = num(:,index);
%label={'T','Po','P','U', 'VV', 'Td'};
%% 数据初始化
[M, N] = size(data);
% T的缺失值较少，人工处理
data(1262, 1) = (data(1261, 1) + data(1263, 1)) / 2;
data(11252, 1) = (data(11251, 1) + data(11253, 1)) / 2;
% Po的缺失值处理
for i = 1:M             %处理Po缺失值，用最近两个数据的均值填充
    if isnan(data(i, 2))
       j = i;
       while isnan(data(j - 1, 2))
           j = j - 1;
       end
       k = i;
       while isnan(data(k + 1, 2))
           k = k + 1;
       end
       data(i, 2) = (data(j - 1, 2) + data(k + 1, 2))/2;
    end    
end
% P的缺失值处理
for i = 1:M             %处理P缺失值，用最近两个数据的均值填充
    if isnan(data(i, 3))
       j = i;
       while isnan(data(j - 1, 3))
           j = j - 1;
       end
       k = i;
       while isnan(data(k + 1, 3))
           k = k + 1;
       end
       data(i, 3) = (data(j - 1, 3) + data(k + 1, 3))/2;
    end    
end
% U的缺失值较少，人工处理
data(1262, 4) = (data(1261, 4) + data(1263, 4)) / 2;
data(11252, 4) = (data(11251, 4) + data(11253, 4)) / 2;
data(14367, 4) = (data(14366, 4) + data(14368, 4)) / 2;
data(15163, 4) = (data(15164, 4) + data(15165, 4)) / 2;
data(15167, 4) = (data(15166, 4) + data(15169, 4)) / 2;
data(15168, 4) = (data(15167, 4) + data(15169, 4)) / 2;
% VV的NaN数值有两种情况，'低于0.1'或缺失值，
for i = 1:M             %处理VV缺失值，用最近两个数据的均值填充
    if isnan(data(i, 5))
       if strcmp(txt(i + 7,index(5) + 1), '')
            j = i;
            while isnan(data(j - 1, 5))
                j = j - 1;
            end
            k = i;
            while isnan(data(k + 1, 5))
                k = k + 1;
            end
            data(i, 5) = (data(j - 1, 5) + data(k + 1, 5))/2;
       else
            data(i, 5) = 0.05; %若遇'低于0.1'，置为0.05
       end 
    end
end
% Td的缺失值较少，人工处理
data(14367, 6) = (data(14366, 6) + data(14368, 6)) / 2;
data(15163, 6) = (data(15162, 6) + data(15164, 6)) / 2;
data(15167, 6) = (data(15166, 6) + data(15169, 6)) / 2;
data(15168, 6) = (data(15167, 6) + data(15169, 6)) / 2;
% for i = 1:M             %找出缺失值位置
%     if isnan(data(i, 6))
%         disp(i);
%         disp(txt(i + 7,index(6) + 1))
%     end    
% end
%% 构造函数曲线
dataplot = data; %原始数据作图
r = corrcoef(dataplot);  %求六个属性两两之间的相关系数，选取相关系数绝对值大于80%的，并作出关系曲线图
dateplot = datenum(dateplot);
dateplot = dateplot(1:25:end,1);
% dateplot = flipud(dataplot);
%%  T（地面以上2米处的大气温度）随日期的变化曲线
T = dataplot(:,1);
T = T(1:25:end,1);
dateT = [dateplot';T']';
x1 = dateT(:,1);
y1 = dateT(:,2);
H1 = figure;
plot(x1,y1,'r');
title('温度时间曲线');
xlabel('时间');
ylabel('温度');
datetick('x', 26); % 将坐标轴设置为日期格式

%%   Po（气象站水平的大气压）随日期的变化曲线
Po = dataplot(:,2);
Po = Po(1:25:end,1);
datePo = [dateplot';Po']';
x2 = datePo(:,1);
y2 = datePo(:,2);
H2 = figure;
plot(x2,y2,'g');
title('水平气压时间曲线');
xlabel('时间');
ylabel('水平气压');
datetick('x', 26); % 将坐标轴设置为日期格式

%%  P（平均海平面的大气压）随日期的变化曲线
P = dataplot(:,3);
P = P(1:25:end,1);
dateP = [dateplot';P']';
x3 = dateP(:,1);
y3 = dateP(:,2);
H3 = figure;
plot(x3,y3,'b');
title('海平面气压时间曲线');
xlabel('时间');
ylabel('海平面气压');
datetick('x', 26); % 将坐标轴设置为日期格式

%%  U（地面高度2米处的相对湿度）随日期变化曲线

U = dataplot(:,4);
U = U(1:25:end,1);
dateT = [dateplot';U']';
x4 = dateT(:,1);
y4 = dateT(:,2);
H4 = figure;
plot(x4,y4,'r');
title('相对湿度时间曲线');
xlabel('时间');
ylabel('地面2米处相对湿度');
datetick('x', 26); % 将坐标轴设置为日期格式

%%  VV（水平能见度）随时间变化曲线
VV = dataplot(:,5);
VV = VV(1:25:end,1);
dateT = [dateplot';VV']';
x5 = dateT(:,1);
y5 = dateT(:,2);
H5 = figure;
plot(x5,y5,'g');
title('水平能见度时间曲线');
xlabel('时间');
ylabel('水平能见度');
datetick('x', 26); % 将坐标轴设置为日期格式

%%  Td（地面高度2米处的露点温度）随时间变化曲线
Td = dataplot(:,6);
Td = Td(1:25:end,1);
dateTd = [dateplot';Td']';
x6 = dateTd(:,1);
y6 = dateTd(:,2);
H6 = figure;
plot(x6,y6,'b');
title('地面高度2米处露点温度时间曲线');
xlabel('时间');
ylabel('地面高度2米处露点温度');
datetick('x', 26); % 将坐标轴设置为日期格式

%根据相关系数矩阵r得出相关系数百分之八十以上有四对，TPo关系图,TP关系图,,TTd关系图，PoP关系图
%% T（地面以上以上2米处的大气温度）与Po（气象站水平的大气压）关系散点图
T = dataplot(:,1);
Po = dataplot(:,2);
T = T(1:25:end,1);
Po = Po(1:25:end,1);
TPo = [T';Po']';
x7 = TPo(:,1);
y7 = TPo(:,2);
H7 = figure;
plot(x7,y7,'r.');
title('地面以上2米处大气温度与水平大气压关系散点图');
xlabel('地面以上2米处大气温度');
ylabel('Po（气象站水平的大气压）');

%% T（地面以上2米处的大气温度）与P（平均海平面的大气压）的关系散点图
T = dataplot(:,1);
P = dataplot(:,3);
T = T(1:25:end,1);
P = P(1:25:end,1);
TP = [T';P']';
x8 = TP(:,1);
y8 = TP(:,2);
H8 = figure;
plot(x8,y8,'b.');
title('地面以上2米处大气温度与平均海平面大气压关系散点图');
xlabel('地面以上2米处大气温度');
ylabel('平均海平面大气压');
%% T（地面以上2米处的大气温度）与Td（地面高度2米处的露点温度）的关系散点图
T = dataplot(:,1);
Td = dataplot(:,6);
T = T(1:25:end,1);
Td = Td(1:25:end,1);
TTd = [T';Td']';
x9 = TTd(:,1);
y9 = TTd(:,2);
H9 = figure;
plot(x9,y9,'g.');
title('地面以上2米处的大气温度与地面高度2米处的露点温度关系散点图');
xlabel('地面以上2米处大气温度');
ylabel('地面高度2米处露点温度');
%% Po（气象站水平的大气压）与P（平均海平面的大气压）的关系散点图
P = dataplot(:,2);
Po = dataplot(:,3);
P = P(1:25:end,1);
Po = Po(1:25:end,1);
PPo = [P';Po']';
x10 = PPo(:,1);
y10 = PPo(:,2);
H10 = figure;
plot(x10,y10,'b.');
title('水平大气压与平均海平面大气压的关系散点图');
xlabel('气象站水平大气压');
ylabel('平均海平面大气压');
%% 划分训练集、测试集
data = [date data];
data = flipud(data);
labels = {'地面以上以上2米处的大气温度', '气象站水平的大气压', '平均海平面的大气压', '地面高度2米处的相对湿度', '水平能见度', '地面高度2米处的露点温度'};
%[D,D_means,D_std] = zscore(data); %均值标准化
trains = data(1:7/10*M,:);
[trains,train_means,train_std] = zscore(trains);
tests = data(7/10*M + 1:M,:);
[tests,test_mean,test_std] = zscore(tests);
%% 降维
[coef,score,latent,t2] = pca(trains); 
latent = 100 * latent / sum(latent); %获取贡献度，百分比形式展示
% disp(latent);
%剔除贡献度过低的属性
matrix = score(:,[1,2,3]);
score_test = tests * coef;
matrix_test = score_test(:,[1,2,3]);
% retain_dimensions = 4;
% [U,S,V] = svd(cov(data));
% reduced_X = data*U(:,1:retain_dimen-sions);
% disp(data);
%% 构造神经网络模型
% 输入层输出层
input = [matrix(1:3:7/10*M-2, :) matrix(2:3:7/10*M-1, :)]';
target = [matrix(3:3:7/10*M, :)]';
itest = [matrix_test(1:3:3*M/10-2, :) matrix_test(2:3:3*M/10-1, :)]';
ttest = [matrix_test(3:3:3*M/10, :)]';

[input, ps_input] = mapminmax(input,0,1);%归一化
i_test = mapminmax('apply', itest, ps_input);
[target, ps_target] = mapminmax(target,0,1);%归一化
% 神经网络参数
net = patternnet(20); %隐含层神经个数
net.trainFcn = 'trainlm';
net.layers{1}.transferFcn ='logsig';
net.layers{2}.transferFcn ='logsig';
net.trainParam.epochs = 50000; %迭代次数
net.trainParam.show = 50; %训练结果显示频率
net.trainParam.showCommandLine = 0;
%net.trainParam.mc = 0.95; %附加动量因子
%net.trainParam.Mu = 0;
net.trainParam.lr = 0.005; %学习速率
net.trainParam.showWindow = 1; 
net.trainParam.goal = 1e-5; %训练目标最小误差 
net.trainParam.time = inf; %最大训练时间
net.trainParam.min_grad = 1e-12; %性能函数的最小梯度
net.trainParam.max_fail = 10; %最大失败次数
net.performFcn='mse'; %误差判断

[net, tr]= train(net,input,target);
disp('BP神经网络训练完成！');

%y = sim(net,input);
%plotconfusion(target,y);
t = sim(net,i_test); %仿真测试
T_sim = mapminmax('reverse', t, ps_target); %反归一化
save model net;
save pca_coef coef;
save ps_input ps_input;
save ps_target ps_target;

%绘图
result = matrix_test;
result(3:3:3*M/10, :) = T_sim';
result = result * coef(:,[1,2,3])';
result = result .* repmat(test_std,size(tests,1),1) + repmat(test_mean,size(tests,1),1);%反标准化
testss = tests .* repmat(test_std,size(tests,1),1) + repmat(test_mean,size(tests,1),1);%反标准化

testplot = testss(3:3:3*M/10,:);
resultplot = result(3:3:3*M/10,:);

figure
%'地面以上以上2米处的大气温度'
plot(1:10:M/10,testplot(1:10:M/10, 2),'b:',1:10:M/10,resultplot(1:10:M/10, 2),'r-')
title(strcat(labels(1), '回归预测曲线'));
ylabel(labels(1));
xlabel('测试集中数据序号');
legend('真实值','预测值')

figure
% '气象站水平的大气压', 
plot(1:10:M/10,testplot(1:10:M/10, 3),'b:',1:10:M/10,resultplot(1:10:M/10, 3),'r-')
title(strcat(labels(2), '回归预测曲线'));
ylabel(labels(2));
xlabel('测试集中数据序号');
legend('真实值','预测值')

figure
%'平均海平面的大气压', 
plot(1:10:M/10,testplot(1:10:M/10, 4),'b:',1:10:M/10,resultplot(1:10:M/10, 4),'r-')
title(strcat(labels(3), '回归预测曲线'));
ylabel(labels(3));
xlabel('测试集中数据序号');
legend('真实值','预测值')

figure
%'地面高度2米处的相对湿度'
plot(1:10:M/10,testplot(1:10:M/10, 5),'b:',1:10:M/10,resultplot(1:10:M/10, 5),'r-')
title(strcat(labels(4), '回归预测曲线'));
ylabel(labels(4));
xlabel('测试集中数据序号');
legend('真实值','预测值')

figure
%'水平能见度'
plot(1:10:M/10,testplot(1:10:M/10, 6),'b:',1:10:M/10,resultplot(1:10:M/10, 6),'r-')
title(strcat(labels(5), '回归预测曲线'));
ylabel(labels(5));
xlabel('测试集中数据序号');
legend('真实值','预测值')

figure
%, '地面高度2米处的露点温度'
plot(1:10:M/10,testplot(1:10:M/10, 7),'b:',1:10:M/10,resultplot(1:10:M/10, 7),'r-')
title(strcat(labels(6), '回归预测曲线'));
ylabel(labels(6));
xlabel('测试集中数据序号');
legend('真实值','预测值')