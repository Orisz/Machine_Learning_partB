%Project - Part B

clear all; close all; clc;
Data = load('BreastCancerData.mat');
number_of_0 = length(find(Data.y == 0));
number_of_1 = 569 - number_of_0;

% divide data to training set and test set while perserving ratio
%randomize the samples used for train and test set
idx_0s = find(Data.y == 0);
idx_1s = find(Data.y == 1);

trainNum = round(0.8*length(Data.y));
trainNum0 = round(0.6*trainNum);
trainNum1 = round(0.4*trainNum);

trainSet = Data.X(:,idx_0s(1:trainNum0));
trainSet = [trainSet Data.X(:,idx_1s(1:trainNum1))];
GTtrainSet = Data.y(idx_0s(1:trainNum0));
GTtrainSet = [GTtrainSet; Data.y(idx_1s(1:trainNum1))];
trainRand = randperm(trainNum0+trainNum1);
trainSet = trainSet(:,trainRand);
GTtrainSetAll = GTtrainSet(trainRand);

testSet = Data.X(:,idx_0s(trainNum0+1:end));
testSet = [testSet Data.X(:,idx_1s(trainNum1+1:end))];
GTtestSet = Data.y(idx_0s(trainNum0+1:end));
GTtestSet = [GTtestSet; Data.y(idx_1s(trainNum1+1:end))];
testRand = randperm(569-(trainNum0+trainNum1));
testSet = testSet(:,testRand);
GTtestSet = GTtestSet(testRand);

%normalize test set
CNTtest = (testSet - mean(testSet,2))./max(testSet,[],2);
%  divide train set to train and valid sets + normalize
CNTtrain = (trainSet - mean(trainSet,2))./max(trainSet,[],2);
CNTvalidSet = CNTtrain(:,1:floor(0.2*size(CNTtrain,2)));
GTvalid = GTtrainSetAll(1:floor(0.2*size(CNTtrain,2)));
CNTtrainSet = CNTtrain(:,floor(0.2*size(CNTtrain,2))+1:end);
CNTtestSet = (testSet - mean(testSet,2))./max(testSet,[],2);
GTtrainSet = GTtrainSetAll(floor(0.2*size(CNTtrain,2))+1:end);

%divide non normalized train set to train and valid sets

trainSetNN = trainSet(:,1:floor(0.8*size(trainSet,2)));
GTtrainSetNN = GTtrainSetAll(1:floor(0.8*size(trainSet,2)));
validSetNN = trainSet(:,floor(0.8*size(trainSet,2))+1:end);
GTvalidSetNN = GTtrainSetAll(floor(0.8*size(trainSet,2))+1:end);

%% Assignment 1 - Neural Network

%input layer - 30 neurons
%hidden layer - h neurons
%output layer - 1 neuron - for 2 classes. using sigmoid as activation
%function

%section 1 - different number of neurons in hidden layer
%parameters initialization
eta = 0.1;
hiddenLayersNum = 3:1:10;
ValidSetErr_HNum = cell(1,length(hiddenLayersNum));
TrainSetErr_HNum = ValidSetErr_HNum;
activationF = 'sigmoid';
currValidSetErr = inf; 

figure(1);
for i=1:length(hiddenLayersNum)

        weightsIn_H = [ones(1,hiddenLayersNum(i)); 0.3*ones(30,hiddenLayersNum(i))];
        weightsH_Out = [1 ; 0.3*ones(hiddenLayersNum(i),1)];
        [TrainSetErr,ValidSetErr] = neuralNet(CNTtrainSet, GTtrainSet, CNTvalidSet, GTvalid, weightsIn_H, weightsH_Out, activationF, eta);
        
        ValidSetErr_HNum{i} = ValidSetErr;
        TrainSetErr_HNum{i} = TrainSetErr;
        
        if(currValidSetErr > ValidSetErr(end))
            currValidSetErr = ValidSetErr(end);
            bestHNum = i;
        end
        
        plot(1:length(TrainSetErr),TrainSetErr);
        hold on;
end    
title('Train Set Error vs Iteraion','fontsize',20);
xlabel('iteration number');
ylabel('Error');
legend('HL size = 3','HL size = 4','HL size = 5','HL size = 6',...
    'HL size = 7','HL size = 8','HL size = 9','HL size = 10');
hold off; 

figure(2);
hold on;
for i=1:length(hiddenLayersNum)
    plot(1:length(ValidSetErr_HNum{i}),ValidSetErr_HNum{i});
    hold on;
end
title('Validation Set Error vs Iteraion','fontsize',20);
xlabel('iteration number');
ylabel('Error');
legend('HL size = 3','HL size = 4','HL size = 5','HL size = 6',...
    'HL size = 7','HL size = 8','HL size = 9','HL size = 10');
hold off; 
 

%% section 2 - different activation function of hidden layer

%parameters initialization
hiddenLayer = 3; %number of hidden layers for checking other parameters
eta = 0.1;
weightsIn_H = [ones(1,hiddenLayer); 0.3*ones(30,hiddenLayer)];
weightsH_Out = [1 ; 0.3*ones(hiddenLayer,1)];

activationFuncTypes = {'sigmoid','RELU','tanh'};
ValidSetErr_ActF = cell(1,length(activationFuncTypes));
TrainSetErr_ActF = ValidSetErr_ActF;

currValidSetErr = inf;

figure(3)
hold on;
for i=1:3
   [TrainSetErr,ValidSetErr] = neuralNet(CNTtrainSet, GTtrainSet, CNTvalidSet, GTvalid, weightsIn_H, weightsH_Out, activationFuncTypes{i}, eta);
    
   
        ValidSetErr_ActF{i} = ValidSetErr;
        TrainSetErr_ActF{i} = TrainSetErr;
        
        if(currValidSetErr > ValidSetErr(end))
            currValidSetErr = ValidSetErr(end);
            bestActF = i;
        end
        
        plot(1:length(TrainSetErr),TrainSetErr);
        hold on;
end    
title('Train Set Error vs Iteraion','fontsize',20);
xlabel('iteration number');
ylabel('Error');
legend('activation func = sigmoid','activation func = RELU','activation func = tanh');
hold off; 

figure(4);
hold on;
for i=1:length(activationFuncTypes)
    plot(1:length(ValidSetErr_ActF{i}),ValidSetErr_ActF{i});
    hold on;
end
title('Validation Set Error vs Iteraion','fontsize',20);
xlabel('iteration number');
ylabel('Error');
legend('activation func = sigmoid','activation func = RELU','activation func = tanh');
hold off;


%% section 3 - different weights initialization
%parameters initialization
hiddenLayer = 3; %number of hidden layers for checking other parameters
eta = 0.1;
activationF = 'sigmoid';
weightsIn_H1 = [ones(1,hiddenLayer); 0.3*ones(30,hiddenLayer)];
weightsH_Out1 = [1 ; 0.3*ones(hiddenLayer,1)];
weightsIn_H2 = [zeros(1,hiddenLayer); zeros(30,hiddenLayer)];
weightsH_Out2 = [0 ; zeros(hiddenLayer,1)];
weightsIn_H3 = rand(31,hiddenLayer);
weightsH_Out3 = rand(4,1);
weightsIn_H4 = (-1)*rand(31,hiddenLayer);
weightsH_Out4 = (-1)*rand(4,1);

weightsIn_H_Vec = {weightsIn_H1 , weightsIn_H2 , weightsIn_H3 , weightsIn_H4};
weightsH_Out_Vec = {weightsH_Out1 , weightsH_Out2 , weightsH_Out3 , weightsH_Out4};

ValidSetErr_Winit = cell(1,length(weightsIn_H_Vec));
TrainSetErr_Winit = ValidSetErr_Winit;
currValidSetErr = inf;


figure(5)
hold on;
for i=1:length(weightsIn_H_Vec)
    
   [TrainSetErr,ValidSetErr] = neuralNet(CNTtrainSet, GTtrainSet, CNTvalidSet, GTvalid, weightsIn_H_Vec{i},weightsH_Out_Vec{i}, activationF, eta);
    
 
        ValidSetErr_Winit{i} = ValidSetErr;
        TrainSetErr_Winit{i} = TrainSetErr;
        
        if(currValidSetErr > ValidSetErr(end))
            currValidSetErr = ValidSetErr(end);
            bestWinit = i;
        end
        
        plot(1:length(TrainSetErr),TrainSetErr);
        hold on;
end    
title('Train Set Error vs Iteraion','fontsize',20);
xlabel('iteration number');
ylabel('Error');
legend('fixed-weights' , 'zero-weights','randomized-weights','negative-randomized-weights');
hold off; 

figure(6);
hold on;
for i=1:length(weightsIn_H_Vec)
    plot(1:length(ValidSetErr_Winit{i}),ValidSetErr_Winit{i});
    hold on;
end
title('Validation Set Error vs Iteraion','fontsize',20);
xlabel('iteration number');
ylabel('Error');
legend('fixed-weights' , 'zero-weights','randomized-weights','negative-randomized-weights');
hold off;


%% section 4 - normalized data vs not-normalized data

%parameters initialization
hiddenLayer = 3; %number of hidden layers for checking other parameters
eta = 0.1;
weightsIn_H = [ones(1,hiddenLayer); 0.3*ones(30,hiddenLayer)];
weightsH_Out = [1 ; 0.3*ones(hiddenLayer,1)];
activationF = 'sigmoid';

ValidSetErr_NormD = cell(1,2);
TrainSetErr_NormD = ValidSetErr_NormD;

currValidSetErr = inf;
%i=1 - normalized data , i=2 - not normalized data
figure(7)
hold on;
for i=1:2
    if(i == 1)
        [TrainSetErr,ValidSetErr] = neuralNet(CNTtrainSet, GTtrainSet, CNTvalidSet, GTvalid, weightsIn_H,weightsH_Out, activationF, eta);
    else
        [TrainSetErr,ValidSetErr] = neuralNet(trainSetNN, GTtrainSetNN, validSetNN, GTvalidSetNN, weightsIn_H,weightsH_Out, activationF, eta);
    end
   
        ValidSetErr_NormD{i} = ValidSetErr;
        TrainSetErr_NormD{i} = TrainSetErr;
        
        if(currValidSetErr > ValidSetErr(end))
            currValidSetErr = ValidSetErr(end);
            bestNormD = i;
        end
        
        plot(1:length(TrainSetErr),TrainSetErr);
        hold on;
end    
title('Train Set Error vs Iteraion','fontsize',20);
xlabel('iteration number');
ylabel('Error');
xlim([0 250]);
legend('normalized data','Not-Normalized data');
hold off; 

figure(8);
hold on;
for i=1:2
    plot(1:length(ValidSetErr_NormD{i}),ValidSetErr_NormD{i});
    hold on;
end
title('Validation Set Error vs Iteraion','fontsize',20);
xlabel('iteration number');
ylabel('Error');
xlim([0 250]);
legend('normalized data','Not-Normalized data');
hold off;

%% section 5 - different step sizes (eta)

%parameters initialization
hiddenLayer = 3; %number of hidden layers for checking other parameters
etaVec = [0.01 0.07 0.1 0.7 1];
weightsIn_H = [ones(1,hiddenLayer); 0.3*ones(30,hiddenLayer)];
weightsH_Out = [1 ; 0.3*ones(hiddenLayer,1)];
activationF = 'sigmoid';

ValidSetErr_eta = cell(1,length(etaVec));
TrainSetErr_eta = ValidSetErr_eta;

currValidSetErr = inf;

figure(9)
hold on;
for i=1:length(etaVec)
   [TrainSetErr,ValidSetErr] = neuralNet(CNTtrainSet, GTtrainSet, CNTvalidSet, GTvalid, weightsIn_H, weightsH_Out, activationF, etaVec(i));
    
   
        ValidSetErr_eta{i} = ValidSetErr;
        TrainSetErr_eta{i} = TrainSetErr;
        
        if(currValidSetErr > ValidSetErr(end))
            currValidSetErr = ValidSetErr(end);
            best_eta = i;
        end
        
        plot(1:length(TrainSetErr),TrainSetErr);
        hold on;
end    
title('Train Set Error vs Iteraion','fontsize',20);
xlabel('iteration number');
ylabel('Error');
legend(['\eta = ' num2str(etaVec(1))],['\eta = ' num2str(etaVec(2))],['\eta = ' num2str(etaVec(3))],...
    ['\eta = ' num2str(etaVec(4))],['\eta = ' num2str(etaVec(5))]);
hold off; 

figure(10);
hold on;
for i=1:length(etaVec)
    plot(1:length(ValidSetErr_eta{i}),ValidSetErr_eta{i});
    hold on;
end
title('Validation Set Error vs Iteraion','fontsize',20);
xlabel('iteration number');
ylabel('Error');
legend(['\eta = ' num2str(etaVec(1))],['\eta = ' num2str(etaVec(2))],['\eta = ' num2str(etaVec(3))],...
    ['\eta = ' num2str(etaVec(4))],['\eta = ' num2str(etaVec(5))]);
hold off;



%% section 6 - run neural network on all data using the best parameters

hiddenLayer = hiddenLayersNum(bestHNum);
eta = etaVec(best_eta);
activationF = activationFuncTypes{bestActF};

weightsIn_H_best = weightsIn_H_Vec{bestWinit};
weightsH_Out_best = weightsH_Out_Vec{bestWinit};

if(bestNormD == 1)
   AllData = CNTtrain; 
else %better not to normalize data
   AllData = trainSet;
end
tic;
[TrainSetErr,TestSetErr] = neuralNet(AllData, GTtrainSetAll, CNTtest, GTtestSet, weightsIn_H, weightsH_Out, activationF, eta);
elapedNN = toc;
figure();
plot(1:length(TrainSetErr) , TrainSetErr);
hold on;
plot(1:length(TestSetErr) , TestSetErr);
legend('Train Set Error' , 'Test Set Error');
title('Error Vs. Iteration');
xlabel('Iteration');
ylabel('Error');
hold off;

%% Assignment 2 - SVM
tic;
N = length(GTtrainSetAll); % 80% of all data
setSize = floor(N/10);

CNTrainSet = (trainSet - mean(trainSet,2))./max(trainSet,[],2);
CNTestSet = (testSet - mean(testSet,2))./max(testSet,[],2);

%for choosing the optimal SVM from the cross validation
BestSVMModel = cell(3,1);
BestKernel = [];
ValidSetError = []; %10 sets of train set and valid set
i=2;

currMinValidError = zeros(1,3) + inf;
MinValidErrors = zeros(10,3);
for i=1:3 % 1 - linear ; 2 - gaussian ; 3 - polynomial
%loop - cross validation for training. find the one to have the smallest error
    %loo = leave one out (cross validation method)
   for loo=1:10
        %for cross validation
        
        Train90 = CNTrainSet;
        Train90(:,((loo-1)*setSize+1):loo*setSize) = [];
        TrainLabels90 = GTtrainSetAll;
        TrainLabels90(((loo-1)*setSize+1):loo*setSize) = []; 
        ValidSet = CNTrainSet(:,((loo-1)*setSize+1):loo*setSize);
        ValidLabels = GTtrainSetAll(((loo-1)*setSize+1):loo*setSize);

        %SVM
        % KernelFunction option: specify the kernel used by SVM algorithm
        % BoxConstraint option: determine the constraint C for each sample
        if(i==1)
            SVMM = fitcsvm(Train90',TrainLabels90,'KernelFunction','linear','BoxConstraint',0.8);
            
            %run SVM result on validation set
            ValidOutputLabel = predict(SVMM,ValidSet');
            ValidSetError = mean(abs(ValidOutputLabel - ValidLabels));
            MinValidErrors(loo,i) = ValidSetError;
            if(currMinValidError(i) > ValidSetError)
               currMinValidError(i) = ValidSetError;
               BestSVMModel{i} = SVMM;
            end
            
        elseif(i==2) %KernelScale = sqrt(c)
            SVMM = fitcsvm(Train90',TrainLabels90,'KernelFunction','gaussian','BoxConstraint',1);
                 
            %run SVM result on validation set
            ValidOutputLabel = predict(SVMM,ValidSet');
            ValidSetError = mean(abs(ValidOutputLabel - ValidLabels));
            MinValidErrors(loo,i) = ValidSetError;
            if(currMinValidError(i) > ValidSetError)
               currMinValidError(i) = ValidSetError;
               BestSVMModel{i} = SVMM;
            end
        else
            SVMM = fitcsvm(Train90',TrainLabels90,'KernelFunction','polynomial','PolynomialOrder',3,'BoxConstraint',1);
                 
            %run SVM result on validation set
            ValidOutputLabel = predict(SVMM,ValidSet');
            ValidSetError = mean(abs(ValidOutputLabel - ValidLabels));
            MinValidErrors(loo,i) = ValidSetError;
            if(currMinValidError(i) > ValidSetError)
               currMinValidError(i) = ValidSetError;
               BestSVMModel{i} = SVMM;
            end
        end
   end
end
%calculate errors for all 10 validation leave-one-out set
 meanErrorPeri = mean(MinValidErrors,1);
 STDError = std(MinValidErrors,1);

%run on testSet
[TestOutputLabel_Linear,scores_Linear] = predict(BestSVMModel{1},CNTestSet');
TestSetError_Linear = mean(abs(TestOutputLabel_Linear - GTtestSet));

[TestOutputLabel_Gaussian,scores_Gaussian] = predict(BestSVMModel{2},CNTestSet');
TestSetError_Gaussian = mean(abs(TestOutputLabel_Gaussian - GTtestSet));

[TestOutputLabel_Poly,scores_Poly] = predict(BestSVMModel{3},CNTestSet');
TestSetError_Poly = mean(abs(TestOutputLabel_Poly - GTtestSet));
estimatedTimeSVM = toc;
%%
CNTDataX = (Data.X - mean(Data.X,2))./max(Data.X,[],2);

pca_result = pca(CNTDataX'); %get principal component coefficients
%project data on the first two component coefficients
component1 = CNTDataX'*pca_result(:,1);
component2 = CNTDataX'*pca_result(:,2);
data2 = [component1 component2];
%SVMplotModel = fitcsvm(data2,Data.y,'KernelFunction','linear','BoxConstraint',0.8);
%SVMplotModel = fitcsvm(data2,Data.y,'KernelFunction','gaussian','BoxConstraint',1);
SVMplotModel = fitcsvm(data2,Data.y,'KernelFunction','polynomial','PolynomialOrder',3,'BoxConstraint',1);

trainSetIdx = [idx_0s(1:trainNum0)' idx_1s(1:trainNum1)'];
testSetIdx = [idx_0s((trainNum0+1):end)' idx_1s((trainNum1+1):end)'];

d=0.02;
[x1Grid,x2Grid] = meshgrid(min(component1):d:max(component1),...
    min(component2):d:max(component2));
[x1X,x1Y] = size(x1Grid);
xGrid = [x1Grid(:) , x2Grid(:)];
[~,scores] = predict(SVMplotModel,xGrid);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure();
h(1:2) = gscatter(component1,component2,Data.y,'rg','+*');
hold on
sv1 = SVMplotModel.SupportVectors(:,1);
sv2 = SVMplotModel.SupportVectors(:,2);
plot(sv1,sv2,'ko','MarkerSize',10);
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
legend('0 (training)','1 (training)','Support Vectors','Location','Southeast');
axis equal
hold off
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%calculate w using the support vectors
% SVs = SVMplotModel.IsSupportVector;
% SV_comp1 = component1(SVs == 1); %holds the samples' component1
% SV_comp2 = component2(SVs == 1); %holds the samples' component2
% SV_y = Data.y(SVs == 1);
% SV_comps = [SV_comp1 SV_comp2];
% w=sum(((SVMplotModel.Alpha.*SV_y).*SV_comps),1);
% w = w';
% 
% %try to determine the line for SVs that uphold (wT*x_i+b)<epsilon ((wT*x_i+b)=0 in theory)
% %didn't quite work
% epsilon = 0.05;
% b=-SVMplotModel.Bias;
% decisionVec = abs(w'*SV_comps'+b);
% defineLine = find(decisionVec<epsilon);
% LineCoord = SV_comps(defineLine,:);
% 
% figure;
% gscatter(component1,component2,Data.y);
% hold on
% sv1 = SVMplotModel.SupportVectors(:,1);
% sv2 = SVMplotModel.SupportVectors(:,2);
% plot(sv1,sv2,'ko','MarkerSize',10);
% %calculate the line w^T*x+b using an example found online
% m = w(1)/w(2);
% f=@(x) m.*x+b;
% fplot(f,[min(component1) max(component1)],'k');
% xlim([min(component1)-0.5 max(component1)+1]);
% ylim([min(component2)-1 max(component2)+1]);
% legend('0','1','Support Vector')
% hold off;