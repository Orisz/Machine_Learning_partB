function [TrainSetErr,ValidSetErr] = neuralNet(trainData, trainLabels, validData, validLabels, wIn_H, wH_Out, activationFunc, learnRate)
%NEURALNET implement learning - forward and backward propagation

% wIn_H,wH_Out contain bias
[D,N] = size(trainData);

TrainSetErr = [];
ValidSetErr = [];
%for diffrent sizes of hidden layer we need to duplicate 'sample' for grad
%calc
hidLayerSize = size(wIn_H,2);
maxIter = 1000;
thresh = 1e-4;
for j=1:maxIter
    randIdx = randperm(N);
    for i=1:N %for each sample in train data
       sample = [1;trainData(:,randIdx(i))];
       sampleMat = [];
       for k=1:hidLayerSize
            sampleMat = [sampleMat sample];
       end
       %forward propagation
       VinH = wIn_H'*sample;
       actHout = actFunc(activationFunc,VinH);
       VinOut = wH_Out'*[1;actHout];
       actOutput = actFunc('sigmoid',VinOut);   
       %backpropagation
       dError_dVinOut = (trainLabels(randIdx(i)) - actOutput).*actOutput.*(1-actOutput); %d/dVinOut
       grad_w2 = dError_dVinOut.*[1;actHout]; %d/dw2 (w2 = wH_Out)

       %sample transpose? wH_Out(2:end)?
       if strcmp(activationFunc, 'sigmoid')
            grad_w1 = (dError_dVinOut.*wH_Out(2:end).*actHout.*(1-actHout))'.*sampleMat;
       elseif strcmp(activationFunc, 'RELU')
            grad_w1 = (dError_dVinOut.*wH_Out(2:end).*(VinH>0))'.*sampleMat; 
       else %tanh
            grad_w1 = (dError_dVinOut.*wH_Out(2:end).*tanh_der(VinH))'.*sampleMat;
       end
       %update weights
       prev_wIn_H = wIn_H;
       prev_wH_Out = wH_Out;
       wIn_H = prev_wIn_H + learnRate*grad_w1; % -1 factor for eta included
       wH_Out = prev_wH_Out + learnRate*grad_w2; % -1 factor for eta included
    end
    
    TrainSetErr(j) = ClassifyError(trainData, trainLabels, wIn_H, wH_Out, activationFunc);
    ValidSetErr(j) = ClassifyError(validData, validLabels, wIn_H, wH_Out, activationFunc);

    if(norm(wH_Out-prev_wH_Out)<thresh && norm(wIn_H-prev_wIn_H)<thresh)
        break;
    end
end

end

function result = actFunc(activationFunc,VinH)

    if strcmp(activationFunc, 'sigmoid')
        result = 1./(1+exp(-VinH));
    elseif strcmp(activationFunc, 'RELU')
        result = max(VinH,0);
    else %tanh
        result = (1-exp(-2*VinH))./(1+exp(-2*VinH));
    end
end

function result = tanh_der(VinH)
    result = (1-((1-exp(-2*VinH))./(1+exp(-2*VinH))).^2);
end
