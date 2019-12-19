function [totError] = ClassifyError(Data, gtLabels, wIn_H, wH_Out, activationFunc);
%CLASSIFYERROR
[~,N] = size(Data);
errors = zeros(1,N);
for i=1:N

   sample = [1;Data(:,i)];
   %forward propagation
   VinH = wIn_H'*sample;
   actHout = actFunc(activationFunc,VinH);
   VinOut = wH_Out'*[1;actHout];
   actOutput = actFunc('sigmoid',VinOut);   
   if(actOutput > 0.5)
       actOutput = 1;
   else
       actOutput = 0;
   end
   errors = [errors abs(gtLabels(i) - actOutput)];
    
end

totError = mean(errors);

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

