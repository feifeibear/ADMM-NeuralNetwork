function [weights] = NeuralNetwork(train_X, train_Y, test_X, test_Y, opts)

%  Author: Thanh Nguyen-Duc (Potato Thanh)
%  Email: thanhnguyen.cse@gmail.com
%  June 2017

    %init variable to GPU
    a0 = gpuArray.zeros(opts.a0Size);

    w1 = gpuArray.zeros(opts.w1Size); %256x784
    w2 = gpuArray.zeros(opts.w2Size); %256x256
    w3 = gpuArray.zeros(opts.w3Size); %10x256

    z1 = gpuArray.randn(opts.z1Size); %256xn
    a1 = gpuArray.randn(opts.a1Size); %256xn
    z2 = gpuArray.randn(opts.z2Size); %256xn
    a2 = gpuArray.randn(opts.a2Size); %256xn
    z3 = gpuArray.randn(opts.z3Size); %10xn

    lambda = gpuArray.ones(opts.lambda); %10xn

    beta3  = gpuArray(opts.beta3);
    beta2  = gpuArray(opts.beta2);
    beta1  = gpuArray(opts.beta1);

    gama2  = gpuArray(opts.gama2);
    gama1  = gpuArray(opts.gama1);

    rho    = gpuArray(opts.rho); %params for l2 regularization

    maxIter  = gpuArray(opts.maxIter);
    numLayer = gpuArray(opts.numLayer);

    % assign data to GPU 
    trainX = gpuArray(trainX);
    trainY = gpuArray(trainY);
    testX  = gpuArray(testX);
    testY  = gpuArray(testY);

    a0 = trainX;
    y_labels  = trainY;

    %warm start (run without update lambda)
    warm = opts.iswarm;
    if warm
        for i=1:opts.numWarm
            disp('--warming--');
            disp(i);
            disp('----');
            
            %layer 1
            w1  =  weight_update(z1, a0, rho);
            a1  =  activation_update(w2, z2, z1, beta2, gama1);
            z1  =  argminz(a1, w1, a0, beta1, gama1);

            %layer 2
            w2  =  weight_update(z2, a1, rho);
            a2  =  activation_update(w3, z3, z2, beta3, gama2);
            z2  =  argminz(a2, w2, a1, beta2, gama2);

            %layer 3
            w3  =  weight_update(z3, a2, rho);
            z3  =  argminlastz(y_labels, lambda, w3, a2, beta3);
        end
    end

    iter_TrainLoss = [];
    iter_TestLoss = [];
    iter_TrainAccuracy = [];
    iter_TestAccuracy = [];
    for i=1:maxIter
        disp('----------------------------');
        disp('--training--');
        disp(i);
        disp('----');
        %layer 1
        w1  =  weight_update(z1, a0, rho);
        a1  =  activation_update(w2, z2, z1, beta2, gama1);
        z1  =  argminz(a1, w1, a0, beta1, gama1);

        %layer 2
        w2  =  weight_update(z2, a1, rho);
        a2  =  activation_update(w3, z3, z2, beta3, gama2);
        z2  =  argminz(a2, w2, a1, beta2, gama2);

        %layer 3
        w3  =  weight_update(z3, a2, rho);
        z3  =  argminlastz(y_labels, lambda, w3, a2, beta3);
        lambda = lambda_update(z3, w3, a2, beta3);
        

        % Training data
        forward = w3*relu(w2*relu(w1*trainX));
        loss_train = (forward - trainY).^2;
        loss_train = sum(loss_train(:))
        [M1, I1] = max(y_labels);
        [M2, I2] = max(forward);
        accracy_train = mean(I1 == I2)

        disp('------');

        %test data
        forward1 = w3*relu(w2*relu(w1*testX));
        loss_test = (forward1 - testY).^2;
        loss_test = sum(loss_test(:))
        [M11, I11] = max(testY);
        [M22, I22] = max(forward1);
        accracy_test = mean(I11 == I22)
        
        %drawing curve
        iter_TrainLoss = [iter_TrainLoss, loss_train];
        iter_TrainAccuracy = [iter_TrainAccuracy, accracy_train];
        iter_TestLoss = [iter_TestLoss, loss_test];
        iter_TestAccuracy = [iter_TestAccuracy, accracy_test];
        
        if opts.isShow
            figure(1);
            hold on;
            plot(iter_TrainLoss);
            hold on;
            plot(iter_TestLoss);
            xlabel('Iterations');
            ylabel('loss');
            drawnow();

            figure(2);
            hold on;
            plot(iter_TrainAccuracy);
            hold on;
            plot(iter_TestAccuracy);
            xlabel('Iterations');
            ylabel('accuracy');
            drawnow();
        end
    end

    %output weights after training
    weights.w1 = gather(w1);
    weights.w2 = gather(w2);
    weights.w3 = gather(w3);
end