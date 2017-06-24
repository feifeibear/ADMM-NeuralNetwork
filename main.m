
% Implemented based on Taylor, Gavin, et al. 
% Training neural networks without gradients: A scalable admm approach.
% International Conference on Machine Learning. 2016.

% NeuralNetwork has 1 hidden layer, using least square loss instead of binhinge loss
% MNIST for traing and testing data

%  Author: Thanh Nguyen-Duc (Potato Thanh)
%  Email: thanhnguyen.cse@gmail.com
%  June 2017


clear all, close all, clc;
g = gpuDevice(1);
reset(g);

% load data
load trainX
load trainY
load testY
load testX

n_hidden_1 = 256; % 1st layer number of features
n_hidden_2 = 256; % 2nd layer number of features
n_input = 784; % MNIST data input (img shape: 28*28)
n_classes = 10; % MNIST total classes (0-9 digits) 
n_batchsize = size(trainX, 2); %MNIST number of sample 6000

opts.a0Size = [n_input, n_batchsize]; %784xn

opts.w1Size = [n_hidden_1, n_input]; %256x784
opts.w2Size = [n_hidden_2, n_hidden_1]; %256x256
opts.w3Size = [n_classes, n_hidden_2]; %10x256

opts.z1Size = [n_hidden_1, n_batchsize]; %256xn
opts.a1Size = [n_hidden_1, n_batchsize]; %256xn
opts.z2Size = [n_hidden_2, n_batchsize]; %256xn
opts.a2Size = [n_hidden_2, n_batchsize]; %256xn
opts.z3Size = [n_classes , n_batchsize]; %10xn

opts.lambda = [n_classes, n_batchsize]; %10xn

opts.beta3  = 5;
opts.beta2  = 5;
opts.beta1  = 5;

opts.gama2  = 5;
opts.gama1  = 5;

opts.rho    = 0.0; %params for l2 regularization

opts.maxIter = 50;
opts.numLayer= 3;

%option for warm start
opts.iswarm  = 1;
opts.numWarm = 5;

%option for show curve
opts.isShow = 1;

%training + testing model
weights = NeuralNetwork(trainX, trainY, testX, testY);

%save model
save weights.mat weights;
