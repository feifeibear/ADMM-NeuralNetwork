function [z] = argminlastz(targets, lambda, w, a_in, beta)
    % Minimization of the last output matrix, using the above function. We use least square loss instead of binhinge loss

    % :param targets:  target matrix (equal dimensions of z) (y)
    % :param eps:      lagrange multiplier matrix (equal dimensions of z) (lambda)
    % :param w:        weight matrix (w_l)
    % :param a_in:     activation matrix l-1 (a_l-1)
    % :return:         output matrix last layer

    %  Author: Thanh Nguyen-Duc (Potato Thanh)
    %  Email: thanhnguyen.cse@gmail.com
    %  June 2017

    m = w*a_in;
    z = (targets - lambda + beta*m)./(1+beta);
    
    
end