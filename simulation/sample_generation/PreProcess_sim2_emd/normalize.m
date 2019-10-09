function In = normalize(Ihp)
% Normalize fringe patterns using the quasi-quadrature transform method in [1]
% Input:
%   Ihp   high-pass filtered fringe intensity
%
% Output:
%   In    normalized fringe intensity 
%
% Chao Tian at the University of Michigan
% Last modified: May 25, 2017
% 
% Reference: 
% [1] J. A. Quiroga and M. Servin, "Isotropic n-dimensional fringe pattern 
%     normalization," Optics communications 224, 221-227 (2003).


[nRows,nCols]= size(Ihp);


% % % % % % % % % % % % % % % quasi-Quadrature transform
lb = floor(nCols/2);  % 128 -> 64; 127 -> 63
ub = floor(nRows/2);  % 128 -> 64; 127 -> 63
u = -lb : nCols-lb-1;  % spectrum is in the center of the image
v = -ub : nRows-ub-1;    % 128 -> -64:63; 127 -> -63:63
[U,V] = meshgrid(u,v);
e1 = 1;
e2 = 1i;

flag2 = 1;
if flag2==1
    FI1 = (-1i*U./sqrt(U.^2+V.^2)).*fftshift(fft2((Ihp)));
    FI2 = (-1i*V./sqrt(U.^2+V.^2)).*fftshift(fft2((Ihp)));
else
    FI1 = (-1i*U./sqrt(U.^2+V.^2)).*fftshift(fft2((I))).*fi;  % different methods
    FI2 = (-1i*V./sqrt(U.^2+V.^2)).*fftshift(fft2((I))).*fi;
end
FI1(isnan(FI1)) = 0;  % 
FI2(isnan(FI2)) = 0;

H2 = (ifft2(ifftshift(FI1)))*e1 + (ifft2(ifftshift(FI2)))*e2;
Q2 = abs(H2);  % quasi-quadrature opterator


% % % compute Inorm
wrappedPhi1 = atan2(-Q2, Ihp);  % wrapped phase
In = cos(wrappedPhi1);



