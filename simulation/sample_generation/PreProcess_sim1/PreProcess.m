% % % % % % % % % % % % % % parameter setting
order = 3;  % Zernike order
J = double((order+1)*(order+2)/2);  % number of terms

% Search boundaries must be LARGER than the fringe number* in the mask
%D = J + 1;   % total number of unknowns, dimensions D = 11 in this case
xmax = [1; 10*ones(J-1, 1); 1]*pi; % upper search boundary
xmin = -xmax;  % lower search boundary 

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % generate interferograms
mCols = 256;
x = linspace(-0.9, 0.9, mCols);
y = linspace(-0.9, 0.9, mCols);
[X, Y] = meshgrid(x, y);

additive_noise = 0.6*randn(mCols);
OPD = 0.25*(X.^2 + Y.^2 - 10*cos(2*pi*X) - 10*cos(2*pi*Y) );  % Rastrigin's function
phi_true = 2*pi*OPD;

delta_true = 0.01*pi;  % simulated phase shift
Io = zeros(size(phi_true,1),size(phi_true,2),2);
Io(:, :, 1) = cos(phi_true) + additive_noise;
Io(:, :, 2) = cos(phi_true + delta_true) + additive_noise;

% % % % % % Gaussian high-pass filtering 
sigma = 0.03;
x = linspace(-1, 1, mCols);
[X, Y] = meshgrid(x, x);

Ghp = 1 - exp(-(X.^2+Y.^2)./sigma^2);   % Gaussian high pass
I_filtered = zeros(size(Io));
I_filtered(:, :, 1) = real(ifft2(ifftshift( fftshift(fft2( Io(:, :, 1) )).*Ghp )));  % use the real component
I_filtered(:, :, 2) = real(ifft2(ifftshift( fftshift(fft2( Io(:, :, 2) )).*Ghp )));  % use the real component

% % % % % Define the mask
% xc = 128;  yc = 128;  % mask 1
xc = 200;  yc = 130;  % mask 2
% xc = 95;  yc = 165;   % mask 3
radius = round( 0.2*(mCols/2) );

if yc-radius<1 || xc-radius<1 || xc+radius>mCols || xc+radius>mCols
    error('exceed matrix boundaries');
else
    Imask = I_filtered(yc-radius:yc+radius, xc-radius:xc+radius, :);  % take the mask out
end

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % compute standard Zernike polynomials
mColsMask = size(Imask, 1);
x = linspace(-1, 1, mColsMask);  % unit square
[X, Y] = meshgrid(x, x);
[theta, rou] = cart2pol(X, Y);
mask = rou<=1;  % unit disk

j = 1:J;
n = fix(sqrt(2*j-1)+0.5) - 1;
m = zeros(1,length(j));
for k = 1:length(n)
    if ~mod(n(k),2)  % even
        m1 = 2 * fix( ( 2*j(k)+1 - n(k).*(n(k)+1) )/4 );
    else
        m1 = 2 * fix( ( 2*(j(k)+1) - n(k).*(n(k)+1) )/4 ) - 1;
    end
    m(k) = m1;  % m is non-negative
end
zern = zernStandardFun(n, m, rou(mask), theta(mask), 'nonnorm');

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % Fringe pattern normalization
In = zeros(size(Imask));
In(:, :, 1) = normalize( Imask(:, :, 1) ); % quasi-Quadurature transform
In(:, :, 2) = normalize( Imask(:, :, 2) ); % quasi-Quadurature transform

% Vectorize the intensity data
in_vector = zeros(sum(mask(:)), 2);  % pre-locate memory
for i = 1:2
    Itemp = In(:, :, i);
    in_vector(:, i) = Itemp(mask(:));  % vectorize, InVector: 3096:*2
end

in_vector = single(in_vector);
xmin = single(xmin);
xmax = single(xmax);
zern = single(zern);
save for_c.mat in_vector zern xmin xmax
save for_further.mat Io delta_true I_filtered xc yc radius zern in_vector








