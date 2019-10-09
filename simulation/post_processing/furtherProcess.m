clear
selpath = uigetdir(pwd, 'result folder');
indexStr = strfind(selpath, '\');
selpathUp = selpath(1:indexStr(end-1)-1);

load([selpath '\' 'g_best.mat']); %best fitness
load ([selpath '\' 'g_best_pos.mat']); %dims
load ([selpathUp '\' 'for_further.mat'], 'radius', 'I_filtered', 'zern',...
    'in_vector', 'xc', 'yc', 'Io');

% 
% load g_best.mat %best fitness
% load g_best_pos.mat %dims
% load for_further.mat %xc yc radius I_filtered zern in_vector

coeffs = g_best_pos(1:end-1);  % estimated coefficients
delta = g_best_pos(end);  % estimated phase shift

x = linspace(-1, 1, 2*radius+1);  % unit square
[X, Y] = meshgrid(x, x);
[theta, rou] = cart2pol(X, Y);
mask = rou<=1;  % unit disk

phiMask = zeros(2*radius+1);
phiTemp = zern*coeffs;
phiMask(mask) = phiTemp;
Ir = cos(phiMask);  % reconstructed intensity map

load ([selpathUp '\' 'phi_true.mat']); %phi_true.mat

tp = phi_true(yc-radius:yc+radius, xc-radius:xc+radius);
phiTrueMask = zeros(size(tp));
phiTrueMask(mask) = tp(mask);
residuePhiMask = phiMask - phiTrueMask;
residuePhiMask(mask) = residuePhiMask(mask) - mean(residuePhiMask(mask));
residue_phi_mask_1 = phiMask + phiTrueMask;
residue_phi_mask_1(mask) = residue_phi_mask_1(mask) - mean(residue_phi_mask_1(mask));
if (rms(residuePhiMask(mask)) > rms(residue_phi_mask_1(mask)))
    residuePhiMask = residue_phi_mask_1;
end

% phi_wrapped = phiMask - round(phiMask/(2*pi)) * (2*pi);  % rewrap the phase for comparison
phiMaskWrapped = mod(phiMask, 2*pi);  % rewrap the phase for comparison
phiMaskWrapped = adjustPiston(mod(phiTrueMask,2*pi).*mask, phiMaskWrapped,mask);

Imask = zeros(2*radius+1); % one of the original fringe in aperature
Imask(mask) = in_vector(:,1);

figure(2), 
subplot(2,4,1), imagesc(mod(phiTrueMask,2*pi).*mask),
colormap gray(256), axis square, axis off
title('True phase in the mask');

subplot(2,4,5),imagesc(cos(phiTrueMask).*mask),
colormap gray(256), axis square, axis off
title('True intensity in the mask');

subplot(2,4,2),imagesc(Io(yc-radius:yc+radius, xc-radius:xc+radius).*mask),
colormap gray(256), axis square, axis off
title('Original intensity in the mask');

subplot(2,4,6),imagesc(Imask.*mask), colormap gray(256), axis square, axis off
title('Intensity in the mask for optimization');

subplot(2,4,3),imagesc(phiMaskWrapped.*mask), colormap gray(256), axis square, axis off
title('Retrieved phase in the mask')

subplot(2,4,7), imagesc(Ir.*mask), colormap gray(256), axis square, axis off
title('Retrieved intensity in the mask')

subplot(2,4,4), imagesc(residuePhiMask), colormap gray(256), axis square, axis off
title('Residue phase in the mask')

subplot(2,4,8), imagesc((mod(residuePhiMask,2*pi).*mask)), colormap gray(256), axis square, axis off
title('Residue phase (wrapped) in the mask')

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % phase retrieval
%[I_filtered(:,:,1),~,~,~,~] = BSEMD_m( I_filtered(:,:,1),2 );
%[I_filtered(:,:,2),~,~,~,~] = BSEMD_m( I_filtered(:,:,2),2 );
Q1 = (I_filtered(:, :, 1)*cos(delta) - I_filtered(:, :, 2))./sin(delta);
I1 = I_filtered(:, :, 1);
tpPhase = atan2(Q1, I1);
tpPhase = adjustPiston(phi_true,tpPhase);
figure, imagesc(tpPhase), colormap gray(256), axis square, axis off
title('Retrieved phase')

load([selpath '\' 'fit_rec.mat']);
% load fit_rec.mat
fit_rec_processed = fit_rec(find(fit_rec > 0, 1):end);
if (exist([selpath '\' 'iter_rec.mat'], 'file') == 2)
    load([selpath '\' 'iter_rec.mat']);
% if (exist('iter_rec.mat', 'file') == 2)
%     load iter_rec.mat
    iter_rec_processed = iter_rec(find(fit_rec > 0, 1):end);
    figure, plot(iter_rec_processed, fit_rec_processed,'r*'); 
    title('fitness'); xlabel('iteration');
    ylabel('fitness');
else
%     fit_rec_processed = fit_rec(find(fit_rec > 0, 1):end);
    figure, plot(fit_rec_processed,'r*'); title('fitness'); xlabel('iteration');
    ylabel('fitness');
end

figure,imshow(Io(:,:,1),[]);
viscircles([xc,yc],radius,'DrawBackgroundCircle',false,'LineStyle','--');

