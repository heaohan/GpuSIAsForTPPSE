clear
selpath = uigetdir(pwd, 'result folder');
indexStr = strfind(selpath, '\');
selpathUp = selpath(1:indexStr(end-1)-1);

load([selpath '\' 'g_best.mat']); %best fitness
load ([selpath '\' 'g_best_pos.mat']); %dims
load ([selpathUp '\' 'for_further.mat'], 'radius', 'zern',...
    'in_vector', 'I_filtered', 'xc', 'yc', 'delta_true'); %radius I_bsemd zern in_vector

figure,imshow(I_filtered(:,:,1),[]);
viscircles([xc,yc],radius,'DrawBackgroundCircle',false,'LineStyle','--');

coeffs = g_best_pos(1:end-1);  % estimated coefficients
delta = abs(g_best_pos(end));  % estimated phase shift

x = linspace(-1, 1, 2*radius+1);  % unit square
[X, Y] = meshgrid(x, x);
[theta, rou] = cart2pol(X, Y);
mask = rou<=1;  % unit disk

phiMask = zeros(2*radius+1);
phiTemp = zern*coeffs;
phiMask(mask) = phiTemp;
phi_wrapped = phiMask - round(phiMask/(2*pi)) * (2*pi);  % rewrap the phase for comparison
Ir = cos(phiMask);  % reconstructed intensity map

Io = zeros(2*radius+1); % one of the original fringe in aperature
Io_1 = Io;
Io(mask) = in_vector(:,1);
Io_1(mask) = in_vector(:,2);

figure, 
subplot(2,3,4), imagesc(Io.*mask), colormap gray(256), axis square, axis off
title('Intensity in the mask for optimization');
subplot(2,3,5), imagesc(Io_1.*mask), colormap gray(256), axis square, axis off
title('Intensity in the mask for optimization: the other');
subplot(2,3,3), imagesc(phi_wrapped.*mask), colormap gray(256), axis square, axis off
title('Retrieved phase in the mask')
subplot(2,3,6), imagesc(Ir.*mask), colormap gray(256), axis square, axis off
title('Computed intensity map in the mask')
subplot(2,3,1), imagesc(I_filtered(yc-radius:yc+radius, xc-radius:xc+radius, 1).*mask),...
colormap gray(256), axis square, axis off
title('Original intensity in the mask')
subplot(2,3,2), imagesc(I_filtered(yc-radius:yc+radius, xc-radius:xc+radius, 2).*mask),...
colormap gray(256), axis square, axis off
title('Original intensity in the mask: the other')
%saveas(gcf,[selpath '\' 'aperture.bmp']);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % phase retrieval
load ([selpathUp '\' 'tpPts-pts.mat'], 'BW', 'pts');
load ([selpathUp '\' 'tpPts-pts.mat'], 'transformedPts',...
    'tform', 'slmRow', 'slmCol', 'chessRow', 'chessCol',...
    'upleftChessVertex');

unwrapFlag = false;

% using just high-pass filtered fringe to retrieve
Q1 = (I_filtered(:, :, 1)*cos(delta) - I_filtered(:, :, 2))./sin(delta);
I1 = I_filtered(:, :, 1);
tpPhase = atan2(Q1, I1);

Q1 = (I_filtered(:, :, 1)*cos(delta_true) - I_filtered(:, :, 2))./sin(delta_true);
I1 = I_filtered(:, :, 1);
tpPhase_true = atan2(Q1, I1);
argout_hp = ...
    TransformProcess(BW, pts, transformedPts, tform, slmRow, slmCol,...
    chessRow, chessCol, upleftChessVertex, tpPhase, unwrapFlag);
argout_hp_true = ...
    TransformProcess(BW, pts, transformedPts, tform, slmRow, slmCol,...
    chessRow, chessCol, upleftChessVertex, tpPhase_true, false);

load([selpath '\' 'fit_rec.mat']);
fit_rec_processed = fit_rec(find(fit_rec > 0, 1):end);
if (exist([selpath '\' 'iter_rec.mat'], 'file') == 2)
    load([selpath '\' 'iter_rec.mat']);
    iter_rec_processed = iter_rec(find(fit_rec > 0, 1):end);
    figure, plot(iter_rec_processed, fit_rec_processed,'r*'); 
    title('fitness'); xlabel('iteration');
    ylabel('fitness');
else
    %fit_rec_processed = fit_rec(find(fit_rec > 0, 1):end);
    figure, plot(fit_rec_processed,'r*'); title('fitness'); xlabel('iteration');
    ylabel('fitness');
end

answer = questdlg('Simple mode? no PSI phase', ...
	'Simple mode or not', ...
	'Yes','No','Yes');
if strcmp(answer, 'Yes')
    simpleFlag = true;
else
    simpleFlag = false;
end

if ~simpleFlag
    % result from 4 step phase shift
    uiwait(msgbox('Select cut images for PSI!'));
    [filename,user_canceled] = imgetfile('MultiSelect',true, 'InitialPath', selpathUp);
    if user_canceled
        error('No image selected!');
    end

    % must make sure the filename is in correct order
    fid = fopen([selpathUp '\' 'loadImages.txt'],'w');
    imagenames = cell(2,1);
    for count = 1:length(filename)
        [~,tp,~] = fileparts(filename{count});
        fprintf(fid,'%s\r\n',tp);
        imagenames{count} = tp;
    end
    fclose(fid);
    dos(['notepad ' selpathUp '\loadImages.txt &']);
    h = msgbox('Please check the sequence of images!');
    uiwait(h);

    tp = mat2gray(imread(filename{1}));
    Im = zeros(size(tp,1),size(tp,2),length(filename));
    for count = 1:length(filename)
        tp = mat2gray(imread(filename{count}));
        if ndims(tp) ~= 3
            Im(:,:,count) = mat2gray(tp);
        else
            Im(:,:,count) = mat2gray(tp(:,:,1));
        end
    end
    tpPhase = PSI( Im,0,pi/2 ); % start from 0
    argout_PSI = ...
        TransformProcess(BW, pts, transformedPts, tform, slmRow, slmCol,...
        chessRow, chessCol, upleftChessVertex, tpPhase, unwrapFlag);
    
    figure,
    subplot(1,3,1),
    imshow(argout_hp,[]);
    title('hp');
    subplot(1,3,2),
    imshow(argout_PSI,[]);
    title('PSI');
    subplot(1,3,3),imshow(argout_hp_true,[]);
    title('hp with delta true');
else
    figure,
    subplot(1,2,1),
    imshow(argout_hp,[]);
    title('hp');
    subplot(1,2,2),
    imshow(argout_hp_true,[]);
    title('hp with delta true');
end



%%
function argout = ...
    TransformProcess(BW, pts, transformedPts, tform, slmRow, slmCol,...
    chessRow, chessCol, upleftChessVertex, tpPhase, unwrapFlag)
wrappedPhase = zeros(size(BW));
wrappedPhase(min(pts(:,2)):max(pts(:,2)),min(pts(:,1)):max(pts(:,1))) =...
  tpPhase;
%figure,imshow(wrappedPhase,[]);
tpPhase1 = mod(tpPhase,2*pi);

if unwrapFlag
    qualityMap = qualityMapByPhaseDerivativeVariance(tpPhase1);
    tpUnwrappedPhase = fp_unwrapping_qg_i2l2(tpPhase1,qualityMap);
    unwrappedPhase = zeros(size(BW));
    unwrappedPhase(min(pts(:,2)):max(pts(:,2)),min(pts(:,1)):max(pts(:,1))) =...
      tpUnwrappedPhase;
    unwrappedPhaseTransformed = imwarp(unwrappedPhase,tform);
    unwrappedPhaseTransformed = unwrappedPhaseTransformed(...
        min(transformedPts(:,2)):max(transformedPts(:,2)),...
        min(transformedPts(:,1)):max(transformedPts(:,1)));
    unwrappedPhaseTransformed = imresize(unwrappedPhaseTransformed,[chessRow chessCol]);
    unwrappedPhaseTransformed = unwrappedPhaseTransformed - min(min(unwrappedPhaseTransformed));
    unwrappedPhaseTransformed = rot90(unwrappedPhaseTransformed,2);
    figure,mesh(unwrappedPhaseTransformed/4/pi);
    xlabel('x / pixel'), ylabel('y / pixel'), zlabel('wavelength / \lambda');
    unwrappedPhaseTransformedSLM = zeros(slmRow,slmCol);
    unwrappedPhaseTransformedSLM(upleftChessVertex(1):upleftChessVertex(1)+chessRow-1,...
        upleftChessVertex(2):upleftChessVertex(2)+chessCol-1) = unwrappedPhaseTransformed;
end

wrappedPhaseTransformed = imwarp(wrappedPhase,tform);
wrappedPhaseTransformed = wrappedPhaseTransformed(...
    min(transformedPts(:,2)):max(transformedPts(:,2)),...
    min(transformedPts(:,1)):max(transformedPts(:,1)));
wrappedPhaseTransformed = imresize(wrappedPhaseTransformed,[chessRow chessCol]);
wrappedPhaseTransformed = rot90(wrappedPhaseTransformed,2);
% qualityMap = qualityMapByPhaseDerivativeVariance(wrappedPhaseTransformed);
% unwrappedPhase = fp_unwrapping_qg_i2l2(wrappedPhaseTransformed,qualityMap);

wrappedPhaseTransformedSLM = zeros(slmRow, slmCol);
wrappedPhaseTransformedSLM(upleftChessVertex(1):upleftChessVertex(1)+chessRow-1,...
    upleftChessVertex(2):upleftChessVertex(2)+chessCol-1) = wrappedPhaseTransformed;

if unwrapFlag
    argout{1} = wrappedPhaseTransformedSLM;
    argout{2} = unwrappedPhaseTransformedSLM;
else
    argout = wrappedPhaseTransformedSLM;
end

end
