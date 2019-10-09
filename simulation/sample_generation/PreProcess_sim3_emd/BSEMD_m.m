function [I_F,I_noise,I_background,k1,k2] = BSEMD_m( I,mode )
%BSEMD_m
%   Modified BSEMD; provide 2 mode
% mode == 1: the k1 and k2 are estimated during the decomposition;
% mode == 2: decomposition and grouping are separate; after grouping, will
% provide the k2 and the image of after k2 BIMFs, and provide the choice
% for manually determining k2.
if nargin < 2
  mode = 1;
end

MaxAvDist = max(size(I)) / 3; % 3 5 % 5 for others, 3 for RPT
IMFnumb = 1;
I_i = I;
i = 1;
k1 = NaN;
k2 = NaN;

tic
if mode == 1
  while isnan(k2)
    BIMFt_1 = EFEMD(I_i,MaxAvDist,IMFnumb);
    t = toc;
    if isempty(BIMFt_1) || t > 30 % when T > MaxAvDist, the result of EFEMD will be NaN
      k2 = length(BIMF) - 1;
      break;
    end
    s_i = SinDesign(BIMFt_1{1},i);
    BIMFt_1_plus = EFEMD(BIMFt_1{1} + s_i,MaxAvDist,IMFnumb);
    BIMFt_1_minus = EFEMD(BIMFt_1{1} - s_i,MaxAvDist,IMFnumb);
    if isempty(BIMFt_1_plus) || isempty(BIMFt_1_minus) % when T > MaxAvDist, the result of EFEMD will be NaN
      k2 = length(BIMF) - 1;
      break;
    end
    BIMF{i} = (BIMFt_1_plus{1} + BIMFt_1_minus{1})/2;
    I_i = I_i - BIMF{i};
    i = i + 1;
    if isnan(k1)
      pos = K1Cal(BIMF);
      if ~isnan(pos)
        k1 = pos;
      end
    else
      if length(BIMF) - k1 >= 3
        pos = K2Cal(BIMF,k1);
        if ~isnan(pos)
          k2 = pos;
        end
      end
      
    end
  end
  
  
else
  while true
    BIMFt_1 = EFEMD(I_i,MaxAvDist,IMFnumb);
    t = toc;
    if isempty(BIMFt_1) || t > 30 % when T > MaxAvDist, the result of EFEMD will be []
      break;
    end
    s_i = SinDesign(BIMFt_1{1},i);
    BIMFt_1_plus = EFEMD(BIMFt_1{1} + s_i,MaxAvDist,IMFnumb);
    BIMFt_1_minus = EFEMD(BIMFt_1{1} - s_i,MaxAvDist,IMFnumb);
    if isempty(BIMFt_1_plus) || isempty(BIMFt_1_minus) % when T > MaxAvDist, the result of EFEMD will be []
      break;
    end
    BIMF{i} = (BIMFt_1_plus{1} + BIMFt_1_minus{1})/2;
    I_i = I_i - BIMF{i};
    i = i + 1;
  end
  k1 = K1Cal(BIMF);
  k2 = K2Cal(BIMF,k1);
  if isnan(k2)
    k2 = length(BIMF) - 1;
  end
  h = msgbox(['Estimated k2:' num2str(k2) ' Number of BIMF:' num2str(length(BIMF))]);
  uiwait(h);
  prompt = 'Enter the number displayed after k2';
  dlg_title = 'Display until';
  num_lines = 1;
  defAns = {num2str(k2+8)};
  options.Resize='on';
  options.WindowStyle='normal';
  options.Interpreter='tex';
  tp = inputdlg(prompt,dlg_title,num_lines,defAns,options);
  tp = uint16(floor(eval(tp{1})));
  if tp > length(BIMF) || tp < k2
      error('Wrong number for display!');
  end
  figs = cell(tp - k2 + 1,1);
%   if length(BIMF)-k2+1 > 8
%     figs = cell(8,1);
%   else
%     figs = cell(length(BIMF)-k2+1,1);
%   end
  for i = k2:k2+length(figs)-1
    figs{i-k2+1} = figure;
    imshow(BIMF{i},[]);title(['BIMF{',num2str(i),'}']);
  end

  prompt = 'Enter the k2';
  dlg_title = 'Input for k2';
  num_lines = 1;
  defAns = {num2str(k2)};
  options.Resize='on';
  options.WindowStyle='normal';
  options.Interpreter='tex';
  tp = inputdlg(prompt,dlg_title,num_lines,defAns,options);
  k2 = eval(tp{1});
  
  for i = 1:length(figs)
    close(figs{i});
  end
end

  
% reconstruction
EK = zeros(k2-k1,1);
for i = k1+1:k2
  EK(i-k1) = sum(sum(BIMF{i}.^2));
end
[~,k3] = max(EK);
k3 = k3 + k1;

if k3 ~= k1+1
  for i = k1+1:k3
    modulation=HS(BIMF{i});
    thresh = multithresh(modulation,i-1);
    IDX = imquantize(modulation,thresh);
%     IDX = otsu(modulation,i);
    str_size = 2/FgCal(BIMF{i});
    SE = strel('disk',round(str_size/2));
    IDX = imopen(IDX,SE);
    IDX = imclose(IDX,SE);
    IDX = imdilate(IDX,SE);
    BIMF{i}(IDX == 1) = 0;
  end
end

I_F = zeros(size(I));
for i = k1+1:k2
  I_F = BIMF{i} + I_F;
end

I_noise = zeros(size(I));
for i = 1:k1-1
  I_noise = BIMF{i} + I_noise;
end
I_background = I - I_F - I_noise;

end

function s_i = SinDesign(BIMFt_1,i)
if i == 1
  f_i = 0.5;
else
  [M,N] = size(BIMFt_1);
  maskExt = strel('square',3); % strel('disk',3);
  MaxDil = imdilate(BIMFt_1,maskExt); %MAX filtration
  MinDil = -imdilate(-BIMFt_1,maskExt); %MIN filtration

  MaxMap = ~(BIMFt_1 - MaxDil); %binary map of maxima
  MinMap = ~(BIMFt_1 - MinDil); %binary map of minima
  NumbExtrema = round(0.5*(sum(sum(MaxMap)) + sum(sum(MinMap)))); %mean number of extrema
  AvD = round(sqrt(N*M/NumbExtrema)); %mean distance between extrema
  f_i = 1.5 / AvD;
end
a_i = 4*max(max(abs(BIMFt_1)));
[x,y] = meshgrid(linspace(0,size(BIMFt_1,2)-1,size(BIMFt_1,2)),linspace(0,size(BIMFt_1,1)-1,size(BIMFt_1,1)));
s_i = a_i * cos(2*pi*f_i*x) .* cos(2*pi*f_i*y);
end

function pos = K1Cal(BIMF)
if length(BIMF) == 1
  pos = NaN;
  return;
end
% E1 = sum(sum(BIMF{1}.^2));
% E2 = sum(sum(BIMF{2}.^2));
% if E1 < E2
%   pos = 1;
%   return;
% end
for i = 1:length(BIMF)-1
  E1 = sum(sum(BIMF{i}.^2));
  E2 = sum(sum(BIMF{i+1}.^2));
  if E1 < E2
    pos = i;
    return;
  end
end
pos = NaN;
% error('k1 cannot be determined\n');
end

function pos = K2Cal(BIMF,k1)
pos = NaN;
k2_c = NaN;
if k1+1 > length(BIMF)-2 -2 % at least 3 BIMFs are used for the k2_c determination
  return;
end

for i = k1+1:length(BIMF)-2 % at least 3 BIMFs are left for k2 determination: BIMF{end-2:end}
  if length(BIMF)-2 - k1 < 2 
    return;
  end
  if i ~= k1+1
    tp1 = tp2;
    tp2 = FgCal(BIMF{i+1});
  else
    tp1 = FgCal(BIMF{i});
    tp2 = FgCal(BIMF{i+1});
  end
  Rf = tp1 / tp2;
  if Rf > 2 %default 2; may need adjustment (1.1)
    k2_c = i;
    break;
  end
end
if isnan(k2_c) % when Rf > 2 is not found
%   pos = length(BIMF)-1;
  return;
end

for i = k2_c:length(BIMF)-2
  if i ~= k2_c
    Raf1 = Raf2;
    Raf2 = Raf3;
    Raf3 = RafCal(BIMF{i+2});
  else
    Raf1 = RafCal(BIMF{i});
    Raf2 = RafCal(BIMF{i+1});
    Raf3 = RafCal(BIMF{i+2});
  end
  if Raf2 < Raf1 && Raf2 < Raf3
    pos = i+1;
    return;
  end
  if Raf2 > Raf1 && Raf2 > Raf3
    pos = i;
    return;
  end
end
if length(BIMF) - k1 >= 5 % make sure k2 is determined when there are enough signal BIMFs
  if Raf2 > Raf1 % may not suitable when inital fringe pattern is too sparse
    pos = k1 + 1;
  else
    pos = length(BIMF)-1;
  end
end

end

% function fg = FgCal(imf)
% [mRow,mCol] = size(imf);
% corr = xcorr2(imf);
% rowCorr = mCol * (corr(mRow,mCol:-1:1))' ./ (corr(mRow,mCol).*...
%   (mCol:-1:1)');
% colCorr = mRow * (corr(mRow:-1:1,mCol)) ./ (corr(mRow,mCol).*...
%   (mRow:-1:1)');
% [~,posRow] = findpeaks(rowCorr);
% [~,posCol] = findpeaks(colCorr);
% fg = hypot(1/(posRow(1)-1),1/(posCol(1)-1));
% end

function fg = FgCal(imf)
imf = mat2gray(imf);
[mRow,mCol] = size(imf);
centerValue = sum(sum(imf.^2));
tp1 = imf; tp2 = imf; tp3 = imf;
f1g = 1/(mCol-2);
for x_t = 1:mCol-2
  if x_t ~= 1
    tp1 = tp2;
    tp2 = tp3;
    tp3(:,x_t+2:end) = tp3(:,x_t+1:end-1); tp3(:,x_t+1) = 0;
    a1 = a2;
    a2 = a3;
    a3 = mCol * sum(sum(imf.*tp3)) / (centerValue * (mCol-x_t-1));
  else
    tp2(:,2:end) = tp2(:,1:end-1); tp2(:,1) = 0;
    tp3(:,3:end) = tp3(:,1:end-2); tp3(:,1:2) = 0;
    a1 = mCol * sum(sum(imf.*tp1)) / (centerValue * (mCol-x_t+1));
    a2 = mCol * sum(sum(imf.*tp2)) / (centerValue * (mCol-x_t));
    a3 = mCol * sum(sum(imf.*tp3)) / (centerValue * (mCol-x_t-1));
  end

  if a2 > a1 && a2 > a3
    f1g = 1/x_t;
    break;
  end
end
tp1 = imf; tp2 = imf; tp3 = imf;
f2g = 1/(mRow-2);
for y_t = 1:mRow-2
  if y_t ~= 1
    tp1 = tp2;
    tp2 = tp3;
    tp3(y_t+2:end,:) = tp3(y_t+1:end-1,:); tp3(y_t+1,:) = 0;
    a1 = a2;
    a2 = a3;
    a3 = mRow * sum(sum(imf.*tp3)) / (centerValue * (mRow-y_t-1));
  else
    tp2(2:end,:) = tp2(1:end-1,:); tp2(1,:) = 0;
    tp3(3:end,:) = tp3(1:end-2,:); tp3(1:2,:) = 0;
    a1 = mRow * sum(sum(imf.*tp1)) / (centerValue * (mRow-y_t+1));
    a2 = mRow * sum(sum(imf.*tp2)) / (centerValue * (mRow-y_t));
    a3 = mRow * sum(sum(imf.*tp3)) / (centerValue * (mRow-y_t-1));
  end
  if a2 > a1 && a2 > a3
    f2g = 1/y_t;
    break;
  end
end
fg = hypot(f1g,f2g);
end


% function fg = FgCal(imf)
% for x_t = 1:size(imf,2)-2
%   a1 = sum(sum(imf.*ImfChange(imf,x_t-1,'x')));
%   a2 = sum(sum(imf.*ImfChange(imf,x_t,'x')));
%   a3 = sum(sum(imf.*ImfChange(imf,x_t+1,'x')));
%   if a2 > a1 && a2 > a3 && a2 > 0 % changed: add a2 > 0
%     f1g = 1/x_t;
%     break;
%   end
% end
% for y_t = 1:size(imf,1)-2
%   a1 = sum(sum(imf.*ImfChange(imf,y_t-1,'y')));
%   a2 = sum(sum(imf.*ImfChange(imf,y_t,'y')));
%   a3 = sum(sum(imf.*ImfChange(imf,y_t+1,'y')));
%   if a2 > a1 && a2 > a3 && a2 > 0 % changed: add a2 > 0
%     f2g = 1/y_t;
%     break;
%   end
% end
% fg = sqrt(f1g.^2 + f2g.^2);
% end

% function imf = ImfChange(imf,t,dir)
% if t == 0
%   return
% end
% if dir == 'x'
% %   tp = imf(:,1:t);
%   imf(:,1:end-t) = imf(:,t+1:end);
%   imf(:,end-t+1:end) = 0;%tp;
% elseif dir == 'y'
% %   tp = imf(1:t,:);
%   imf(1:end-t,:) = imf(t+1:end,:);
%   imf(end-t+1:end,:) = 0;%tp;
% else
%   error('uncorrect input\n');
% end
% end

function Raf1 = RafCal(imf)
Raf1 = sqrt(sum(sum(imf.^2))) / FgCal(imf);
end



  
