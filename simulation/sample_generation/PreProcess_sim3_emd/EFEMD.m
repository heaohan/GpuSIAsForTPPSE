function [efemd,AvD] = EFEMD(Im, MaxAvDist,IMFnumb)
%This function calculates Extremely Fast Empirical Mode Decomposition
%Matlab Image Processing Toolbox is required
%Im - image to be decomposed
%MaxAvDist - maximum accepted value of the mean distance between extrema,
%to be used as stop condition
%IMFnumb - number of decomposition elements minus 1, to be used as stop condition 
%AvD - vector of average distances, calculated in each loop
%Example:
%Decomposition = EFEMD(Image);
%This code can be used freely for research purposes.
%Please cite one of the following papers if you use this code:
%.....
%not yet published, new codes
%Maciek Wielgus [www.mwielgus.cba.pl], Maciek Trusiak 
%Institute of Micromechanics and Photonics
%Warsaw University of Technology, March 2013

% This EFEMD has been modified to only calculate the BIMF1; efemd = []
% when avdist > MaxAvDist

[m0,n0] = size(Im);

if nargin < 3
  IMFnumb = 0; stopCond=1;
    if nargin < 2  
    MaxAvDist = max(m0,n0)/3; %Maximum average distance between extrema large enough so that the residue is close to being monotonous
    end  
else stopCond = 0;
end

%---Preparing data with the mirroring procedure

ext = round(min(m0,n0)/10); %extension thickness 10%
f1 = flipud(Im(1:ext,:)) ;
f2 = flipud(Im(m0-ext+1:m0,:));
f3 = fliplr(Im(:,1:ext));
f4 = fliplr(Im(:,n0-ext+1:n0)) ;
f5 = Im(1:ext,1:ext);
f6 = Im(1:ext,n0-ext+1:n0);
f7 = Im(m0-ext+1:m0,1:ext);
f8 = Im(m0-ext+1:m0,n0-ext+1:n0);
Im = [fliplr(rot90(f5)), f1, f6'; f3, Im, f4; f7', f2, fliplr(rot90(f8))];
%---------------------------------------------

[M,N] = size(Im);

%---Initial calculation of extrema number and filter size
maskExt = strel('square',3); % strel('disk',3);
MaxDil = imdilate(Im,maskExt); %MAX filtration
MinDil = -imdilate(-Im,maskExt); %MIN filtration

MaxMap = ~(Im - MaxDil); %binary map of maxima
MinMap = ~(Im - MinDil); %binary map of minima
NumbExtrema = round(0.5*(sum(sum(MaxMap)) + sum(sum(MinMap)))); %mean number of extrema
AvDist = round(sqrt(N*M/NumbExtrema)); %mean distance between extrema
AvD(1) =AvDist;

maskDil = strel('disk',round(AvDist/2));  %mask for dilation operation
%maskDil = strel('square',AvDist);  %mask for dilation operation
maskConv = fspecial('disk',round(AvDist/2));%mask for smoothing
%maskConv = fspecial('average',[AvDist,AvDist]);%mask for smoothing
%--------------------------------------------------------

cou = 1;
% pointer = 0;

while (cou <= IMFnumb) || (stopCond==1) %sifting loop
  MaxDil = imdilate(Im,maskDil); %MAX filtration
  MinDil = -imdilate(-Im,maskDil); %MIN filtration
  
  MaxMap = ~(Im - MaxDil);
  MinMap = ~(Im - MinDil);
  NumbExtrema = round(0.5*(sum(sum(MaxMap)) + sum(sum(MinMap))));
  AvDistNew = round(sqrt(N*M/NumbExtrema));

%---Determining the size of morphological filters used in the next iteration    
  if AvDistNew > AvDist
    AvDist = AvDistNew; %mean distance between extrema
    %[cou,AvDist]
  else
    AvDist = round(2*AvDist);
  end
%---------------------------------------------------------------    
AvD(cou+1) =AvDist;

%---Breaking the sifting loop if extrema distributed too sparsely
  if AvDist > MaxAvDist
    efemd = [];
%     efemd{cou} = Im(ext+1:m0+ext,ext+1:n0+ext);
%     pointer = 1;
    break
  end
  %---------------------------------------------------------------
  SmoothMean = conv2(0.5*(MaxDil + MinDil),maskConv,'same');
  %---Saving IMF and defining new input for the sifting procedure
  IMF = Im - SmoothMean;
  efemd{cou} = IMF(ext+1:m0+ext ,ext+1:n0+ext);
  Im = SmoothMean;
  %--------------------------------------------------------------
  maskDil = strel('disk',round(AvDist/2));  %mask for dilation operation
  %maskDil = strel('square',AvDist);  %mask for dilation operation
  maskConv = fspecial('disk',round(AvDist/2));%mask for smoothing
  %maskConv = fspecial('average',[AvDist,AvDist]);%mask for smoothing
  cou = cou+1;
end

% if pointer ==0
%   efemd{cou} = Im(ext+1:m0+ext ,ext+1:n0+ext); %saving the residue
% end

end