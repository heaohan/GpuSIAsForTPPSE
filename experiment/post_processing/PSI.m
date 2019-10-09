function wrappedPhase = PSI( fringes,startPhaseShift,phaseShift )
%PSI Calculate the phase from the n-step phase shifting interfermetry
%method
%   Input:
%     fringes: cell vector (n*1 or 1*n) % m*n*k
steps = size(fringes,3);
angles = (startPhaseShift:phaseShift:startPhaseShift+(steps-1)*phaseShift)';
sumCos = sum(sum(cos(angles)));
sumSin = sum(sum(sin(angles)));
sumSinCos = sum(sum(sin(angles).*cos(angles)));
sumSinSin = sum(sum(sin(angles).^2));
sumCosCos = sum(sum(cos(angles).^2));
helpMatrixOne = [steps, sumCos, sumSin;
              sumCos, sumCosCos, sumSinCos;
              sumSin, sumSinCos, sumSinSin];
helpMatrixTwo = [ones(1,steps); (cos(angles))'; (sin(angles))'];
helpMatrixThree = helpMatrixOne\helpMatrixTwo;
a1 = zeros(size(fringes(:,:,1)));
a2 = a1;
for count = 1:steps
  a1 = a1 + helpMatrixThree(2,count) * fringes(:,:,count);
  a2 = a2 + helpMatrixThree(3,count) * fringes(:,:,count);
end
wrappedPhase = atan2(-a2,a1);
end

