function z = zernStandardFun(n,m,r,theta,nflag)
%ZERNFUN Zernike functions of order N and frequency M on the unit circle.
%   Z = ZERNFUN(N,M,R,THETA) returns the Zernike functions of order N
%   and angular frequency M, evaluated at positions (R,THETA) on the
%   unit circle.  N is a vector of positive integers (including 0), and
%   M is a vector with the same number of elements as N.  Each element
%   k of M must be a positive integer, with possible values M(k) = -N(k)
%   to +N(k) in steps of 2.  R is a vector of numbers between 0 and 1,
%   and THETA is a vector of angles.  R and THETA must have the same
%   length.  The output Z is a matrix with one column for every (N,M)
%   pair, and one row for every (R,THETA) pair.

% REFERENCES 
% [1] R. J. Noll, "Zernike polynomials and atmospheric turbulence," 
%     J. Opt. Soc. Am. 66, 207-211 (1976).
% [2] C. Tian and S. Liu, "Demodulation of two-shot fringe patterns with 
%     random phase shifts by use of orthogonal polynomials and global 
%     optimization," Optics express 24, 3202-3215 (2016).


% Check and prepare the inputs:
% -----------------------------
if ( ~any(size(n)==1) ) || ( ~any(size(m)==1) )
    error('zernfun:NMvectors','N and M must be vectors.')
end

if length(n)~=length(m)
    error('zernfun:NMlength','N and M must be the same length.')
end

n = n(:);
m = m(:);
if any(mod(n-m,2))
    error('zernfun:NMmultiplesof2', ...
          'All N and M must differ by multiples of 2 (including 0).')
end

if any(m>n)
    error('zernfun:MlessthanN', ...
          'Each M must be less than or equal to its corresponding N.')
end

if any( r>1 | r<0 )
    error('zernfun:Rlessthan1','All R must be between 0 and 1.')
end

if ( ~any(size(r)==1) ) || ( ~any(size(theta)==1) )
    error('zernfun:RTHvector','R and THETA must be vectors.')
end

r = r(:);
theta = theta(:);
length_r = length(r);
if length_r~=length(theta)
    error('zernfun:RTHlength', ...
          'The number of R- and THETA-values must be equal.')
end

% Check normalization:
% --------------------
if nargin>=5 && ischar(nflag)
    isnorm = strcmpi(nflag,'norm');
%     if ~isnorm
%         error('zernfun:normalization','Unrecognized normalization flag.')
%     end
else
    isnorm = false;
end

order = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the Zernike Polynomials
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Determine the required powers of r:
% -----------------------------------
m_abs = abs(m);
%rpowers = [];
rpowers = zeros(1,0);
for j = 1:length(n)
    rpowers = [rpowers m_abs(j):2:n(j)];  % for s=0 power = n, s=(n-m)/2, power = m
end
rpowers = unique(rpowers);

% Pre-compute the values of r raised to the required powers,
% and compile them in a matrix:
% -----------------------------
if rpowers(1)==0
    %rpowern = arrayfun(@(p)r.^p,rpowers(2:end),'UniformOutput',false);
%     rpowern = cell(1, length(rpowers)-1);
%     for count = 1:length(rpowers)-1
%         rpowern{count} = r.^rpowers(count + 1);
%     end
%     rpowern = cat(2,rpowern{:});
%     rpowern = [ones(length_r,1) rpowern];    
    rpowern = ones(length_r, length(rpowers));
    for count = 2:length(rpowers)
        rpowern(:,count) = r.^rpowers(count);
    end
else
    %rpowern = arrayfun(@(p)r.^p,rpowers,'UniformOutput',false);
%     rpowern = cell(1, length(rpowers));
%     for count = 1:length(rpowers)
%         rpowern{count} = r.^rpowers(count);
%     end
%     rpowern = cat(2,rpowern{:});
    rpowern = zeros(length_r, length(rpowers));
    for count = 1:length(rpowers)
        rpowern(:,count) = r.^rpowers(count);
    end
end

% Compute the values of the polynomials:
% --------------------------------------
y = zeros(length_r,length(n));
z = y;
for j = 1:length(n)
    s = 0:(n(j)-m_abs(j))/2;
    pows = n(j):-2:m_abs(j);
    for k = length(s):-1:1
        p = (1-2*mod(s(k),2))* ...
                   prod(2:(n(j)-s(k)))/              ...
                   prod(2:s(k))/                     ...
                   prod(2:((n(j)-m_abs(j))/2-s(k)))/ ...
                   prod(2:((n(j)+m_abs(j))/2-s(k)));
        idx = (pows(k)==rpowers);
        y(:,j) = y(:,j) + p*rpowern(:,idx);
    end
    
    if isnorm
        y(:,j) = y(:,j)*sqrt( (1+(m(j)~=0))*(n(j)+1) );  % why divide pi
    end
    
    if order == 1  % the first way of sorting
        if ~mod(j,2) && m(j)~=0  % for even j
            z(:,j) = y(:,j).*cos(m_abs(j)'*theta);
        else if mod(j,2) && m(j)~=0 % for odd j
                z(:,j) = y(:,j).*sin(m_abs(j)'*theta);
            else
                z(:,j) = y(:,j);
            end
        end
    else
        if m(j)>=0  % for even j
            z(:,j) = y(:,j).*cos(m_abs(j)'*theta);
        else % for odd j
            z(:,j) = y(:,j).*sin(m_abs(j)'*theta);
        end
    end
    
end
return
% END: Compute the Zernike Polynomials
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%