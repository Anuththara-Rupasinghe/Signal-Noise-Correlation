function b = redblue(m)
%REDBLUE   Bilinear red/blue color map.
%
%   REDBLUE(M) returns an M-by-3 matrix containing the colormap.
%   REDBLUE, by itself, is the same length as the current colormap.
%
%   For example, to reset the colormap of the current figure:
%
%             colormap(redblue)
%
%
%   See also BROWNBLUE, YARG, YARGPRINT, GRAYPRINT.
%   See also HSV, HOT, COOL, BONE, COPPER, PINK, FLAG, 
%   COLORMAP, RGBPLOT.

if nargin < 1, m = size(get(gcf,'colormap'),1); end
top = floor(m/2);
bot = m-top;

btop = [1.0*ones(top,1),([1:top]'/top).^1.7,ones(top,1)];   % 1.0 hue good red
bbot = [0.7*ones(bot,1),(1-[1:bot]'/bot).^1.7,ones(bot,1)]; % 0.7 hue good blue
% btop = [1.0*ones(top,1),([1:top]'/top).^1.3,ones(top,1)];   % 1.0 hue good red
% bbot = [0.7*ones(bot,1),(1-[1:bot]'/bot).^1.3,ones(bot,1)]; % 0.7 hue good blue
b=hsv2rgb([bbot;btop]);

%btop = interp1([0;1],[1,0,0;1,1,1],([1:top]'/top).^0.6);
%bbot = interp1([0;1],[0,0,1;1,1,1],([1:bot]'/bot).^0.6);
%b = [bbot; flipud(btop)];
