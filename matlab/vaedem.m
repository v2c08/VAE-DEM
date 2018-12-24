clear all
clear classes
close all hidden

% Dynamic expectation maxmisation (Variational Laplacian filtering)
% FORMAT DEM   = spm_DEM(DEM)
%
% DEM.M  - hierarchical model
% DEM.Y  - response variable, output or data - Clss labels
% DEM.U  - inputs - Images

nT = 3;
n_latents = 8;
n_activations = 9;
n_modes = 2;
imLen = 64*64*3;

shape_m = 1;
colour_m = 2;

label.factor         = {'shape','colour'};
label.name{shape_m}  = {'Cube','Icosphere','Cylinder', 'Cone', 'Torus'};
label.name{colour_m} = {'r','g','b','w'};

label.modality          = {'shape','colour'};
label.outcome{shape_m}  = {'Cube','Icosphere','Cylinder', 'Cone', 'Torus'};
label.outcome{colour_m} = {'r','g','b','w'};

% Outcome Modalities / Discrete Generative Factors
shapes = containers.Map;
shapes('Cube')      = [0, 1, 0, 0, 0, 0, 0, 0, 0];
shapes('Icosphere') = [0, 0, 0, 1, 0, 0, 0, 0, 0];
shapes('Cylinder')  = [0, 0, 1, 0, 0, 0, 0, 0, 0];
shapes('Cone')      = [1, 0, 0, 0, 0, 0, 0, 0, 0];
shapes('Torus')     = [0, 0, 0, 0, 1, 0, 0, 0, 0];

colours = containers.Map;
colours('r')   = [0, 0, 0, 0, 0, 0, 0, 1, 0];
colours('g') = [0, 0, 0, 0, 0, 0, 1, 0, 0];
colours('b')  = [0, 0, 0, 0, 0, 1, 0, 0, 0];
colours('w') = [0, 0, 0, 0, 0, 0, 0, 0, 1];

% Get series of images
f = dir('data/train/Cube/w/s0/c/x/*.png');
files = {f.name};

U = zeros(nT,imLen);
Y = zeros(nT,n_activations);

for k = 1:nT
   U(k,:) = spm_vec(im2double(imread(files{k})));
   Y(k,find(shapes('Cube'))) = 1;
   Y(k,find(colours('w')))   = 1;
end

% true causes for every combination of discrete states
%--------------------------------------------------------------------------
for i = 1:length(label.name{shape_m})
    for j = 1:length(label.name{colour_m})
        c = [find(contains(label.outcome{shape_m}, label.name{shape_m}(i))); 
             find(contains(label.outcome{colour_m}, label.name{colour_m}(j)))];
        demi.C{i,j} = c*ones(1,nT);
        demi.U{i,j} = c*ones(1,nT);
    end
end

% Level 1
% Recognition Model
M(1).g = 'gx_encoder_m';
M(1).f = @(x,v,P)x;

M(1).m = n_activations;   % number of inputs v(i + 1);
M(1).n = n_latents; % number of states x(i)
M(1).l = imLen;   % number of output v(i)

% Generative Model
G(1).g = 'gx_encoder_g';
G(1).f = @(x,v,a,P)x+a;

G(1).m = n_activations; % number of inputs v(i + 1);
G(1).n = n_latents; % number of states x(i);
G(1).l = imLen; % number of output v(i);
G(1).k = n_latents;
G(1).U = exp(4);


% Level 2
% Recognition Model
M(2).g = 'gx_classifier_m';
M(2).f = @(x,v,P)x;

M(2).m = n_modes;
M(2).n = n_latents;
M(2).l = n_activations;

M(2).V  = exp(4);  
M(2).W  = exp(10); 

% Generative Process
G(2).g = 'gx_classifier_g';
G(2).f = @(x,v,a,P)x;

G(2).m = n_modes;
G(2).n = n_latents;
G(2).l = n_activations;
G(2).k  = 0;

M(1).E.n = 2; % embedding order

DEM.M  = M;
DEM.G  = G;
DEM.db = 1;


o(shape_m)  = find(contains(label.outcome{shape_m}, 'Cube'));
o(colour_m) = find(contains(label.outcome{colour_m}, 'w'));

O{shape_m}  = spm_softmax(sparse(1:length(shapes), 1, 1, length(shapes),  1));
O{colour_m} = spm_softmax(sparse(1:length(colours),1, 4, length(colours), 1));

DEM    = spm_MDP_DEM(DEM,demi,O,o);

%DEM.U = Im;

% prior beliefs about initial states
%--------------------------------------------------------------------------
D{1} = [1;0;0;0;0;0;0;0;0]; % what: {'Cube','Icosphere','Cylinder', 'Cone', 'Torus'}
D{2} = [0;0;1;0;0;0;0;0;0];   % colour:    {'r','g','b','w'}

% Outcome Modalities 
A{shape_m}  = zeros(5,n_activations, n_activations) + 1/n_activations;
A{colour_m} = zeros(4,n_activations, n_activations) + 1/n_activations;
% Discrete Generative Factors
B{shape_m}  = zeros(n_activations) + 1/n_activations;
B{colour_m} = zeros(n_activations) + 1/n_activations;
% Preferences
C{shape_m}  = zeros(5,1);
C{colour_m} = zeros(4, 1);

% MDP Structure
%--------------------------------------------------------------------------
mdp.T     = 2;                      % number of updates
mdp.A     = A;                      % observation model
mdp.a     = A;                      % A prior
mdp.B     = B;                      % transition probabilities
mdp.b     = b;                      % B prior
mdp.D     = D;                      % prior over initial states
mdp.DEM   = DEM;
mdp.demi  = demi;
MDP = spm_MDP_check(mdp);

MDP  = spm_MDP_VB_X(MDP);


