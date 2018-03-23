# lua-nn
Lua based Neural Network

It's WIP.

Just do like:

local nn = require("nn")
local x = 3   --amount of layers

local y = 5   --default amount of neurons per layer

local y_inputs = 2  --nodes of input layer (the original 5 of y become 2 for the first layer, *not* a new layer with 2 as input)

local y_outputs = 1  --nodes of output layer

local activation_func = nn.func.sig
local act_f_drv = nn.func.asig
local lear_rate = 0.02

new_net = nn(x,y,y_inputs,y_outputs,activation_func, act_f_drv, lear_rate) --generate a new_net
new_net:addlayer(y) --becomes new output layer
new_net:build() --make weights for all connections

local train_pairs = {
	{{0,0},{0}},
	{{1,0},{1}},
	{{0,1},{1}},
	{{1,1},{0}},
  
  --{{<inputs>},{<expected_outputs>}},
}
local batches = 10
local batchsize = 10

for I = 1, batches
  for J = 1, batchsize do
    new_net:smart_train(train_pairs,0.6) --0.6  60% chance of picking a random pair, 40% picking the one that gives least error
    --smart_train is experimental and picks a random pair from train_pairs and keeps track of the error
  end
  new_net:applybatch() --apply weight updates
end
