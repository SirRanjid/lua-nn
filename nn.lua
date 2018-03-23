--lua nn SirRanjid (https://github.com/SirRanjid/lua-nn) 2018

--checking for nanas to randomly reinitiate a weight if the number gets too big/small
local nans = {["nan"] = true,["inf"] = true,["-inf"] = true}
local function isNaN( v ) return nans[tostring(v)] end

local e = 2.71828182845904523536028747135266249775724709369995
local tanh = math.tanh

local func = {
	tanh = tanh,
	atnh = function(x) return 1-tanh(x) end,

	sig = function(x) return 1/(1+(e^(-x))) end,
	asig = function(x) return (1/(1+(e^(-x)))) * (1-(1/(1+(e^(-x))))) end,

	relu = function(x) return math.max(0,x) end,
	arlu = function(x) return (x > 0 and 1) or 0 end, --#

	lin = function(x) return x end,
	alin = function(x) return 1 end,
}

local nn = {nets = {}, func = func}
nn.__index = nn

function nn:build() 	-- [L-1]->[W-1]->[L] => O
	local net = self.net
	local w = self.w
	local lr = {}
	for x,r in ipairs(net) do --for all layers
		if x > 1 then 			--if not input layer/1st layer
			w[x-1] = {}			--initiate table to store weigths coming to this layer

			for y,n in ipairs(r) do-- for all nodes in that layer
				w[x-1][y] = {}		--initiate table for all nodes
				for i,_ in ipairs(lr) do --for all nodes of prev layer
								-- {weight,update_value_sum,update_count,velocity}
					w[x-1][y][i] = {math.random(),0,0,0} --initite table with weight connecting current and prev node
				end
			end
		end
		lr = r
	end
end

function nn:addlayer(y,fn,afn)
	local x = #self.net+1
	self.net[x] = {b = math.random(-100000,100000)/100000}
	for J = 1,y do
		self.net[x][J] = {0,0,0,0,0,act = fn or self.act, drv = afn or self.drv}
	end
	self.out = self.net[x]
end

setmetatable(nn,{__call = function(self,x,y,yi,yo,act,drv,lr)

	local new = {net = {},w = {},c = false,lr = lr or 0.01,rt = {},rv = 0,act = act,drv = drv,feed={},fed={}}
	setmetatable(new,nn)

	x = math.max(2,x)

	for I = 1,x do
		new.net[I] = {b = math.random(-100000,100000)/100000}
		if I == 1 then --input layer
			for J = 1,yi do
							--= {output,net_input,bp_err,activation_function,act_func_derivative}
				new.net[I][J] = {0,0,0,act = act, drv = drv}
			end
		elseif I == x then --output layer
			for J = 1,yo do
				new.net[I][J] = {0,0,0,act = act, drv = drv}
			end
		else
			for J = 1,y do
				new.net[I][J] = {0,0,0,act = act, drv = drv}
			end
		end
	end

	new.inp = new.net[1]
	new.out = new.net[x]

	nn.nets[new] = new
	new:build()

	return new
end})

--propagate forward with tbl being an array of input values like {number input_1, ... , number input_yi}
--returns true
function nn:run(tbl)
	if not tbl or #tbl ~= #self.inp then return false end
	
	for i,n in ipairs(self.inp) do
		self.inp[i][1] = tbl[i] or 0
	end

	-- j[L-1]->[W-1]->i[L]->[W]->L[L+1]

	for x,r in ipairs(self.net) do --for all layers

		if x > 1 then --if not input/1st layer

			for y,n in ipairs(r) do --for each node
				n[2] = r.b --n[5] bias --add bias

				for i,ln in ipairs(lr) do --for all nodes in previous layer(l-1)
					
					--pt(ln,0,"ln")
					n[2] = n[2] + ln[1] * self.w[x-1][y][i][1] --add outputs of prev layer node * weight connecting both
					
				end
				n[1] = n.act(n[2]) --f(net-input_i) run sum through activation function
			end
		end
		lr = r --keep track of last layer
	end

	for k,v in pairs(self.feed) do
		v[1] = k[1] --nn.feed[nn.out[y1]] = nn2.inp[y2]
	end

	return true
end

--input tbl is an array of expected output values, has to match the length of output layer
--returns error value and tableof output deltas
function nn:getError(tbl)
	local err,ev = {}, 0
	for k,v in ipairs(self.out) do
		local kde = tbl[k]-v[1]
		local kde2 = ((kde > 0 and 1) or (kde < 0 and -1) or 0) * (kde^2)/2
		ev = ev + ((kde)^2)/2
		err[k] = kde2
	end
	self.rt = err
	self.rv = ev
	return ev,err
end

--input tbl is an array of expected output values, has to match the length of output layer (set nil/false if nn was fed by another nn (nn:feed(nn2,...)))
--second input is a radom jitter to eachs node error value
--returns error value and tableof output deltas
function nn:backprop(tbl,j)

	local err_v, err_t = 0, {}
	local xm = #self.net
	local Lr = {}
	if tbl then
		err_v, err_t = self:getError(tbl)
	else
		for i,v in ipairs(self.out) do
			err_t[i] = v[3]
		end
	end
	

	
	for x = xm,1,-1 do --for each layer backwards: get deltas
		local r = self.net[x] --layer
		local ds = 0 --deltasum
		if x == xm then --delta for output-layer --#
			for y,n in ipairs(r) do --for each neuron in a layer
				if j then
					n[3] = n.drv(n[2]) * err_t[y] + math.random(-j,j)--* (tbl[y]-n[1])
				else
					n[3] = n.drv(n[2]) * err_t[y]--* (tbl[y]-n[1])
				end
			end
		else --if x>1 then
			for i,n in ipairs(r) do --for each neuron in a layer
				local err_sum = 0
				for y,Ln in ipairs(Lr) do --apply weight update
					local w = self.w[x][y][i] --weight from current to prev layers node

					err_sum = err_sum + Ln[3] * w[1]
				end

				if j then
					n[3] = n.drv(n[2]) * err_sum + math.random(-j,j) --random jitter to nodes error value
				else
					n[3] = n.drv(n[2]) * err_sum--* (tbl[y]-n[1])
				end
			end
		end
		Lr = r
	end

	--lr = err_t
	for x,r in ipairs(self.net) do --update weights

		if x > 1 then --ignoring the input layer

			if x == 2 then
				for y,n in ipairs(r) do

					for i,ln in ipairs(lr) do
						local w = self.w[x-1][y][i] --weight from current to prev layers node
						w[2] = w[2] + (self.lr*n[3]*ln[1]) --W = W + dW; dW = learnrate * di (l) * net-input_j (l-1)
						w[3] = w[3] + 1

					end
				end
			else
				for y,n in ipairs(r) do

					for i,ln in ipairs(lr) do
						local w = self.w[x-1][y][i] --weight from current to prev layers node
						w[2] = w[2] + (self.lr*n[3]*ln[2]) --W = W + dW; dW = learnrate * di (l) * net-input_j (l-1)
						w[3] = w[3] + 1

					end
				end
			end
		end
		lr = r
	end

	for k,v in pairs(self.fed) do --feed back to net that fed this one on the forward pass
		v[1] = k[1]	--nn2.fed[nn2.inp[y2]] = nn.out[y1]
		v[2] = k[2]
		v[3] = k[3]
	end

	return err_v, err_t
end

--apply all weightupdates generated by nn:backprop
function nn:applybatch(j) --jitter to weights
	local lr = {}
	for x,r in ipairs(self.net) do
		if x > 1 then

			for y,n in ipairs(r) do

				for i,ln in ipairs(lr) do --apply weight update
					local w = self.w[x-1][y][i]

					local add = w[2] /w[3]

					if j then add = add + math.random(-j,j) end

					w[1] = w[1] + add + w[4]
					w[4] = w[4] + add * self.lr--velovity

					if isNaN(w[1]) then w[1] = math.random() end
					w[2] = 0
					w[3] = 0
				end
			end
		end
		lr = r
	end
end

--input inp is an array of input values
--input exp_out is an array of expected output values
--input j is the jitter that forwards to nn:backprop
--returns what backprop returns of false
function nn:train(inp,exp_out,j)
	if not j then
		if not inp then return false end
		j = exp_out
		exp_out = inp[2]
		inp = inp[1]

	end

	if self:run(inp) then
		return self:backprop(exp_out,j)
	end
	return false
end

--links the outputs of nn ranging from ([n1y1 >= 1] to [n1y2 <= nn_yo]) to the inputs of nn2 ranging from ([n2y1 >= 1] to [n2y2 <= nn2_yo])
--so if you nn:run the outputs from nn get fed into nn2s inputs and vice versa for nn2:backprop
function nn:feed(nn2,n1y1,n1y2,n2y1,n2y2)
	if n1y2 - n1y1 == n2y2 - n2y1 then
		--nn2.fed = {}
		for I = 0, n1y2-n1y1-1 do
			local y1 = n1y1 + I
			local y2 = n2y1 + I

			--nn2.inp[y2][1] = nn.out[y1][1]

			nn.feed[nn.out[y1]] = nn2.inp[y2]
			nn2.fed[nn2.inp[y2]] = nn.out[y1]
			--table.insert(nn2.fed,{nn,n1y1,n1y2,n2y1,n2y2})
		end
	end
end

function nn:pairpicker(train_pairs,r)
	if not self.paircache then
		self.paircache = {}

		for k,v in ipairs(train_pairs) do
			self.paircache[v] = {0,0,0} -- = {lerp?,error value}
		end
	end

	if math.random(100)/100 < (r or 50) then
		local ret = train_pairs[math.random(#train_pairs)]
		return ret
	else
		local m,mv = {},-1
		for v,k in pairs(self.paircache) do
			if math.abs(k[2]) < mv or mv == -1 then
				m = v
				mv = math.abs(k[2])
			end
		end
		return m
	end

	--return m --]]
end

function nn:smart_train(train_pairs,p,j)
	local pair = self:pairpicker(train_pairs,p)
	local pc = self:getPairCache(pair)
	if pc[1] <= pc[2] then
		local err,err_t = self:train(pair)
		self:pairupdate(pair,err)
	else
		local err,err_t = self:train(pair,pc[3]*self.lr^3 * (j or 0))
		self:pairupdate(pair,err)
	end
end


function nn:getPairCache(pair)
	return self.paircache[pair]
end



function nn:pairupdate(pair,err)
	--if not err then return end
	--if not self.paircache[pair] then self.paircache[pair] = {0,0,0} end
	self.paircache[pair][3] = self.paircache[pair][3] + self.paircache[pair][1] - self.paircache[pair][2]
	self.paircache[pair][2] = self.paircache[pair][1]--self.paircache[pair][1]+(err-self.paircache[pair][1])*self.lr --lerp?
	self.paircache[pair][1] = err
	return self.paircache
end

return nn