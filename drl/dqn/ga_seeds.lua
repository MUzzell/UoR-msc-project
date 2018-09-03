
function ga_set_network_from_seeds(net, filename, n_params)

	local ga_seeds = GASeeds{n_params=n_params}

	idx, seeds = ga_seeds:load_file(filename)

	net = ga_seeds:set_weights(net, idx, seeds)

	return net

end

function ga_get_network_weights_from_seeds(filename, n_params)
	local ga_seeds = GASeeds{n_params=n_params}

	idx, seeds = ga_seeds:load_file(filename)

	return ga_seeds:get_weights(idx, seeds)
end

function ga_get_network_weights_from_weights(filename, n_params)
	print("Loading file: "..filename)

	theta = torch.Tensor(n_params)
	local iter = io.lines(filename)
	local i = 1
	for line in iter do
		theta[i] = tonumber(line)
		i = i + 1
	end

	if i - 1 ~= n_params then
		error(string.format("Invalid weights file, %d / %d weights",
							i, n_params))
	end

	local ga_seeds = GASeeds{n_params=n_params}

	return ga_seeds:load_weights(theta)
end


local gas = torch.class('GASeeds')
function gas:__init(args)
	--self.noise = SharedNoisetable{}
	self.n_params = tonumber(args.n_params)
end

function gas:load_file(filename)

	print("Loading file: "..filename)

	seeds = {}
	local iter = io.lines(filename)
	local idx = iter()
	if idx == nil then
		error("File "..filename.." is empty")
	end
	idx = tonumber(idx)
	for line in iter do
		seed = string.match(line, "%d+")
		power = string.match(line, "%d+%.%d+")
	    seeds[#seeds + 1] = {
	    	idx=tonumber(seed),
	    	power=tonumber(power)
	    }
	end
	return idx, seeds
end

function gas:get_weights(idx, seeds)

	local weights = {}
	local s_idx = 1
	print("Building theta")
	local theta = self.noise:get(idx, self.n_params)
	print("Setting scalars")
	--conv1.w scalar
	theta[{{s_idx, s_idx + 8191}}] = theta[{{s_idx, s_idx + 8191}}] * 0.0625
	s_idx = s_idx + 8192
	theta[{{s_idx, s_idx + 31}}] = theta[{{s_idx, s_idx + 31}}] * 0
	s_idx = s_idx + 32

	--conv2.w scalar
	theta[{{s_idx, s_idx + 32767}}] = theta[{{s_idx, s_idx + 32767}}] * 0.04419417
	s_idx = s_idx + 32768
	theta[{{s_idx, s_idx + 63}}] = theta[{{s_idx, s_idx + 63}}] * 0
	s_idx = s_idx + 64

	--conv3.w scalar
	theta[{{s_idx, s_idx + 36863}}] = theta[{{s_idx, s_idx + 36863}}] * 0.04166667
	s_idx = s_idx + 36864
	theta[{{s_idx, s_idx + 63}}] = theta[{{s_idx, s_idx + 63}}] * 0
	s_idx = s_idx + 64

	--fc scalar
	theta[{{s_idx, s_idx + 3964927}}] = theta[{{s_idx, s_idx + 3964927}}] * 0.01136364
	s_idx = s_idx + 3964928
	theta[{{s_idx, s_idx + 511}}] = theta[{{s_idx, s_idx + 511}}] * 0
	s_idx = s_idx + 512

	--out scalar
	theta[{{s_idx, s_idx + 3071}}] = theta[{{s_idx, s_idx + 3071}}] * 0.00441942
	s_idx = s_idx + 3072
	theta[{{s_idx, s_idx + 5}}] = theta[{{s_idx, s_idx + 5}}] * 0
	s_idx = s_idx + 6

	if s_idx-1 ~= self.n_params then
		error(s_idx.." != "..self.n_params)
	end

	print("Appling mutations")
	for i=1,#seeds do
		seed = seeds[i]
		theta = self:compute_mutation(theta, seed.idx, seed.power)
	end

	return load_weights(theta)
end

function gas:load_weights(theta)
	local w_idx = 1

	local weights = {}

	local weight = theta[{{w_idx, w_idx + 8191}}]:resize(8, 8, 4, 32)
	w_idx = w_idx + 8192
	local bias = theta[{{w_idx, w_idx + 31}}]
	w_idx = w_idx + 32

	--weight = weight:index(1, torch.linspace(8192, 1, 8192):long())
	--weight = nn.Reshape(32, 4, 8, 8):forward(weight)
	weight = weight:permute(4,3,1,2)
	print(weight:size())

	--bias = bias:index(1, torch.linspace(32, 1, 32):long())
	--bias = bias:permute(2, 1)

	--conv1
	weights[#weights + 1] = {
		weights = weight:clone(),
		bias = bias:clone()
	}

	weight = theta[{{w_idx, w_idx + 32767}}]:resize(4, 4, 32, 64)
	w_idx = w_idx + 32768
	bias = theta[{{w_idx, w_idx + 63}}]
	w_idx = w_idx + 64

	--weight = weight:index(1, torch.linspace(32768, 1, 32768):long())
	--weight = nn.Reshape(64, 32, 4, 4):forward(weight)
	weight = weight:permute(4,3,1,2)

	--bias = bias:index(1, torch.linspace(64, 1, 64):long())
	--bias = bias:permute(2, 1)

	--conv2
	weights[#weights + 1] = {
		weights = weight:clone(),
		bias = bias:clone()
	}

	weight = theta[{{w_idx, w_idx + 36863}}]:resize(3, 3, 64, 64)
	w_idx = w_idx + 36864
	bias = theta[{{w_idx, w_idx + 63}}]
	w_idx = w_idx + 64

	--weight = weight:index(1, torch.linspace(36864, 1, 36864):long())
	--weight = nn.Reshape(64, 64, 3, 3):forward(weight)
	weight = weight:permute(4,3,1,2)

	--bias = bias:index(1, torch.linspace(64, 1, 64):long())

	--conv3
	weights[#weights + 1] = {
		weights = weight:clone(),
		bias = bias:clone()
	}

	weight = theta[{{w_idx, w_idx + 3964927}}]:resize(7744, 512)
	w_idx = w_idx + 3964928
	bias = theta[{{w_idx, w_idx + 511}}]
	w_idx = w_idx + 512

	--weight = weight:index(1, torch.linspace(3964928, 1, 3964928):long())
	--weight = nn.Reshape(512, 7744):forward(weight)
	weight = weight:permute(2, 1)

	--bias = bias:index(1, torch.linspace(512, 1, 512):long())

	--fc
	weights[#weights + 1] = {
		weights = weight:clone(),
		bias = bias:clone()
	}

	weight = theta[{{w_idx, w_idx + 3071}}]:resize(512, 6)
	w_idx = w_idx + 3072
	bias = theta[{{w_idx, w_idx + 5}}]
	w_idx = w_idx + 6

	--weight = weight:index(1, torch.linspace(3072, 1, 3072):long())
	--weight = nn.Reshape(6, 512):forward(weight)
	weight = weight:permute(2, 1)

	--bias = bias:index(1, torch.linspace(6, 1, 6):long())

	--out
	weights[#weights + 1] = {
		weights = weight:clone(),
		bias = bias:clone()
	}

	if w_idx-1 ~= self.n_params then
		error(w_idx.." != "..self.n_params)
	end

	return weights
end

function gas:set_weights(net, idx, seeds)
	local theta = self.noise:get(idx, self.n_params)

	for i=1, #seeds do
		local seed = seeds[i]
		theta = self:compute_mutation(theta, seed.idx, seed.power)
	end

	w_idx = 1
	conv1 = net:get(2)
	conv1.weights = theta[{{w_idx, w_idx + 8191}}]:resize(torch.LongStorage({32, 4, 8, 8}))
	w_idx = w_idx + 8192
	conv1.bias = theta[{{w_idx, w_idx + 31}}]
	w_idx = w_idx + 32

	print(w_idx)

	conv2 = net:get(4)
	conv2.weights = theta[{{w_idx, w_idx + 32767}}]:resize(torch.LongStorage({64, 32, 4, 4}))
	w_idx = w_idx + 32768
	conv2.bias = theta[{{w_idx, w_idx + 63}}]
	w_idx = w_idx + 64

	print(w_idx)


	conv3 = net:get(6)
	conv3.weights = theta[{{w_idx, w_idx + 36863}}]
	w_idx = w_idx + 36864
	conv3.bias = theta[{{w_idx, w_idx + 63}}]
	w_idx = w_idx + 64


	print(w_idx)

	fc = net:get(9)
	fc.weights = theta[{{w_idx, w_idx + 3964927}}]
	w_idx = w_idx + 3964928
	fc.bias = theta[{{w_idx, w_idx + 511}}]
	w_idx = w_idx + 512

	print(w_idx)

	out = net:get(11)
	out.weights = theta[{{w_idx, w_idx + 3071}}]
	w_idx = w_idx + 3072
	out.bias = theta[{{w_idx, w_idx + 5}}]
	w_idx = w_idx + 6

	print(w_idx)

	if w_idx-1 ~= self.n_params then
		error(w_idx.." != "..n_params)
	end

	--[[
	(1): nn.Reshape(4x84x84)
	(2): nn.SpatialConvolution(4 -> 32, 8x8, 4,4, 2,2)
		w: 8192 (32, 4, 8, 8)
		b: 32
	(3): nn.Rectifier
	(4): nn.SpatialConvolution(32 -> 64, 4x4, 2,2, 2,2)
		w: 32768 (64, 32, 4, 4)
		b: 64
	(5): nn.Rectifier
	(6): nn.SpatialConvolution(64 -> 64, 3x3, 1,1, 1,1)
		w: 36864 (64, 64, 3, 3)
		b: 64
	(7): nn.Rectifier
	(8): nn.Reshape(7744)
	(9): nn.Linear(7744 -> 512)
		w: 3964928 (512, 7744)
		b: 512
	(10): nn.Rectifier
	(11): nn.Linear(512 -> 6)
		w: 3072 (6, 512)
		b: 6
	TOTAL = 4046502
	]]

	return net

end

function gas:compute_mutation(theta, idx, power)
	return theta + power * self.noise:get(idx, self.n_params)
end