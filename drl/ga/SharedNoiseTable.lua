require 'torch'

local sharednoise = torch.class('SharedNoiseTable')
function sharednoise:__init(args)
	local gen = torch.Generator()
	torch.manualSeed(gen, 123)
	print("Generating noise table")
	self.noise = torch.randn(gen, 1, 250000000)
	self.gen = gen
end

function sharednoise:get(i, dim)
	return self.noise[{1, {i,i+dim-1}}]
end

function sharednoise:sample_index(dim)
	return torch.random(1, self.noise:nElement() - dim + 1)
end
