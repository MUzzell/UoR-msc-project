
require 'torch'
require 'nn'
require 'nngraph'
require 'image'
require 'Scale'

local num_params = 4046502


function compute_weights(noise, scales, seeds)

	local idx = seeds[1]
	local theta = torch.cmul(noise:get(idx, num_params), scales)

	assert(theta:nElement() == num_params)

	for i=2,#seeds do
		local idx = seeds[i].idx
		local power = seeds[i].power
		theta:add(theta, torch.mul(noise:get(idx, num_params), power))
	end

	assert(theta:nElement() == num_params)
	return theta
end

function mutate(parent, noise, power)
	local parent_theta = parent[1]
	idx = noise:sample_index(num_params)
	parent[#parent+1] = {idx=idx, power=power}
	return parent
end

function randomise(noise)
	local seeds = {}
	seeds[1] = noise:sample_index(num_params)
	return seeds
end


function do_step(screen, network, gpu)
	preproc = nn.Scale(84, 84, true)
	state = preproc:forward(screen:float()):clone():resize(1, 4, 84, 84)

	if gpu >= 0 then
		state = state:cuda()
	end

	q = network:forward(state):float():squeeze()
	local maxq = q[1]
    local besta = {1}

    -- Evaluate all other actions (with random tie-breaking)
    for a = 2, 6 do
        if q[a] > maxq then
            besta = { a }
            maxq = q[a]
        elseif q[a] == maxq then
            besta[#besta+1] = a
        end
    end
    local r = torch.random(1, #besta)

    return besta[r]
end

function set_model_weights(model, theta, gpu)

	local w_idx = 1

	local n_idx = 2

	if gpu >= 0 then
		n_idx = 3
	end

	local weight = theta[{{w_idx, w_idx + 8191}}]:resize(32, 4, 8, 8)
	w_idx = w_idx + 8192
	local bias = theta[{{w_idx, w_idx + 31}}]
	w_idx = w_idx + 32

	--conv1
	model:get(n_idx).weights = weight:clone()
	model:get(n_idx).bias = bias:clone()

	n_idx = n_idx + 2

	weight = theta[{{w_idx, w_idx + 32767}}]:resize(64, 32, 4, 4)
	w_idx = w_idx + 32768
	bias = theta[{{w_idx, w_idx + 63}}]
	w_idx = w_idx + 64

	--conv2
	model:get(n_idx).weights = weight:clone()
	model:get(n_idx).bias = bias:clone()

	n_idx = n_idx + 2

	weight = theta[{{w_idx, w_idx + 36863}}]:resize(64, 64, 3, 3)
	w_idx = w_idx + 36864
	bias = theta[{{w_idx, w_idx + 63}}]
	w_idx = w_idx + 64

	--conv3
	model:get(n_idx).weights = weight:clone()
	model:get(n_idx).bias = bias:clone()

	n_idx = n_idx + 2

	weight = theta[{{w_idx, w_idx + 3964927}}]:resize(512, 7744)
	w_idx = w_idx + 3964928
	bias = theta[{{w_idx, w_idx + 511}}]
	w_idx = w_idx + 512

	--fc
	model:get(n_idx).weights = weight:clone()
	model:get(n_idx).bias = bias:clone()

	n_idx = n_idx + 2

	weight = theta[{{w_idx, w_idx + 3071}}]:resize(6, 512)
	w_idx = w_idx + 3072
	bias = theta[{{w_idx, w_idx + 5}}]
	w_idx = w_idx + 6

	--out
	model:get(n_idx).weights = weight:clone()
	model:get(n_idx).bias = bias:clone()

	if w_idx-1 ~= num_params then
		error(w_idx.." != "..num_params)
	end

	if gpu >= 0 then
		model:cuda()
	end

	return model

end

function build_model(noise, model, theta, seeds, gpu)

	for i=1, #seeds do
		seed = seeds[i]
		theta = theta + seed.mutation_power * noise:get(seed.idx, num_params)
	end

	return set_model_weights(model, theta, gpu)

end

function run_worker(model, seeds, theta, gameEnv, gameActions, n_ep, gpu)

	model = set_model_weights(model, theta, gpu)

	ep = 1
	result = {seeds = seeds, episodes = torch.Tensor(n_ep, 2)}
	while ep <= n_ep do
		screen, reward, terminal = gameEnv:newGame()
		ep_reward = 0
		ep_step = 0
		while not terminal do
			local action_idx = do_step(screen, model, gpu)

			screen, reward, terminal = gameEnv:step(gameActions[action_idx])

			ep_reward = ep_reward + reward
			ep_step = ep_step + 1
		end
		--print(string.format("Ep reward: %d, steps: %d", ep_reward, ep_step))
		result.episodes[{ep, 1}] = ep_reward
		result.episodes[{ep, 2}] = ep_step
		ep = ep + 1
	end
	collectgarbage()
	return result

end