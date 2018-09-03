
--[[local cv = require 'cv'
require 'cv.videoio'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'
]]

require "init"
require "paths"
require "SharedNoiseTable"
require "ga_worker"

local threads = require 'threads'


local cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:text()
cmd:text('Train GA in Environment:')
cmd:option('-framework', 'alewrap', 'framework to use')
cmd:option('-name', 'snapshot_torch_ga', 'name of snapshots')
cmd:option('-game_path', '', 'Path to game rom')
cmd:option('-data_path', '/tmp', 'Path to save logs & snapshots')
cmd:option('-snapshot_count', '5', '# of snapshots to keep')
cmd:option('-env', '', 'name of game to play')
cmd:option('-gpu', -1, 'gpu flag')

cmd:option('-netfile', '', 'iunno')
cmd:option('-mutation_power', 0.002)

cmd:option('-population', 100, 'Population size')
cmd:option('-pop_ep', 1, 'number of episodes to run per pop member')
cmd:option('-steps', 4*10^6, 'Max steps to run for')
cmd:option('-pop_selection_size', 20, '# of pop members to select')
cmd:option('-validation_ep', 30, '# of episodes for elite selection')
cmd:option('-elite_test_ep', 200, '# of episodes for testing elite(s)')
cmd:option('-elite_size', 1, '# of elites')
cmd:option('-threads', 4, '# of threads to use')

cmd:text()

local opt = cmd:parse(arg)

function get_network(model_file, args)

	local model = nil
	local scales = nil

	local msg, err = pcall(require, model_file)
    if not msg then
        -- try to load saved agent
        local err_msg, exp = pcall(torch.load, model_file)
        if not err_msg then
            error("Could not find network file "..model_file)
        end
        model = exp.model
    else
        print('Creating Agent Network from ' .. model_file)
        model = err
        model, scales = model(args)
    end

	if args.gpu and args.gpu > -1 then
		model:cuda()
	else
		model:float()
	end

	return model, scales
end

function get_snapshots(data_path)

	snapshots = {}

	for file in paths.iterfiles(data_path) do
		if paths.extname(file) == 't7' then
			snapshots[#snapshots+1] = file
		end
	end

	return snapshots

end

function save_snapshot(snapshot, data_path)

	paths.files()

end

function get_next_snapshot_name(data_path, snapshot_idx, name)

	return paths.concat(data_path, name.."_"..snapshot_idx..".t7")
end

function truncate_pop(results, pop_select)

	assert(#results >= pop_select,
		   string.format("#results (%d) < pop_select (%d)",
		   				 #results, pop_select))

	local ep_results = torch.Tensor(#results)

	for i=1,#results do
		episodes = results[i].episodes
		ep_results[i] = torch.mean(episodes:select(2, 1))
	end

	ep_results, idx = torch.sort(ep_results, true)

	local truncated_results = {}

	for i=1,pop_select do
		truncated_results[#truncated_results+1] = results[idx[i]]
	end

	return truncated_results, idx
end

function get_pop_stats(results)

	local steps = 0
	local rew_max = 0
	local ep_results = torch.Tensor(#results)

	for i=1,#results do
		episodes = results[i].episodes
		rew_max = math.max(rew_max, torch.max(episodes:select(2, 1)))
		if #results == 1 then
			ep_results = episodes:select(2,1)
		else
			ep_results[i] = torch.mean(episodes:select(2, 1))
		end
		steps = steps + torch.sum(episodes:select(2, 2))
	end

	return rew_max, torch.max(ep_results), torch.mean(ep_results),
		   torch.std(ep_results), steps

end

function get_fitness(results)

	local fitness = torch.Tensor(#results, 2)

	for i=1,#results do
		episodes = results[i].episodes:select(2, 1)
		fitness[{i, 1}] = torch.mean(episodes)
		fitness[{i, 2}] = torch.std(episodes)
	end

	return fitness, torch.max(fitness:select(2, 1)), torch.max(fitness:select(2, 2))

end

function print_fitness(results)
	local fitness, max_avg, max_std = get_fitness(results)

	local fit_s = string.format("FIT: %d %d:", max_avg, max_std)
	for i=1,fitness:size()[1] do
		fit_s = fit_s..string.format(" %d,%d", fitness[{i, 1}], fitness[{i, 2}])
	end

	print(fit_s)
end

function run_loop(opt)

	--preproc = get_network("net_downsample_2x_full_y", opt)

	local elite = {}
	local population = {}
	local step = 0
	local iteration = 0
	local model = nil


	snapshots = get_snapshots(opt.data_path)
	local snapshot_count = 0

	model, scales = get_network("ga_convnet", opt)

	if next(snapshots) ~= nil then
		snapshot_file = snapshots[1]
		snapshot_file = paths.concat(opt.data_path, snapshot_file)
		print("Loading snapshot: "..snapshot_file)

		snapshot = torch.load(snapshot_file)

		model = snapshot.model
		elite = snapshot.elite_seeds
		population = snapshot.pop_seeds
		step = snapshot.step
		iteration = snapshot.iteration

		_, scales = get_network("ga_convnet", opt)

		snapshot_count = #snapshots

		print(string.format("Loaded iteration %d, %d / %d steps",
							iteration, step, opt.steps))
	end

	local noise = SharedNoiseTable{}

	local pool = threads.Threads(
		opt.threads,
		function(threadid)
			require 'nn'
			require "SharedNoiseTable"
			require "Scale"
			require 'ga_worker'
			require 'init'
			torchSetup(opt)

	    	gameEnv, gameActions = buildGameEnv(opt)
			print("Starting new thread "..threadid)
		end
	)

	while step < opt.steps do
		iteration = iteration + 1
		print(string.format("Iteration: %d, %d/%d",
							iteration, step, opt.steps))
		local iter_time = sys.clock()
	    args = {
	    	opt = opt,
	    	model = model,
	    	scales = scales,
	    	population = population
		}

		local results = {}
		local count = 0
		print("Generate Population")
		local eval_time = sys.clock()
		--POP Evaluation
		for i=1,opt.population do
			local seeds = randomise(noise)
			if #population > 0 then
				parent = population[torch.random(1, #population)]
				seeds = mutate({table.unpack(parent)}, noise, opt.mutation_power)
				assert(#parent == #seeds - 1)
			end
			local theta = compute_weights(noise, scales, seeds)
			pool:addjob(
				function(args, seeds, theta)
					return run_worker(args.model, seeds, theta, gameEnv,
									  gameActions, args.opt.pop_ep, args.opt.gpu)
				end,
				function(result)
					results[i] = result
					count = count + 1
					if count % 10 == 0 then
						print(string.format("%d / %d (%d%%)", count, opt.population,
											math.floor((count / opt.population)*100)))
					end
				end,
				args, seeds, theta
			)
		end

		pool:synchronize()
		collectgarbage()
		eval_time = sys.clock() - eval_time

		assert(#results >= opt.population)

		local rew_max, rew_max_avg, rew_mean, rew_std, steps = get_pop_stats(results)

		local top_pop_results, idx = truncate_pop({table.unpack(results)}, opt.pop_selection_size)

		print(string.format("Population Results: Max: %d, Mean: %d, Std: %d",
							rew_max, rew_mean, rew_std))

		print(string.format("Population Evaluation: %d / %ds (%d fps)",
							steps, eval_time, steps / eval_time))

		idx_s = idx[1]
		local count = idx:nElement()
		if count > opt.pop_selection_size then
			count = opt.pop_selection_size
		end
		for i=2,count do
			idx_s = idx_s..","..idx[i]
		end


		print(string.format("Selected Pop Idx: %s", idx_s))

		step = step + steps

		for i=1,#elite do
			top_pop_results[#top_pop_results+1] = {seeds = elite[i]}
		end

		assert(#top_pop_results == opt.pop_selection_size + #elite)

		local truncated_results = {}

		print("Validate Population")
		eval_time = sys.clock()
		--Truncated Pop, Elite Selection
		for i=1,#top_pop_results do
			local seeds = top_pop_results[i].seeds
			local theta = compute_weights(noise, scales, seeds)
			pool:addjob(
				function(args, seeds, theta)
					return run_worker(args.model, seeds, theta, gameEnv,
									  gameActions, args.opt.validation_ep, args.opt.gpu)
				end,
				function(result)
					truncated_results[i] = result
				end,
				args, seeds, theta
			)
		end

		pool:synchronize()
		collectgarbage()
		eval_time = sys.clock() - eval_time

		assert(#truncated_results == #top_pop_results)
		assert(truncated_results[1].episodes:nElement() == opt.validation_ep * 2)

		rew_max, rew_max_avg, rew_mean, rew_std, steps = get_pop_stats(truncated_results)

		print(string.format("Validation Results: Max: %d, Max Avg: %d, Mean: %d, Avg Std: %d",
							rew_max, rew_max_avg, rew_mean, rew_std))

		print_fitness(truncated_results)


		local top_val_results, idx = truncate_pop({table.unpack(truncated_results)}, opt.elite_size)

		idx_s = idx[1]
		for i=2,opt.elite_size do
			idx_s = idx_s..","..idx[i]
		end

		print(string.format("Elite Idx: %s", idx_s))

		print(string.format("Validation Evaluation: %d / %ds (%d fps)",
							steps, eval_time, steps / eval_time))

		assert(#top_val_results == opt.elite_size)

		local elite_results = {}

		print("Evaluate Elite")
		eval_time = sys.clock()
		--Elite Evaluation
		for i=1,#top_val_results do
			local seeds = top_val_results[i].seeds
			local theta = compute_weights(noise, scales, seeds)
			pool:addjob(
				function(args, seeds, theta)
					return run_worker(args.model, seeds, theta, gameEnv,
									  gameActions, args.opt.elite_test_ep, args.opt.gpu)
				end,
				function(result)
					elite_results[#elite_results+1] = result
				end,
				args, seeds, theta
			)
		end

		pool:synchronize()
		collectgarbage()
		eval_time = sys.clock() - eval_time

		assert(#elite_results >= #top_val_results)
		assert(elite_results[1].episodes:nElement() == opt.elite_test_ep * 2)

		rew_max, _, rew_mean, rew_std, steps = get_pop_stats(elite_results)

		print(string.format("Elite Results: Max: %d, Mean: %d, Std: %d",
							rew_max, rew_mean, rew_std))

		print(string.format("Elite Evaluation: %d / %ds (%d fps)",
							steps, eval_time, steps / eval_time))

		snapshot_filename = get_next_snapshot_name(opt.data_path, iteration,
												   opt.name)

		population = {}
		for i=1,#top_pop_results do
			population[#population+1] = top_pop_results[i].seeds
		end
		elite = {}
		for i=1,#elite_results do
			elite[#elite+1] = elite_results[i].seeds
		end

		torch.save(snapshot_filename,
				   {model = model,
					pop_seeds = population,
					elite_seeds = elite,
					step = step,
					iteration = iteration
					})
		print("Saved snapshot: "..snapshot_filename)
		iter_time = sys.clock() - iter_time
		print(string.format("Iter %d done in %ds", iteration, iter_time))
	end

	print(string.format("Step %d / %d, shutting down", step, opt.steps))

	pool:terminate()

end

local opt = setup(opt)

run_loop(opt)