require "init"
require "ga_worker"
require "SharedNoiseTable"

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

function load_seeds(noise, seeds)

    local model, scales = get_network("ga_convnet", {gpu=0,verbose=0})

    return compute_weights(noise, scales, seeds)
end

function load_snapshot_elite(snap_file)
    local snapshot = torch.load(snap_file)

    return snapshot.elite
end

function test()
    torchSetup({gpu=0})
    elite = load_snapshot_elite("/data/torch-ga_space_invaders_4B/snapshot_torch_ga_22.t7")
    noise = SharedNoiseTable()

    return noise, elite
end