
local cv = require 'cv'
require 'cv.videoio'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'
local alewrap = require 'alewrap'
local nn = require 'nn'



require "init"
require "paths"
require "SharedNoiseTable"
require "ga_worker"

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:text()
cmd:text('Run GA Policy:')
cmd:option('-snapshot', '', 'Snapshot file to use')
cmd:option('-pop_idx', -1, 'Use specific pop member (use elite by default)')
cmd:option('-game_path', '../roms/', 'path of roms')
cmd:option('-game', 'space_invaders', 'Game to run')
cmd:option('-no_vid', false, "Do not output a video, just run and print scores")
cmd:option('-video', 'out_video.mp4', 'Out video')
cmd:option('-run_count', 3, '# of runs')
cmd:option('-gpu', -1, 'Use GPU')
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

function torch_to_opencv(img)
   local img_opencv
   if img:nDimension() == 2 then
      img_opencv = img
   else
      img_opencv = img:permute(2,3,1):contiguous()
   end
   img_opencv:mul(255)
   img_opencv = img_opencv:byte()
   -- uncomment this if the channel order matters for you,
   -- for example if you're going to use imwrite on the result:
   -- cv.cvtColor{img_opencv, img_opencv, cv.COLOR_BGR2RGB}
   return cv.resize{img_opencv, {1024, 780}}
end



assert(opt.snapshot, "Provide a snapshot file")

torchSetup({gpu=opt.gpu, verbose=0})

local snapshot = torch.load(opt.snapshot)

local seeds = snapshot.elite_seeds[1]
if opt.pop_idx ~= -1 then
	seeds = snapshot.pop_seeds[opt.pop_seeds]
end

print(string.format("Loaded snapshot, iteration %d (%d steps)",
					snapshot.iteration, snapshot.step))

local noise = SharedNoiseTable()

local model, scales = get_network("ga_convnet", {gpu=opt.gpu,verbose=0})

local theta = compute_weights(noise, scales, seeds)

model = set_model_weights(model, theta, opt.gpu)

local out_fourcc = cv.VideoWriter.fourcc {
    c1 = 'X',
    c2 = 'V',
    c3 = 'I',
    c4 = 'D'
}
local video = cv.VideoWriter{
    opt.video,
    out_fourcc,
    16,
    {1024, 780}
}

local episodes = torch.Tensor(opt.run_count)
local gameEnv, gameActions = buildGameEnv({game_path=opt.game_path,env=opt.game,framework='alewrap'})

for i=1,opt.run_count do

	local screen, reward, terminal = gameEnv:newGame()

	if not opt.no_vid and not video:isOpened() then
	    video:open {
	        opt.video,
	        out_fourcc,
	        16,
	        {1024, 780}
	    }
	end

	episode_reward = 0

	video:write { torch_to_opencv(screen:squeeze()) }

	while not terminal do
		local action_idx = do_step(screen, model, opt.gpu)
		screen, reward, terminal = gameEnv:step(gameActions[action_idx])
		video:write { torch_to_opencv(screen:squeeze()) }

		episode_reward = episode_reward + reward

	end

	episodes[i] = episode_reward

end

if video:isOpened() then
	video:release()
end

print(string.format("Max: %d, Avg: %d, Std: %d",
	  				torch.max(episodes), torch.mean(episodes), torch.std(episodes)))