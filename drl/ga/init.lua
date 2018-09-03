
require 'torch'
require 'nn'
require 'nngraph'
--require 'nnutils'
require 'image'
require 'Scale'

function set_gpu(opt)
    --- set gpu device
    if opt.gpu and opt.gpu >= 0 then
        require 'cutorch'
        require 'cunn'
        if opt.gpu == 0 then
            local gpu_id = tonumber(os.getenv('GPU_ID'))
            if gpu_id then opt.gpu = gpu_id+1 end
        end
        if opt.gpu > 0 then cutorch.setDevice(opt.gpu) end
        opt.gpu = cutorch.getDevice()
        print('Using GPU device id:', opt.gpu-1)
    else
        opt.gpu = -1
        if opt.verbose >= 1 then
            print('Using CPU code only. GPU device id:', opt.gpu)
        end
    end
    return opt
end

function torchSetup(_opt)
    _opt = _opt or {}
    local opt = table.copy(_opt)
    assert(opt)

    --- general setup
    opt.tensorType =  opt.tensorType or 'torch.FloatTensor'
    torch.setdefaulttensortype(opt.tensorType)
    if not opt.threads then
        opt.threads = 4
    end
    torch.setnumthreads(opt.threads)
    if not opt.verbose then
        opt.verbose = 10
    end
    if opt.verbose >= 1 then
        print('Torch Threads:', torch.getnumthreads())
    end

    opt = set_gpu(opt)

    --- set up random number generators
    -- removing lua RNG; seeding torch RNG with opt.seed and setting cutorch
    -- RNG seed to the first uniform random int32 from the previous RNG;
    -- this is preferred because using the same seed for both generators
    -- may introduce correlations; we assume that both torch RNGs ensure
    -- adequate dispersion for different seeds.
    math.random = nil
    opt.seed = opt.seed or 1
    torch.manualSeed(opt.seed)
    if opt.verbose >= 1 then
        print('Torch Seed:', torch.initialSeed())
    end
    local firstRandInt = torch.random()
    if opt.gpu >= 0 then
        cutorch.manualSeed(firstRandInt)
        if opt.verbose >= 1 then
            print('CUTorch Seed:', cutorch.initialSeed())
        end
    end

    return opt
end

function setup(_opt)
    local opt = torchSetup(_opt)

    if opt.verbose >= 1 then
        print('Set up Torch using these options:')
        for k, v in pairs(opt) do
            print(k, v)
        end
    end
    return opt
end

function buildGameEnv(opt)

	local framework = require(opt.framework)

    local gameEnv = framework.GameEnvironment(opt)
    local gameActions = gameEnv:getActions()

    return gameEnv, gameActions
end

--- other functions
function str_to_table(str)
    if type(str) == 'table' then
        return str
    end
    if not str or type(str) ~= 'string' then
        if type(str) == 'table' then
            return str
        end
        return {}
    end
    local ttr
    if str ~= '' then
        local ttx=tt
        loadstring('tt = {' .. str .. '}')()
        ttr = tt
        tt = ttx
    else
        ttr = {}
    end
    return ttr
end

function table.copy(t)
    if t == nil then return nil end
    local nt = {}
    for k, v in pairs(t) do
        if type(v) == 'table' then
            nt[k] = table.copy(v)
        else
            nt[k] = v
        end
    end
    setmetatable(nt, table.copy(getmetatable(t)))
    return nt
end