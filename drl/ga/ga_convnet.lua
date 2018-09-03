
--require "ga_seeds"

function countParameters(model)
    local n_parameters = 0
    for i=1, model:size() do
       local params = model:get(i):parameters()
       if params then
         --print(model:get(i).output:size())
         local weights = params[1]
         local biases  = params[2]
         print(type(model), " weights: ", weights:nElement())
         print(type(model), " biases: ", biases:nElement())
         n_parameters  = n_parameters + weights:nElement() + biases:nElement()
       end
    end
    --print("Total params: "..n_parameters)
    return n_parameters
end

return function(args)

	local net = nn.Sequential()
	local conv = nn.SpatialConvolution

    if args.gpu >= 0 then
        net:add(nn.Transpose({1,2},{2,3},{3,4}))
        convLayer = nn.SpatialConvolutionCUDA
    end


	net:add(nn.Reshape(4, 84, 84))
	local conv1 = conv(4, 32,
		               8, 8,
		               4, 4,
		               2)

	local conv2 = conv(32, 64,
			           4, 4,
			           2, 2,
			           2)

	local conv3 = conv(64, 64,
				       3, 3,
				       1, 1,
				       1)

	net:add(conv1)
	net:add(nn.ReLU())
	net:add(conv2)
	net:add(nn.ReLU())
	net:add(conv3)
	net:add(nn.ReLU())

	local nel
    if args.gpu >= 0 then
        nel = net:cuda():forward(torch.zeros(1,4, 84, 84)
                :cuda()):nElement()
    else
        nel = net:forward(torch.zeros(1,4,84,84)):nElement()
    end
    if nel ~= 7744 then
    	error(nel.." != 7744")
    end

    local fc = nn.Linear(nel, 512, true)
    local out = nn.Linear(512, 6, true)

    net:add(nn.Reshape(nel))
    net:add(fc)
    net:add(nn.ReLU())
    net:add(out)

    if args.gpu >=0 then
        net:cuda()
    end
    if args.verbose >= 2 then
        print(net)
        print('Convolutional layers flattened output size:', nel)
    end
    n_params = countParameters(net)
    assert(n_params == 4046502)

    local scales = torch.Tensor(n_params)
    assert(scales:nElement() == n_params)

    local w_idx = 1
    print(conv1.weight:nElement())
    scales[{{w_idx, w_idx + conv1.weight:nElement()-1}}] = 0.0625
    w_idx = w_idx + conv1.weight:nElement()
    scales[{{w_idx, w_idx + conv1.bias:nElement()-1}}] = 0
    w_idx = w_idx + conv1.bias:nElement()

    scales[{{w_idx, w_idx + conv2.weight:nElement()-1}}] = 0.04419417
    w_idx = w_idx + conv2.weight:nElement()
    scales[{{w_idx, w_idx + conv2.bias:nElement()-1}}] = 0
    w_idx = w_idx + conv2.bias:nElement()

    scales[{{w_idx, w_idx + conv3.weight:nElement()-1}}] = 0.04166667
    w_idx = w_idx + conv3.weight:nElement()
    scales[{{w_idx, w_idx + conv3.bias:nElement()-1}}] = 0
    w_idx = w_idx + conv3.bias:nElement()

    scales[{{w_idx, w_idx + fc.weight:nElement()-1}}] = 0.01136364
    w_idx = w_idx + fc.weight:nElement()
    scales[{{w_idx, w_idx + fc.bias:nElement()-1}}] = 0
    w_idx = w_idx + fc.bias:nElement()

    scales[{{w_idx, w_idx + out.weight:nElement()-1}}] = 0.00441942
    w_idx = w_idx + out.weight:nElement()
    print(w_idx)
    scales[{{w_idx, w_idx + out.bias:nElement()-1}}] = 0
    w_idx = w_idx + out.bias:nElement()

    assert(w_idx - 1 == n_params)

    --print(args)
    if args.seed_file then
        local weights = ga_get_network_weights_from_weights(args.seed_file, 4046502)

        --conv1:reset()
        conv1.weight = weights[1].weights:clone()
        --conv1.gradWeight = weights[1].weights:clone()
        conv1.bias = weights[1].bias:clone()
        --conv1.gradBias = weights[1].bias:clone()

        --conv2:reset()
        conv2.weight = weights[2].weights:clone()
        --conv2.gradWeight = weights[2].weights:clone()
        conv2.bias = weights[2].bias:clone()
        --conv2.gradBias = weights[2].bias:clone()

        --conv3:reset()
        conv3.weight = weights[3].weights:clone()
        --conv3.gradWeight = weights[3].weights:clone()
        conv3.bias = weights[3].bias:clone()
        --conv3.gradBias = weights[3].bias:clone()

        --fc:reset()
        fc.weight = weights[4].weights:clone()
        --fc.gradWeight = weights[4].weights:clone()
        fc.bias = weights[4].bias:clone()
        --fc.gradBias = weights[4].bias:clone()

        --out:reset()
        out.weight = weights[5].weights:clone()
        --out.gradWeight = weights[5].weights:clone()
        out.bias = weights[5].bias:clone()
        --out.gradBias = weights[5].bias:clone()
    end

    return net, scales
end
