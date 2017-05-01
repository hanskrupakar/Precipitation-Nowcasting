require 'rnn'
require 'gnuplot'
require 'cutorch'
require 'cunn'
require 'hdf5'

function sleep(n)
  os.execute("sleep " .. tonumber(n))
end

cutorch.setDevice(1) -- GPU

function cmd_args()

	cmd = torch.CmdLine()
	cmd:text()
	cmd:text()
	cmd:text('Precipitation Nowcasting')
	cmd:text()
	cmd:text('Options')
	cmd:option('-num_layers',4,'No of hidden layers')
	cmd:option('-ntype','FastLSTM','Type of RNN (FastLSTM, GRU, MuFuRu [Multi-functional Recurrent Unit], Norm [LSTM + NormStabilizer])')
	cmd:option('-test',false,'Train/Test Flag')
	cmd:option('-iters',100,'No. of iterations on dataset')
	cmd:option('-batch_size',32,'Batch size for BGD')
	cmd:option('-actual_err', false, 'Set flag to train over actual normalized errors rather than denormalized errors')
	cmd:option('-seqlen',24,'No. of sequences of 15 min precipitation parameters (should be same as preprocessing data.py script)')
	cmd:option('-hidden_size',1000,'Hidden Layer Size')
	cmd:option('-input_size',15,'No. of parameters (15)')
	cmd:option('-learning_rate',0.001,'Learning rate for training')
	cmd:option('-output_size',1,'Size of predicted output (1 - precipitation values)')
	cmd:option('-load_from','','Checkpoint save file to load model from')
	cmd:option('-lr_decay',0.1,'Learning Rate Decay')
	cmd:option('-decay_rate', 3,'Num epochs per every learning rate decay') 
	cmd:option('-finetune', false, 'Finetune on large error batches to account for lesser # of precipitation values compared to 0 prec (0.83%)')
	cmd:option('-finetune_err', 1, 'Error threshold to select finetune batches')
	cmd:option('-num_finetune', 3, 'Number of times to finetune the data')
	cmd:option('-preprocessing', 'minmax', 'Method of preprocessing used (minmax/zscore)')
	cmd:text()

	params = cmd:parse(arg)
	return params

end

function define_model(mtype, nlayers)

	local rnn = nn.Sequential()
	rnn:add(nn.Linear(params.input_size, params.hidden_size))

	for i=1, nlayers do
		
		if mtype=='FastLSTM' then
			rnn:add(nn.FastLSTM(params.hidden_size, params.hidden_size))
		elseif mtype=='Norm' then
			rnn:add(nn.FastLSTM(params.hidden_size, params.hidden_size))
			rnn:add(nn.NormStabilizer())
		elseif mtype=='GRU' then
			rnn:add(nn.GRU(params.hidden_size, params.hidden_size))
		elseif mtype=='MuFuRu' then
			rnn:add(nn.MuFuRu(params.hidden_size, params.hidden_size))
		end
	end

	rnn:add(nn.Linear(params.hidden_size, params.output_size))

	rnn = nn.Sequencer(rnn)
	rnn = rnn:cuda()
	rnn:training()
	
	return rnn
end

function file_exists(file_name)

	local f = io.open(file_name, 'r')
	if f==nil then
		return false
	else
		f:close()
		return true
	end
end 

function maybe_checkpoint_restore(params)
	
	local iteration, checkpoint = 1, 1 
	if not params.finetune and file_exists(string.format('%s_full_params.txt', params.preprocessing)) then
		local iter = 1
		for line in io.lines(string.format('%s_full_params.txt', params.preprocessing)) do
			if iter==1 then
				iteration=tonumber(line)
			elseif iter==2 then
				checkpoint=tonumber(line)+1
			else
				if params.learning_rate>=tonumber(line) then
					params.learning_rate=tonumber(line)
				end
			end
			iter = iter + 1
		end
		if checkpoint == 3000 then
			iteration=iteration+1
		end
	end
	return iteration, checkpoint
end

function denorm(params, pre_params, tensor)

	if params.actual_err then
		return tensor
	elseif params.preprocessing=='zscore' then
		if type(tensor)=='number' then
			return tensor*pre_params[4]+pre_params[3]
		else
			return torch.add(torch.mul(tensor, pre_params[4]), pre_params[3])
		end
	else
		if type(tensor)=='number' then
			return (tensor+1)*0.5*(pre_params[2]-pre_params[1])+pre_params[1]
		else
			return torch.add(torch.mul(torch.add(tensor, 1), 0.5*(pre_params[2]-pre_params[1])), pre_params[1])
		end
	end

end 

function finetune(params, pre_params, trainx, trainy)

	local fine, grad_zero = {}, {}
	for step=1,params.seqlen do
	  grad_zero[step] = torch.zeros(params.batch_size,1):cuda()
	end
	
	for line in io.lines(string.format('finetune_full_%f.txt', params.finetune_err)) do
		if tonumber(line) then
			table.insert(fine, tonumber(line))
		end
	end 

	print (string.format('\n\t\t\t\tFINETUNE OPERATION:\n\t\tOver %d batches selected using finetune_err\n\n', #fine))
	local epoch = 1
	
	while epoch<=params.num_finetune do
		local global = 1
		for i=1, #fine do
		
			local inputs, targets = {}, {}		
			for j=1, trainx:size()[2] do
				inputs[j]=trainx[{fine[i], j, {}, {}}]:cuda()
			end
			targets[1] = torch.reshape(trainy[{fine[i], 1, {}}], params.batch_size, 1):cuda()

			rnn:zeroGradParameters()
			local outputs = rnn:forward(inputs)
			local err = criterion:forward(denorm(params, pre_params, outputs[params.seqlen]), denorm(params, pre_params, targets[1]))
			
			print(string.format("Iter %d Batch %d/%d: Error = %f ; Learning Rate = %f", epoch, global, #fine, err, params.learning_rate))
			local gradOutputs = criterion:backward(outputs[params.seqlen], targets[1])
			
			grad_zero[params.seqlen] = gradOutputs
			local gradInputs = rnn:backward(inputs, grad_zero)
			rnn:updateParameters(params.learning_rate)
			global=global+1
			
		end
		epoch = epoch+1
	end
end

function read_dataset(params, mode)

	local f = hdf5.open(string.format('%s_full_dataset.h5', params.preprocessing), 'r')
	local x = f:read(string.format('%s/x', mode)):all():transpose(2,3)
	local y = f:read(string.format('%s/y', mode)):all():transpose(2,3)
	local mean, std, min, max = nil, nil, nil, nil
	
	if params.preprocessing=='zscore' then
		mean = f:read('mean'):all()[16]
		std = f:read('std'):all()[16]
	else
		min = f:read('min'):all()[16]
		max = f:read('max'):all()[16]
	end
	f:close()
	local pre_params = {min, max, mean, std}
	return x, y, pre_params

end

function train(params)

	local iteration, checkpoint, rnn, flag = nil, nil, nil, true
	if params.load_from == '' then 
		rnn = define_model(params.ntype, params.num_layers)
	else 
		rnn = torch.load(params.load_from)
	end
	iteration, checkpoint = maybe_checkpoint_restore(params)
	
	--print(rnn)

	local criterion = nn.AbsCriterion()
	criterion=criterion:cuda()
	
	local trainx, trainy, pre_params = read_dataset(params, 'train')
	local grad_zero = {}
	for step=1,params.seqlen do
	  grad_zero[step] = torch.zeros(params.batch_size,1):cuda()
	end
	
	if not file_exists(string.format('finetune_full_%f.txt', params.finetune_err)) and params.finetune or not params.finetune then 
		
		if params.finetune then
			iteration=1
			params.iters=1
			flag=false
		end

		while iteration <= params.iters do

			if checkpoint~=1 and flag then
				flag=false
			else
				checkpoint=1
			end
			
			local prev_err, ck = 0, 0
			
			for i=checkpoint, trainx:size()[1] do

				local inputs, targets = {}, {}		
				for j=1, trainx:size()[2] do
					inputs[j]=trainx[{i, j, {}, {}}]:cuda()
				end
				targets[1] = torch.reshape(trainy[{i, 1, {}}], params.batch_size, 1):cuda()
		
				rnn:zeroGradParameters()
				local outputs = rnn:forward(inputs)
				local err = criterion:forward(denorm(params, pre_params, outputs[params.seqlen]), denorm(params, pre_params, targets[1]))
				
				local file = io.open(string.format('finetune_temp.txt', params.finetune_err), 'a')	
				
				if prev_err<params.finetune_err then
					if ck>0 then
						file:write(string.format('%d', i-1)..'\n')
						ck=ck-1
					end
				else 
					if err>params.finetune_err then
						ck=ck+1
					end
				end
				
				if err>params.finetune_err then
					file:write(i..'\n')
				end
				file:close()
				
				prev_err = err
				
				if not params.finetune then
						
					print(string.format("Iter %d Batch %d/%d: Error = %f ; Learning Rate = %f", iteration, i, trainx:size()[1], err, params.learning_rate))
					local gradOutputs = criterion:backward(outputs[params.seqlen], targets[1])
					grad_zero[params.seqlen] = gradOutputs
					local gradInputs = rnn:backward(inputs, grad_zero)
					rnn:updateParameters(params.learning_rate)
			
					if i%500==0 then
						torch.save(string.format("%s_full_%d%ss.t7", params.preprocessing, params.num_layers, params.ntype), rnn)
						local file = io.open(string.format('%s_full_params.txt', params.preprocessing), 'w')
						file:write(iteration..'\n'..i..'\n'..params.learning_rate)
						file:close()
					end
				end
			end

			if iteration%params.decay_rate==0 and not params.finetune then
				params.learning_rate = params.learning_rate*params.lr_decay
			end

			iteration = iteration + 1
			
			if not params.finetune then
				os.remove(string.format('finetune_full_%f.txt', params.finetune_err))
				os.rename('finetune_temp.txt', string.format('finetune_full_%f.txt', params.finetune_err))
			end
		end
	end
	
	if params.finetune then
	
		finetune(params, pre_params, trainx, trainy)
		torch.save(string.format("%s_full_%d%ss_FINETUNED.t7", params.preprocessing, params.num_layers, params.ntype), rnn)
	end
end

function test(params)

	local rnn = torch.load(params.load_from)
	rnn:evaluate()
	rnn:float()
	print("NETWORK DESIGN:\n\n")
	print(rnn)
	local criterion = nn.AbsCriterion()
	rnn:evaluate()
	local testx, testy, pre_params = read_dataset(params, 'test')
	
	for i=1, testx:size()[1] do		
		
		local inputs, targets = {}, {}
		for j=1, testx:size()[2] do
			inputs[j]=testx[{i, j, {}, {}}]:float()
		end
		targets[1] = torch.reshape(testy[{i, 1, {}}], params.batch_size, 1):float()
		
		local output = rnn:forward(inputs)
		
		local err = criterion:forward(denorm(params, pre_params, output[params.seqlen]), denorm(params, pre_params, targets[1]))
		
		print ('BATCH: '..i..':\n ERROR for 32 ENTRIES: '.. string.format("%1.3f", err) ..' inches\n')
		
		for i=1, targets[1]:size()[1] do
			print ('\tPredicted Prec: '..string.format("%1.3f", denorm(params, pre_params, output[params.seqlen][i][1]))..' mm\tActual Prec: '..string.format("%1.3f", denorm(params, pre_params, targets[1][i][1]))..' mm')
		end
		
		print ('\n\t\t........................................\n')
		
		--[[
		local x, y, iteration = output[params.seqlen], targets[1], 0
		print ("Pearson Correlation Co-efficient R: "..string.format("%1.4f", (32*torch.sum(torch.cmul(x,y))-torch.sum(x)*torch.sum(y))/math.sqrt((32*torch.sum(torch.cmul(x,x))-torch.sum(x)*torch.sum(x))*(32*torch.sum(torch.cmul(y,y))-torch.sum(y)*torch.sum(y)))))
		--]]
		
		for line in io.lines(string.format('%s_full_params.txt', params.preprocessing)) do
			iteration=tonumber(line)
			break
		end
		
		gnuplot.pngfigure(string.format('%s_full_%d%ss_%d_TEST%d.png', params.preprocessing, params.num_layers, params.ntype, iteration,i))
		
		--gnuplot.axis({0,35,0,0.5})
		gnuplot.plot({'Actual Precipitation', torch.range(1, params.batch_size), denorm(params, pre_params, torch.reshape(targets[1], params.batch_size)), '+'}, {'Predicted Line', torch.range(1, params.batch_size), denorm(params, pre_params, torch.reshape(output[params.seqlen], params.batch_size)),'-'})
		
		gnuplot.plotflush()
		
		--sleep(1)
	end	

end

function main()

	if (arg ~= nil and arg[-1] ~= nil) then
		local params = cmd_args()
		if not params.test then
			train(params)
		else
			test(params)
		end
	end
end

main()
