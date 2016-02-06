-- Only requirement allowed
require("hdf5")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')

-- Hyperparameters
-- ...


function main() 
   -- Parse input params
   opt = cmd:parse(arg)
   local f = hdf5.open(opt.datafile, 'r')
   nclasses = f:read('nclasses'):all():long()[1]
   nfeatures = f:read('nfeatures'):all():long()[1]
   valid_input = f:read('valid_input'):all()
   valid_output = f:read('valid_output'):all()
   train_input = f:read('train_input'):all()
   train_output = f:read('train_output'):all()

   nclasses = 3
   nfeatures = 5
   dummy_input = torch.IntTensor({{3,4,5,1,1},{2,3,1,1,1},{3,1,1,1,1},{2,1,1,1,1},{4,5,1,1,1}})
   dummy_output = torch.IntTensor({1,2,3,1,2})

   --local W, b = mini_batch_SGD(train_input, train_output)
   local W, b = mini_batch_SGD(dummy_input, dummy_output)
   -- local W, b = get_naive_bayes(train_input, train_output, .2)

   -- Train.
   print(test(W, b, dummy_input, dummy_output))
   --print(test(W, b, valid_input, valid_output))

   -- Test.
end

function test(W, b, input, output)
   local input = input
   local output = output
   local size = output:size(1)
   local temp = torch.DoubleTensor(#input[1])
   local num_correct = 0
   for i = 1, size do
      -- gets features --
      truncated = input[{{i,i},{1, input[i]:gt(1):sum()}}][1]
      -- basically multiplies by feature vector --
      --print('wx', W:index(1,truncated:long()):sum(1):view(nclasses))
      --print('b', b)
      temp:add(W:index(1,truncated:long()):sum(1):view(nclasses), b)
      -- print(temp)
      -- gets output class --
      local maxval, argmax = temp:max(1)
      -- print(argmax[1])
      if argmax[1] == output[i] then
	 num_correct = num_correct + 1
      end
   end
   -- print(size)
   -- print(num_correct)
   return num_correct/size
end

function get_X_y(minibatch, input, ys, output)
   local input = input
   ndata = input:size(1)
   for i = 1, ndata do
      ys[i][output[i]] = 1
      for j = 1, input:size(2) do
	 if input[i][j] ~= 1 then
	    minibatch[i][input[i][j]] = 1
	 end
      end
   end
end	 


function LR_grad(chosen_outputs, i, W_grad, b_grad, Z, Z_temp, sample_size, minibatch)
	 local correct_class = chosen_outputs[i]
	 Z[i][correct_class] = -(1 - Z[i][correct_class])
	 W_grad:addcmul(W_grad,
			1 / sample_size,
			(torch.expand(minibatch:sub(i, i), nclasses, nfeatures)):transpose(1,2),
			(torch.expand(Z[i]:view(nclasses, 1), nclasses, nfeatures)):transpose(1,2))
	 Z[i]:mul(1 / sample_size)
	 b_grad:add(b_grad, Z[i])
end

function hinge_grad(chosen_outputs, i, W_grad, b_grad, Z, Z_temp, sample_size, minibatch, grad)
   grad:zero()
	 local correct_class = chosen_outputs[i]
   local _, max_class = Z[i]:max(1) 
   if (Z[i][correct_class] - Z[i][max_class[1]] <= 1) then
     grad[correct_class] = -1
     grad[max_class[1]] = 1
   end
	 W_grad:addcmul(W_grad,
			1 / sample_size,
			(torch.expand(minibatch:sub(i, i), nclasses, nfeatures)):transpose(1,2),
			(torch.expand(grad:view(nclasses, 1), nclasses, nfeatures)):transpose(1,2))
	 grad:mul(1 / sample_size)
	 b_grad:add(b_grad, grad)
   return W_grad, b_grad
end

function mini_batch_SGD(input, output)

   local eta = 1 
   local lambda = 0.1
   local sample_size = 5 
   
   local input = input
   local output = output

   local ndata = input:size(1)
   local shuffle = torch.LongTensor(ndata)
   shuffle:randperm(ndata)

   local shuffled_output = output:index(1, shuffle)
   local shuffled_input = input:index(1, shuffle)
   
   -- What we're trying to estimate
   local W = torch.DoubleTensor(nfeatures, nclasses):zero()
   local b = torch.DoubleTensor(nclasses, 1):zero()

   -- We preallocate these tensors for efficiency
   local chosen_indices = torch.LongTensor(sample_size)
   local chosen_inputs = torch.IntTensor(sample_size, input:size(2))
   local chosen_outputs = torch.IntTensor(sample_size)
   local minibatch = torch.DoubleTensor(sample_size, nfeatures)
   local ys = torch.ByteTensor(sample_size, nclasses)

   local Z = torch.DoubleTensor(sample_size, nclasses)
   local Z_temp = torch.DoubleTensor(sample_size, nclasses)

   local W_grad = torch.DoubleTensor(nfeatures, nclasses):zero()
   local b_grad = torch.DoubleTensor(nclasses, 1):zero()
   
   -- Stores the max 
   local max = torch.DoubleTensor(sample_size, 1)
   local summed = torch.DoubleTensor(sample_size, 1)
   for j = 1, 1 do
      -- Randomly choose some number of samples, properly construct features matrix
      chosen_indices:random(1, ndata)
      chosen_indices = torch.LongTensor({1,2,3,4,5})
      -- print("chosen indices")
      local left = ((j - 1) * sample_size + 1) % ndata
      --local chosen_inputs = input:narrow(1, left, sample_size)
      --local chosen_outputs = shuffled_output:narrow(1, left, sample_size)
      chosen_inputs = input:index(1, chosen_indices)
      chosen_outputs = shuffled_output:index(1, chosen_indices)

      -- Get the proper input matrix and one-hot-encoded output
      get_X_y(minibatch, chosen_inputs, ys, chosen_outputs)


      for i = 1, sample_size do
	 
      end
	 
      -- Compute Z = XW + b
      Z:addmm(torch.expand(b, nclasses, sample_size):transpose(1,2), minibatch, W)
      Z_temp:copy(Z)

      -- Compute the log of the softmax of Z
      -- Subtract the max from each element, take the exponent, sum, take the log, then add back M
      max:max(Z, 2)
      Z_temp:csub(torch.expand(max, sample_size, nclasses))
      Z_temp:exp()
      summed:sum(Z_temp, 2)
      summed:log()
      summed:add(max)
      Z:csub(torch.expand(summed, sample_size, nclasses))

      local losses = Z[ys]
      print(-losses:sum())
      ys:zero()
      -- Convert back to the softmax itself
      Z:exp()
      local grad = torch.DoubleTensor(nclasses):zero()
      
      for i = 1, sample_size do
        LR_grad(chosen_outputs, i, W_grad, b_grad, Z, Z_temp, sample_size, minibatch)
        --W_grad, b_grad = hinge_grad(chosen_outputs, i, W_grad, b_grad, Z, Z_temp, sample_size, minibatch, grad)
      end
      -- Update using "weight decay"
      W:mul(1 - eta * lambda / sample_size)
      b:mul(1 - eta * lambda / sample_size)
      b_grad:mul(eta)
      W_grad:mul(eta)
      W:csub(W_grad)
      b:csub(b_grad)

      W_grad:zero()
      b_grad:zero()
      Z:zero()
      Z_temp:zero()
   end
   print('W',W)
   print('b',b)
   -- print(W)
   return W, b
end  

function get_naive_bayes(input, output, alpha) 
   local input = input
   local output = output
   local W = torch.DoubleTensor(nfeatures, nclasses)
   local b = torch.DoubleTensor(nclasses)
   local F = torch.DoubleTensor(nfeatures, nclasses)
   F:zero()
   b:zero()

   local size = input:size(1)

   -- computes F --
   for i = 1, size do
      print(i)
      b[output[i]] = b[output[i]] + 1
      local curr_class = output[i]
      for j = 1, input:size(2) do
	 -- This is because 1 is the padding --
	 if input[i][j] == 1 then
	    break
	 end
	 -- i iterates across #samples, j across #featuresincluded --
	 F[input[i][j]][curr_class] = F[input[i][j]][curr_class] + 1 
      end
   end

   for i = 1, nclasses do
      -- KW: this can be done outside of the loop right? --
      b[i] = math.log(b[i]/size)
      local s = F:sum(1)[1][i]
      -- Isn't it easier to just add alpha to F? --
      for j = 1, nfeatures do
	 W[j][i] = math.log((F[j][i] + alpha)/(s + j*alpha))
      end
   end

   return W, b
end

function k_fold(k, training_in, training_out) 
  local correct = 0
  local total = 0
  local ranges = torch.LongTensor(k, 2)
  ranges:zero()
  local each = math.floor(training_in:size(1)/k)
  for i = 1, k do
    ranges[i][1] = (i - 1)*(each) + 1
    ranges[i][2] = i*each
  end
  ranges[k][2] = training_in:size(1)
  print(ranges)

  for i = 1, k do
    if i == 1 then
      curr_input = training_in[{{ranges[1][2] + 1, ranges[k][2]}, {}}]
      curr_output = training_out[{{ranges[1][2] + 1, ranges[k][2]}}]
    elseif i == k then
      curr_input = training_in[{{1, ranges[k][1] - 1}, {}}]
      curr_output = training_out[{{1, ranges[k][1] - 1}}]
    else
      curr_input = torch.cat(training_in[{{1, ranges[i][1] - 1}, {}}], training_in[{{ranges[i][2] + 1, ranges[k][2]}, {}}], 1)
      curr_output = torch.cat(training_out[{{1, ranges[i][1] - 1}}], training_out[{{ranges[i][2] + 1, ranges[k][2]}}], 1)
    end
    local W, b = get_naive_bayes(curr_input, curr_output, .2)
    local p, c, t = test(W, b, training_in[{{ranges[i][1], ranges[i][2]},{}}], training_out[{{ranges[i][1], ranges[i][2]}}])
    correct = correct + c
    total = total + t
    print(i)
  end
  print(correct)
  print(total)
  return correct/total
end



main()
