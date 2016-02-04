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

   local W, b = get_naive_bayes(train_input, train_output, .2)
   --print(k_fold(5, train_input, train_output))--

   -- Train.
   print(test(W, b, valid_input, valid_output))

   -- Test.
end

function test(W, b, input, output)
  local input = input
  local output = output
  local size = output:size(1)
  print(size)
  print(input:size(1))
  local temp = torch.DoubleTensor(#input[1])
  local num_correct = 0
  for i = 1, size do
    -- gets features --
    if input[i][1] > 1 then
      truncated = input[{{i,i},{1, input[i]:gt(1):sum()}}][1]
      -- basically multiplies by feature vector --
      print(W:index(1, truncated:long()):sum(1):view(2))
      print(b)
      temp:add(W:index(1,truncated:long()):sum(1):view(2), b)
      -- gets output class --
      local maxval, argmax = temp:max(1)
      if argmax[1] == output[i] then
        num_correct = num_correct + 1
      end
    end
  end
  print(size)
  print(num_correct)
  return num_correct/size, num_correct, size
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
      if input[i][j] == 1 then
        break
      end
      F[input[i][j]][curr_class] = F[input[i][j]][curr_class] + 1 
    end
  end

  for i = 1, nclasses do
    b[i] = math.log(b[i]/size)
    local s = F:sum(1)[1][i]
    for j = 1, nfeatures do
      W[j][i] = math.log((F[j][i] + alpha)/(s + j*alpha))
    end
  end

  return W, b
end


--in progress--
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
