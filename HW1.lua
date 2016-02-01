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

   local W, b = get_naive_bayes(train_input, train_output, 2)

   -- Train.
   print(test(W, b, valid_input, valid_output))

   -- Test.
end

function test(W, b, input, output)
  local input = input
  local output = output
  local size = output:size(1)
  local temp = torch.DoubleTensor(#input[1])
  local num_correct = 0
  for i = 1, size do
    truncated = input[{{i,i},{1, input[i]:gt(1):sum()}}][1]
    temp:add(W:index(2,truncated:long()):sum(2), b)
    print(temp)
    local maxval, argmax = temp:max(1)
    if argmax[1][1] == output[i] then
      num_correct = num_correct + 1
    end
  end
  print(size)
  print(num_correct)
  return num_correct/size
end
    
function get_naive_bayes(input, output, alpha) 
  local input = input
  local output = output
  local W = torch.DoubleTensor(nclasses, nfeatures)
  local b = torch.DoubleTensor(nclasses)
  local F = torch.DoubleTensor(nclasses, nfeatures)
  F:zero()
  b:zero()

  local size = input:size(1)

  for i = 1, size do
    print(i)
    b[output[i]] = b[output[i]] + 1
    local curr_class = output[i]
    for j = 1, input:size(2) do
      if input[i][j] == 1 then
        break
      end
      F[curr_class][input[i][j]] = F[curr_class][input[i][j]] + 1 
    end
  end

  for i = 1, nclasses do
    b[i] = math.log(b[i]/size)
    local s = F:sum(2)[i][1]
    for j = 1, nfeatures do
      W[i][j] = math.log((F[i][j] + alpha)/(s + j*alpha))
    end
  end

  return W, b
end


main()
