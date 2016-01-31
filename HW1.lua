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

   local W = torch.randn(nclasses, nfeatures)
   local b = torch.randn(nclasses)

   -- Train.
   print(test(W, b, valid_input, valid_output))

   -- Test.
end

function test(W, b, valid_input, valid_output)
  local size = valid_output:size(1)
  local output = torch.DoubleTensor(size)
  local temp = torch.DoubleTensor(#valid_input[1])
  local num_correct = 0
  for i = 1, size do
    truncated = valid_input[{{i,i},{1, valid_input[i]:gt(1):sum()}}][1]
    temp:add(W:index(2,truncated:long()):sum(2), b)
    maxval, argmax = temp:max(1)
    if argmax[1][1] == valid_output[i] then
      num_correct = num_correct + 1
    end
  end
  return num_correct/size
end
    


main()
