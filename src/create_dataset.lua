require "create_vocabulary"
require "torch"

function create_dataset(file_name,in_min_freq,out_min_freq)
  local in_column_number = 1
  local out_column_number = 2
  local in_encoder, in_decoder = create_encoder_decoder(file_name,in_column_number,in_min_freq)
  local out_encoder, out_decoder = create_encoder_decoder(file_name,out_column_number,out_min_freq)
  local X = {}
  local Y = {}
  local f = io.open(file_name)
  for line in f.lines() do
    local temp = {}
    for each in line:gmatch("%S+") do table.insert(temp, each) end
    local input = temp[in_column_number]
    local output = temp[out_column_number]
    table.insert(X,input)
    table.insert(Y,output)
  end
  X = torch.Tensor(X)
  Y = torch.Tensor(Y)
  dataset = {}
  dataset["X"] = X
  dataset["Y"] = Y
  return dataset
end
