
function create_encoder_decoder(file_name, column_num, min_freq)

    ---- Processing File
    local f = io.open(file_name)
    local vocabulary = {}

    for line in f:lines() do
        local temp = {}
        --- "word" can be any atomic symbol.
        for word in line:gmatch("%S+") do table.insert(temp, word) end
        local word = temp[column_num]
        if word then
            if vocabulary[word] then vocabulary[word] = vocabulary[word] + 1  else vocabulary[word] = 1 end
        end
    end
    f:close()

    ---- Creating Encoder and Decoder
    local id = 2 --- id 1 is for $UNKNOWN$.
    local encoder = {}
    local decoder = {}
    for word,freq in pairs(vocabulary) do
        if freq > min_freq then
            encoder[word] = id
            decoder[id] = word
            id = id + 1
        end
    end

    ---- Encoder's :encode method
    function encoder:encode(word)
        local id = encoder[word]
        --- word that is not in vocabulary is treated as $UNKNOWN$.
        if id then return id else return 1 end
    end

    ---- Decoder's :decode method
    function decoder:decode(id)
        local word = decoder[id]
        --- word that is not in vocabulary is treated as $UNKNOWN$.
        if word then return word else return "$UNKNOWN$" end
    end

    return encoder, decoder
end


function create_dataset_encoder_decoder(file_name,in_min_freq,out_min_freq)
  local in_column_number = 1
  local out_column_number = 2
  local in_encoder, in_decoder = create_encoder_decoder(file_name,in_column_number,in_min_freq)
  local out_encoder, out_decoder = create_encoder_decoder(file_name,out_column_number,out_min_freq)
  local dataset_encoder_decoder = {}
  dataset_encoder_decoder["in_encoder"] = in_encoder
  dataset_encoder_decoder["in_decoder"] = in_decoder
  dataset_encoder_decoder["out_encoder"] = out_encoder
  dataset_encoder_decoder["out_decoder"] = out_decoder
  return dataset_encoder_decoder
end


function create_dataset(file_name,dataset_encoder_decoder)
  in_column_number = 1
  out_column_number = 2
  in_encoder = dataset_encoder_decoder["in_encoder"]
  out_encoder = dataset_encoder_decoder["out_encoder"]
  local f = io.open(file_name)
  local X = {}
  local Y = {}
  for line in f:lines() do
    local temp = {}
    for w in line:gmatch("%S+") do table.insert(temp, w) end
    local input = temp[in_column_number]
    local output = temp[out_column_number]
    table.insert(X,in_encoder:encode(input))
    table.insert(Y,out_encoder:encode(output))
  end
  X = torch.Tensor(X)
  Y = torch.Tensor(Y)
  dataset = {}
  dataset["X"] = X
  dataset["Y"] = Y
  return dataset
end


---- Main --------
local file_name = "../data/sample_text.txt"
local dataset_encoder_decoder = create_dataset_encoder_decoder(file_name,0,0)
local dataset = create_dataset(file_name,dataset_encoder_decoder)
torch.save("../data/dataset_encoder_decoder.t7", dataset_encoder_decoder)
torch.save("../data/dataset.t7", dataset)
