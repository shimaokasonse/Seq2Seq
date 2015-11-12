
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
    local id = 3 --- id 1 is for $UNKNOWN$, id 2 in for $PAD$
    local encoder = {["$PAD$"]=2}
    local decoder = {[2]="$PAD$"}
    for word,freq in pairs(vocabulary) do
        if freq > min_freq then
            encoder[word] = id
            decoder[id] = word
            id = id + 1
        end
    end
    local size = id

    ---- Encoder's :encode method
    function encoder:encode(word)
        local id = encoder[word]
        --- word that is not in vocabulary is treated as $UNKNOWN$.
        if id then return id else return 1 end
    end
    print(encoder["$PAD$"])
    ---- Decoder's :decode method
    function decoder:decode(id)
        local word = decoder[id]
        --- word that is not in vocabulary is treated as $UNKNOWN$.
        if word then return word else return "$UNKNOWN$" end
    end

    return encoder, decoder, size
end


function create_dataset_encoder_decoder(file_name,in_min_freq,out_min_freq)
  local in_column_number = 1
  local out_column_number = 2
  local in_encoder, in_decoder, in_size = create_encoder_decoder(file_name,in_column_number,in_min_freq)
  local out_encoder, out_decoder, out_size = create_encoder_decoder(file_name,out_column_number,out_min_freq)
  local dataset_encoder_decoder = {}
  dataset_encoder_decoder["in_size"] = in_size
  dataset_encoder_decoder["out_size"] = out_size
  dataset_encoder_decoder["in_encoder"] = in_encoder
  dataset_encoder_decoder["in_decoder"] = in_decoder
  dataset_encoder_decoder["out_encoder"] = out_encoder
  dataset_encoder_decoder["out_decoder"] = out_decoder
  return dataset_encoder_decoder
end



function create_dataset(file_name,dataset_encoder_decoder,batch_size)
  local in_column_number = 1
  local out_column_number = 2
  local in_encoder = dataset_encoder_decoder["in_encoder"]
  local out_encoder = dataset_encoder_decoder["out_encoder"]
  local f = io.open(file_name)
  local dataset = {}
  local sequence = {}
  for line in f:lines() do
    local temp = {}
    for w in line:gmatch("%S+") do table.insert(temp, w) end
    local input = temp[in_column_number]
    local output = temp[out_column_number]
    if not input and sequence ~= {} then
      table.insert(dataset, sequence)
      sequence = {}
    end
    if input then
      table.insert(sequence, {in_encoder:encode(input), out_encoder:encode(output)})
    end
  end

  table.sort(dataset,
    function (a,b)
      return (#a > #b)
    end
    )
  local max_length = #dataset[1]  -- Because dataset is sorted by sequence length
    ---- for each batch compute maxlength, for each sample in the batch do if length < maxlength then padd
  local function edit_batch(batch)
    for i = 1, #batch do
      sequence = batch[i]
      if #sequence < max_length then
        for j = #sequence + 1, max_length do
          table.insert(sequence,{in_encoder:encode("$PAD$"),out_encoder:encode("$PAD$")})
        end
      assert(#sequence == max_length)
      end
    end
  end

  local new_dataset = {}
  local i = 1
  while true do
    batch = {}
    for j = 1, batch_size do
      sequence = dataset[i]
      if not sequence then break end
      table.insert(batch, sequence)
      i = i + 1
    end
    if not sequence then break end
    edit_batch(batch)
    new_batch = torch.Tensor(batch):squeeze()
    table.insert(new_dataset,new_batch)
  end
  return new_dataset
end


---- Main --------
local file_name = "../data/train.txt"
local dataset_encoder_decoder = create_dataset_encoder_decoder(file_name,0,0)
local dataset = create_dataset(file_name,dataset_encoder_decoder,100)
print(dataset[1]:size())
print(dataset[40])
torch.save("../data/dataset_encoder_decoder.t7", dataset_encoder_decoder)
torch.save("../data/dataset.t7", dataset)
