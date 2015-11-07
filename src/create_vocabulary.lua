
function create_encoder_decoder(file_name, column_num, in_freq,)

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
