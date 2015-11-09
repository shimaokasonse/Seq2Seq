

require 'torch'
require 'math'

local BatchLoader = {}
BatchLoader.__index = BatchLoader

function BatchLoader.create(dataset, batch_size, seq_length)
    local self = {}
    setmetatable(self, BatchLoader)

    -- construct a tensor with all the data
    print('loading data files...')
    local dataset = torch.load(dataset)
    local X = dataset.X
    local Y = dataset.Y
    -- input and output sequences sould be same length
    assert(X:size(1)==Y:size(1))

    -- cut off the end so that it divides evenly
    local len = X:size(1)
    if len % (batch_size * seq_length) ~= 0 then
        print('cutting off end of data so that the batches/sequences divide evenly')
        X = X:sub(1, batch_size * seq_length
                    * math.floor(len / (batch_size * seq_length)))
        Y = Y:sub(1, batch_size * seq_length
                    * math.floor(len / (batch_size * seq_length)))
    end

    print(X)
    print(Y)
    self.x_batches = X:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    self.nbatches = #self.x_batches
    self.y_batches = Y:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    assert(#self.x_batches == #self.y_batches)

    self.current_batch = 0
    self.evaluated_batches = 0  -- number of times next_batch() called

    print('data load done.')
    collectgarbage()
    return self
end


function BatchLoader:next_batch()
    self.current_batch = (self.current_batch % self.nbatches) + 1
    self.evaluated_batches = self.evaluated_batches + 1
    return self.x_batches[self.current_batch], self.y_batches[self.current_batch]
end

return BatchLoader
