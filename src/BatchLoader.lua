

require 'torch'
require 'math'

local BatchLoader = {}
BatchLoader.__index = BatchLoader

function BatchLoader.create(dataset_name,dataset_encoder_decoder_name)
    local self = {}
    setmetatable(self, BatchLoader)

    -- construct a tensor with all the data
    print('loading data files...')
    self.dataset = torch.load(dataset_name)
    dataset_encoder_decoder = torch.load(dataset_encoder_decoder_name)
    self.batch_num = 0
    self.size = #(self.dataset)
    self.batch_size = self.dataset[1]:size(1)
    self.seq_length = self.dataset[1]:size(2)
    self.in_size = dataset_encoder_decoder["in_size"]
    self.out_size = dataset_encoder_decoder["out_size"]
    return self
  end


function BatchLoader:next_batch()
  self.batch_num = (self.batch_num % self.size) + 1
  local x = self.dataset[self.batch_num][{{},{},1}]
  local y = self.dataset[self.batch_num][{{},{},2}]
  return x,y
end

return BatchLoader
