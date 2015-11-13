require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'


local model = torch.load("../data/model.t7")


local rnn_size = 50
local lstm_c = torch.zeros(rnn_size)
local lstm_h = lstm_c:clone()


local sequence = {1,4,23,122,1234,112,56}
local embeddings = model.embed:forward(torch.Tensor(sequence))
local labels = {}
for i = 1, #sequence do
  local embedding = embeddings[i]
  lstm_c, lstm_h = unpack(model.lstm:forward{embedding,lstm_c,lstm_h})
  local prediction = model.softmax:forward(lstm_h)
  _, label = prediction:max(1)
  print(label)
end
