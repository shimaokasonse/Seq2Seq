require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
                    -- class name is Embedding (not namespaced)
local model_utils=require 'model_utils'
local BatchLoader = require 'BatchLoader'
local LSTM = require "LSTM"

local rnn_size = 50
local seq_length = 4
local batch_size = 5
local max_epochs = 5000
loader = BatchLoader.create("../data/dataset.t7","../data/dataset_encoder_decoder.t7",batch_size,seq_length)


local protos = {}
protos.embed = Embedding(loader.in_size, rnn_size)
protos.lstm = LSTM.lstm(rnn_size)
protos.softmax = nn.Sequential():add(nn.Linear(rnn_size,loader.out_size )):add(nn.LogSoftMax())
protos.criterion = nn.ClassNLLCriterion()
local params, grad_params = model_utils.combine_all_parameters(protos.embed, protos.lstm, protos.softmax)
params:uniform(-0.08, 0.08)

-- make a bunch of clones, AFTER flattening, as that reallocates memory
local clones = {}
for name,proto in pairs(protos) do
    print('cloning '..name)
    clones[name] = model_utils.clone_many_times(proto, seq_length, not proto.parameters)
end

local initstate_c = torch.zeros(batch_size, rnn_size)
local initstate_h = initstate_c:clone()
local dfinalstate_c = initstate_c:clone()
local dfinalstate_h = initstate_c:clone()


function feval(params_)
    if params_ ~= params then
        params:copy(params_)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch()

    ------------------- forward pass -------------------
    local embeddings = {}            -- input embeddings
    local lstm_c = {[0]=initstate_c} -- internal cell states of LSTM
    local lstm_h = {[0]=initstate_h} -- output values of LSTM
    local predictions = {}           -- softmax outputs
    local loss = 0

    for t=1,seq_length do
        embeddings[t] = clones.embed[t]:forward(x[{{}, t}])
        lstm_c[t], lstm_h[t] = unpack(clones.lstm[t]:forward{embeddings[t], lstm_c[t-1], lstm_h[t-1]})
        predictions[t] = clones.softmax[t]:forward(lstm_h[t])
        loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
    end

    ------------------ backward pass -------------------
    -- complete reverse order of the above
    local dembeddings = {}                              -- d loss / d input embeddings
    local dlstm_c = {[seq_length]=dfinalstate_c}    -- internal cell states of LSTM
    local dlstm_h = {}                                  -- output values of LSTM
    for t=seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}])
        if t == seq_length then
            assert(dlstm_h[t] == nil)
            dlstm_h[t] = clones.softmax[t]:backward(lstm_h[t], doutput_t)
        else
            dlstm_h[t]:add(clones.softmax[t]:backward(lstm_h[t], doutput_t))
        end
        -- backprop through LSTM timestep
        dembeddings[t], dlstm_c[t-1], dlstm_h[t-1] = unpack(clones.lstm[t]:backward(
            {embeddings[t], lstm_c[t-1], lstm_h[t-1]},
            {dlstm_c[t], dlstm_h[t]}
        ))
        -- backprop through embeddings
        clones.embed[t]:backward(x[{{}, t}], dembeddings[t])
    end

    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    initstate_c:copy(lstm_c[#lstm_c])
    initstate_h:copy(lstm_h[#lstm_h])

    -- clip gradient element-wise
    grad_params:clamp(-5, 5)

    return loss, grad_params
end

-- optimization stuff
local losses = {}
local optim_state = {learningRate = 1e-1}
local iterations = max_epochs * loader.nbatches
print(loader.nbatches)
for i = 1, iterations do
    local _, loss = optim.adagrad(feval, params, optim_state)
    print(loss[1])
    losses[#losses + 1] = loss[1]

end
