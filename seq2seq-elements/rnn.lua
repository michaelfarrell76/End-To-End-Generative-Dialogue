require 'rnn'
require 'hdf5'

package.path = '?.lua;' .. package.path
require 'data.lua'

------------
-- Options
------------

cmd = torch.CmdLine()

-- Data files
cmd:text("")
cmd:text("**Data options**")
cmd:text("")
cmd:option('-data_file','data/demo-train.hdf5',[[Path to the training *.hdf5 file 
                                                 from preprocess.py]])
cmd:option('-val_data_file','data/demo-val.hdf5',[[Path to validation *.hdf5 file 
                                                 from preprocess.py]])
cmd:option('-save_file', 'seq2seq_lstm', [[Save file name (model will be saved as 
                         savefile_epochX_PPL.t7 where X is the X-th epoch and PPL is 
                         the validation perplexity]])
cmd:option('-train_from', '', [[If training from a checkpoint then this is the path to the
                                pretrained model.]])

-- RNN model specs
cmd:text("")
cmd:text("**Model options**")
cmd:text("")

cmd:option('-num_layers', 1, [[Number of layers in the LSTM encoder/decoder]])
cmd:option('-hidden_size', 500, [[Size of LSTM hidden states]])
cmd:option('-word_vec_size', 500, [[Word embedding sizes]])
cmd:option('-layer_type', 'lstm', [[Recurrent layer type (rnn, gru, lstm, fast)]])
-- cmd:option('-reverse_src', 0, [[If 1, reverse the source sequence. The original 
--                               sequence-to-sequence paper found that this was crucial to 
--                               achieving good performance, but with attention models this
--                               does not seem necessary. Recommend leaving it to 0]])
-- cmd:option('-init_dec', 1, [[Initialize the hidden/cell state of the decoder at time 
--                            0 to be the last hidden/cell state of the encoder. If 0, 
--                            the initial states of the decoder are set to zero vectors]])

cmd:text("")
cmd:text("**Optimization options**")
cmd:text("")

-- Optimization
cmd:option('-num_epochs', 3, [[Number of training epochs]])
cmd:option('-start_epoch', 1, [[If loading from a checkpoint, the epoch from which to start]])
cmd:option('-param_init', 0.1, [[Parameters are initialized over uniform distribution with support
                                 (-param_init, param_init)]])
cmd:option('-learning_rate', 1, [[Starting learning rate]])
cmd:option('-max_grad_norm', 5, [[If the norm of the gradient vector exceeds this, renormalize it
                                to have the norm equal to max_grad_norm]])
cmd:option('-dropout', 0.3, [[Dropout probability.
                            Dropout is applied between vertical LSTM stacks.]])
cmd:option('-lr_decay', 0.5, [[Decay learning rate by this much if (i) perplexity does not decrease
                        on the validation set or (ii) epoch has gone past the start_decay_at_limit]])
cmd:option('-start_decay_at', 9, [[Start decay after this epoch]])
-- cmd:option('-curriculum', 0, [[For this many epochs, order the minibatches based on source
--                 sequence length. Sometimes setting this to 1 will increase convergence speed.]])
cmd:option('-pre_word_vecs_enc', '', [[If a valid path is specified, then this will load 
                                        pretrained word embeddings (hdf5 file) on the encoder side. 
                                        See README for specific formatting instructions.]])
cmd:option('-pre_word_vecs_dec', '', [[If a valid path is specified, then this will load 
                                        pretrained word embeddings (hdf5 file) on the decoder side. 
                                        See README for specific formatting instructions.]])
cmd:option('-fix_word_vecs_enc', 0, [[If = 1, fix word embeddings on the encoder side]])
cmd:option('-fix_word_vecs_dec', 0, [[If = 1, fix word embeddings on the decoder side]])

cmd:text("")
cmd:text("**Other options**")
cmd:text("")

-- GPU
cmd:option('-gpuid', -1, [[Which gpu to use. -1 = use CPU]])
cmd:option('-gpuid2', -1, [[If this is >= 0, then the model will use two GPUs whereby the encoder
                             is on the first GPU and the decoder is on the second GPU. 
                             This will allow you to train with bigger batches/models.]])

-- Bookkeeping
cmd:option('-save_every', 1, [[Save every this many epochs]])
cmd:option('-print_every', 50, [[Print stats after this many batches]])
cmd:option('-seed', 3435, [[Seed for random initialization]])

opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

------------
-- Misc
------------

-- To renormalize grad params
function renorm_grad(data, th)
    local norm = data:norm()
    if norm > th then
        data:div(norm / th)
    end
end

-- Zeros all tensors in table
function zero_table(t)
    for i = 1, #t do
        if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
            if i == 1 then
                cutorch.setDevice(opt.gpuid)
            else
                cutorch.setDevice(opt.gpuid2)
            end
        end
        t[i]:zero()
    end
end

------------
-- Coupling
------------

-- TODO: expand to work with non lstm modules

-- Forward coupling: copy encoder cell and output to decoder RNN
function forward_connect(enc_rnn, dec_rnn, seq_length)
    dec_rnn.userPrevOutput = nn.rnn.recursiveCopy(dec_rnn.userPrevOutput, enc_rnn.outputs[seq_length])
    dec_rnn.userPrevCell = nn.rnn.recursiveCopy(dec_rnn.userPrevCell, enc_rnn.cells[seq_length])
end

-- Backward coupling: copy decoder gradients to encoder RNN
function backward_connect(enc_rnn, dec_rnn)
    enc_rnn.userNextGradCell = nn.rnn.recursiveCopy(enc_rnn.userNextGradCell, dec_rnn.userGradPrevCell)
    enc_rnn.gradPrevOutput = nn.rnn.recursiveCopy(enc_rnn.gradPrevOutput, dec_rnn.userGradPrevOutput)
end

------------
-- Structure
------------

function build_encoder(recurrence)
    local enc = nn.Sequential()
    local enc_embeddings = nn.LookupTable(opt.vocab_size_enc, opt.word_vec_size)
    enc:add(enc_embeddings)
    enc:add(nn.SplitTable(1, 2))

    local enc_rnn
    for i = 1, opt.num_layers do
        local inp = opt.hidden_size
        if i == 1 then inp = opt.word_vec_size end

        local rnn = recurrence(inp, opt.hidden_size)
        enc:add(nn.Sequencer(rnn))
        if i == opt.num_layers then
            enc_rnn = rnn -- Save final layer of encoder
        elseif opt.dropout > 0 then
            enc:add(nn.Sequencer(nn.Dropout(opt.dropout)))
        end
    end

    enc:add(nn.SelectTable(-1))

    if opt.pre_word_vecs_enc:len() > 0 then
        print('TODO: bootstrap encoder word embeddings')
    end

    return enc, enc_rnn
end

function build_decoder(recurrence)
    local dec = nn.Sequential()
    local dec_embeddings = nn.LookupTable(opt.vocab_size_dec, opt.word_vec_size)
    dec:add(dec_embeddings)
    dec:add(nn.SplitTable(1, 2))

    local dec_rnn
    for i = 1, opt.num_layers do
        local inp = opt.hidden_size
        if i == 1 then inp = opt.word_vec_size end

        local rnn = recurrence(inp, opt.hidden_size)
        dec:add(nn.Sequencer(rnn))
        if i == 1 then -- Save initial layer of decoder
            dec_rnn = rnn
        end
        if opt.dropout > 0 and i < opt.num_layers then
            dec:add(nn.Sequencer(nn.Dropout(opt.dropout)))
        end
    end

    dec:add(nn.Sequencer(nn.Linear(opt.hidden_size, opt.vocab_size_dec)))
    dec:add(nn.Sequencer(nn.LogSoftMax()))

    if opt.pre_word_vecs_dec:len() > 0 then
        print('TODO: bootstrap decoder word embeddings')
    end

    return dec, dec_rnn
end

function build()
    local recurrence = nn.LSTM
    if opt.layer_type == 'rnn' then
        recurrence = nn.Recurrent
        error('RNN layer type not currently supported.')
    elseif opt.layer_type == 'gru' then
        recurrence = nn.GRU
        error('GRU layer type not currently supported.')
    elseif opt.layer_type == 'fast' then
        recurrence = nn.FastLSTM
    end

    print('\nBuilding model with specs:')
    print('Layer type: ' .. opt.layer_type)
    print('Embedding size: ' .. opt.word_vec_size)
    print('Hidden layer size: ' .. opt.hidden_size)
    print('Number of layers: ' .. opt.num_layers)

    -- Encoder, enc_rnn is top rnn in vertical enc stack
    local enc, enc_rnn = build_encoder(recurrence)

    -- Decoder, dec_rnn is lowest rnn in vertical dec stack
    local dec, dec_rnn = build_decoder(recurrence)

    -- Criterion
    local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

    if opt.train_from:len() == 1 then
        error('TODO: implement train_from')
    end

    -- Parameter tracking
    local layers = {enc, dec}
    local num_params = 0
    local params = {}
    local grad_params = {}
    for i = 1, #layers do
        if opt.gpuid2 >= 0 then
            if i == 1 then
                cutorch.setDevice(opt.gpuid)
            else
                cutorch.setDevice(opt.gpuid2)
            end
        end      
        local p, gp = layers[i]:getParameters()
        if opt.train_from:len() == 0 then
            p:uniform(-opt.param_init, opt.param_init)
        end
        num_params = num_params + p:size(1)
        params[i] = p
        grad_params[i] = gp
    end
    print('Number of parameters: ' .. num_params .. '\n')

    -- GPU
    if opt.gpuid >= 0 then
        error('TODO: implement GPU support')
        -- for i = 1, #layers do
        --  if opt.gpuid2 >= 0 then
        --      if i == 1 then
        --          cutorch.setDevice(opt.gpuid) -- Encoder on gpu1
        --      else
        --          cutorch.setDevice(opt.gpuid2) -- Decoder/generator on gpu2
        --      end
        --  end
        --  layers[i]:cuda()
        -- end
        -- if opt.gpuid2 >= 0 then
        --  cutorch.setDevice(opt.gpuid2) --criterion on gpu2
        -- end
        -- criterion:cuda()
    end

    -- Package model for training
    local m = {
        enc = enc,
        enc_rnn = enc_rnn,
        dec = dec,
        dec_rnn = dec_rnn,
        params = params,
        grad_params = grad_params
    }

    return m, criterion
end

------------
-- Training
------------

function train(m, criterion, train_data, valid_data)
    print('Beginning training...')

    local timer = torch.Timer()
    local start_decay = 0
    opt.train_perf = {}
    opt.val_perf = {}

    function clean_layer(layer)
        if opt.gpuid >= 0 then
            layer.output = torch.CudaTensor()
            layer.gradInput = torch.CudaTensor()
        else
            layer.output = torch.DoubleTensor()
            layer.gradInput = torch.DoubleTensor()
        end
        if layer.modules then
            for i, mod in ipairs(layer.modules) do
                clean_layer(mod)
            end
        end
    end

    -- Decay learning rate if validation performance does not improve or we hit
    -- opt.start_decay_at limit
    function decay_lr(epoch)
        if epoch >= opt.start_decay_at then
            start_decay = 1
        end
        
        if opt.val_perf[#opt.val_perf] ~= nil and opt.val_perf[#opt.val_perf-1] ~= nil then
            local curr_ppl = opt.val_perf[#opt.val_perf]
            local prev_ppl = opt.val_perf[#opt.val_perf-1]
            if curr_ppl > prev_ppl then
                start_decay = 1
            end
        end
        if start_decay == 1 then
            opt.learning_rate = opt.learning_rate * opt.lr_decay
        end
    end

    function train_batch(data, epoch)
        local train_nonzeros = 0
        local train_loss = 0
        local batch_order = torch.randperm(data.length) -- Shuffle that ish
        local start_time = timer:time().real
        local num_words_target = 0
        local num_words_source = 0

        for i = 1, data:size() do
            -- zero_table(grad_params, 'zero')
            m.enc:zeroGradParameters()
            m.dec:zeroGradParameters()

            local d = data[batch_order[i]]
            local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
            local batch_l, target_l, source_l = d[5], d[6], d[7]

            -- Quick hack to line up encoder/decoder connection
            -- (we need mini-batches on dim 1)
            -- TODO: change forward/backward_connect rather than transpose here
            source = source:t()
            target = target:t()

            -- Forward prop enc
            local enc_out = m.enc:forward(source)
            forward_connect(m.enc_rnn, m.dec_rnn, source_l)

            -- Forward prop dec
            local dec_out = m.dec:forward(target)
            local loss = criterion:forward(dec_out, target_out)

            -- Backward prop dec
            local grad_output = criterion:backward(dec_out, target_out)
            m.dec:backward(target, grad_output)
            backward_connect(m.enc_rnn, m.dec_rnn)

            -- Backward prop enc
            local zeroTensor = torch.Tensor(enc_out):zero()
            m.enc:backward(source, zeroTensor)

            -- Total grad norm
            local grad_norm = 0
            for j = 1, #m.grad_params do
                grad_norm = grad_norm + m.grad_params[j]:norm()^2
            end
            grad_norm = grad_norm^0.5

            -- Shrink norm
            local param_norm = 0
            local shrinkage = opt.max_grad_norm / grad_norm
            for j = 1, #m.grad_params do
                if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
                    if j == 1 then
                        cutorch.setDevice(opt.gpuid)
                    else
                        cutorch.setDevice(opt.gpuid2)
                    end
                end
                if shrinkage < 1 then
                    m.grad_params[j]:mul(shrinkage)
                end
                -- params[j]:add(grad_params[j]:mul(-opt.learning_rate))
                param_norm = param_norm + m.params[j]:norm()^2
            end
            param_norm = param_norm^0.5

            -- Update params (also could be done as above)
            m.dec:updateParameters(opt.learning_rate)
            m.enc:updateParameters(opt.learning_rate)

            -- Bookkeeping
            num_words_target = num_words_target + batch_l * target_l
            num_words_source = num_words_source + batch_l * source_l
            train_nonzeros = train_nonzeros + nonzeros
            train_loss = train_loss + loss * batch_l
            local time_taken = timer:time().real - start_time

            if i % opt.print_every == 0 then
                local stats = string.format('Epoch: %d, Batch: %d/%d, Batch size: %d, LR: %.4f, ',
                    epoch, i, data:size(), batch_l, opt.learning_rate)
                stats = stats .. string.format('PPL: %.2f, |Param|: %.2f, |GParam|: %.2f, ',
                    math.exp(train_loss / train_nonzeros), param_norm, grad_norm)
                stats = stats .. string.format('Training: %d/%d/%d total/source/target tokens/sec',
                    (num_words_target+num_words_source) / time_taken,
                    num_words_source / time_taken, num_words_target / time_taken)
                print(stats)
            end

            -- Friendly reminder
            if i % 200 == 0 then
                collectgarbage()
            end
        end
        return train_loss, train_nonzeros
    end

    local total_loss, total_nonzeros, batch_loss, batch_nonzeros
    for epoch = opt.start_epoch, opt.num_epochs do

        -- Causing error after 1st epoch (likely because of clean_layer)
        -- TODO: figure out how to fix clean_layer ASAP
        m.enc:training()
        m.dec:training()
        local total_loss, total_nonzeros = train_batch(train_data, epoch)

        local train_score = math.exp(total_loss / total_nonzeros)
        print('Train', train_score)

        local valid_score = eval(m, criterion, valid_data)
        print('Valid', valid_score)

        opt.train_perf[#opt.train_perf + 1] = train_score
        opt.val_perf[#opt.val_perf + 1] = valid_score

        decay_lr(epoch)

        -- Clean and save model
        -- local save_file = string.format('%s_epoch%.2f_%.2f.t7', opt.save_file, epoch, valid_score)
        -- if epoch % opt.save_every == 0 then
        --     print('Saving checkpoint to ' .. save_file)
        --     clean_layer(m.enc); clean_layer(m.dec);
        --     torch.save(save_file, {{m.enc, m.dec}, opt})
        -- end
    end
end

function eval(m, criterion, data)
    m.enc:evaluate()
    m.dec:evaluate()

    local nll = 0
    local total = 0

    for i = 1, data:size() do
        local d = data[i]
        local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
        local batch_l, target_l, source_l = d[5], d[6], d[7]

        -- Line up for forward_connect()
        source = source:t()
        target = target:t()

        -- Forward prop enc
        local enc_out = m.enc:forward(source)
        forward_connect(m.enc_rnn, m.dec_rnn, source_l)

        -- Forward prop dec
        local dec_out = m.dec:forward(target)
        local loss = criterion:forward(dec_out, target_out)

        nll = nll + loss * batch_l
        total = total + nonzeros
    end

    local valid = math.exp(nll / total)
    return valid
end

------------
-- Set up
------------

function main()
    -- Parse input params
    opt = cmd:parse(arg)
    if opt.gpuid >= 0 then
        print('Using CUDA on GPU ' .. opt.gpuid .. '...')
        if opt.gpuid2 >= 0 then
            print('Using CUDA on second GPU ' .. opt.gpuid2 .. '...')
        end
        require 'cutorch'
        require 'cunn'
        cutorch.setDevice(opt.gpuid)
        cutorch.manualSeed(opt.seed)
    end
    
    -- Create the data loader classes
    print('Loading data...')
    train_data = data.new(opt, opt.data_file)
    valid_data = data.new(opt, opt.val_data_file)
    print('Done!')

    print(string.format('Source vocab size: %d, Target vocab size: %d',
        valid_data.source_size, valid_data.target_size))
    opt.max_sent_l = math.max(valid_data.source:size(2),
        valid_data.target:size(2))
    print(string.format('Source max sent len: %d, Target max sent len: %d',
        valid_data.source:size(2), valid_data.target:size(2)))

    opt.vocab_size_enc = valid_data.source_size
    opt.vocab_size_dec = valid_data.target_size
    opt.seq_length = valid_data.seq_length
    
    -- Build
    local model, criterion = build()

    -- Train
    train(model, criterion, train_data, valid_data)

    -- TODO: Test
end

main()
