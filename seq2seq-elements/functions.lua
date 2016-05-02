require 'rnn'
require 'hdf5'

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

    return enc, enc_rnn, enc_embeddings
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

    return dec, dec_rnn, dec_embeddings
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

    opt.print('\nBuilding model with specs:')
    opt.print('Layer type: ' .. opt.layer_type)
    opt.print('Embedding size: ' .. opt.word_vec_size)
    opt.print('Hidden layer size: ' .. opt.hidden_size)
    opt.print('Number of layers: ' .. opt.num_layers)

    -- Encoder, enc_rnn is top rnn in vertical enc stack
    local enc, enc_rnn, enc_embeddings = build_encoder(recurrence)

    -- Decoder, dec_rnn is lowest rnn in vertical dec stack
    local dec, dec_rnn, dec_embeddings = build_decoder(recurrence)

    -- Criterion
    local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

    if opt.gpuid > 0 then
        enc:cuda()
        enc_rnn:cuda()
        dec:cuda()
        dec_rnn:cuda()
        criterion:cuda()    
    end

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
    
    if opt.train_from:len() == 0 then
        if opt.pre_word_vecs:len() > 0 then
            local f = hdf5.open(opt.pre_word_vecs)     
            local pre_word_vecs = f:read('word_vecs'):all()
            for i = 1, pre_word_vecs:size(1) do
                enc_embeddings.weight[i]:copy(pre_word_vecs[i])
                dec_embeddings.weight[i]:copy(pre_word_vecs[i])
            end          
        end
    end

    opt.print('Number of parameters: ' .. num_params .. '\n')

    -- GPU
    if opt.gpuid >= 0 then
        cutorch.setDevice(opt.gpuid)
        cutorch.manualSeed(opt.seed)

        for i = 1, #layers do
         if opt.gpuid2 >= 0 then
             if i == 1 then
                 cutorch.setDevice(opt.gpuid) -- Encoder on gpu1
             else
                 cutorch.setDevice(opt.gpuid2) -- Decoder/generator on gpu2
             end
         end
         layers[i]:cuda()
        end
        if opt.gpuid2 >= 0 then
         cutorch.setDevice(opt.gpuid2) --criterion on gpu2
        end
        criterion:cuda()
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
    opt.print('Beginning training...')

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
            local zeroTensor = torch.Tensor(enc_out:size()):zero()
            if opt.gpuid >=0 then
                zeroTensor = zeroTensor:cuda()
            end
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
                opt.print(stats)
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
        -- TODO: figure out how to fix clean_layer
        m.enc:training()
        m.dec:training()
        local total_loss, total_nonzeros = train_batch(train_data, epoch)

        local train_score = math.exp(total_loss / total_nonzeros)
        opt.print('Train', train_score)

        local valid_score = eval(m, criterion, valid_data)
        opt.print('Valid', valid_score)

        opt.train_perf[#opt.train_perf + 1] = train_score
        opt.val_perf[#opt.val_perf + 1] = valid_score

        decay_lr(epoch)

        -- Clean and save model
        local save_file = string.format('%s_epoch%.2f_%.2f.t7', opt.save_file, epoch, valid_score)
        if epoch % opt.save_every == 0 then
        -- if epoch == opt.num_epochs then
            opt.print('Saving checkpoint to ' .. save_file)
            -- clean_layer(m.enc); clean_layer(m.dec);
            torch.save(save_file, {{m.enc, m.dec, m.enc_rnn, m.dec_rnn}, opt})
        end
    end
end

function beam_bleu_score(beam_results, target)
    local bleu_scores = torch.zeros(opt.beam_k)

    --for each of the beam examples
    for i = 1, opt.beam_k do 
        local pred = beam_results[i]

        local scores = torch.zeros(opt.max_bleu)
        --for each of the n-grams
        for j = 0, opt.max_bleu - 1 do
            local pred_counts = {}

            --loop through preds by n-gram
            for k = 1, pred:size(1) - j  do

                --generate key
                local key = ""
                for l = 0, j do
                    if l > 0 then
                        key = key + " "
                    end
                    key = key + pred[k + l]
                end

                --update pred counts
                if pred_counts[key] == nil then
                    pred_counts[key] = 1
                else
                    pred_counts[key] = 1 + pred_counts[key]
                end

            end

            local target_counts = {}

             --loop through target by n-gram
            for k = 1, target:size(1) - j do

                --generate key
                local key = ""
                for l = 0, j do
                    if l > 0 then
                        key = key + " "
                    end
                    key = key + target[k + l]
                end

                --update target counts
                if target_counts[key] == nil then
                    target_counts[key] = 1
                else
                    target_counts[key] = 1 + target_counts[key]
                end

            end

            local prec = 0
            for key, pred_val in pairs(pred_counts) do
                target_val = target_counts[prec]
                if target_val ~= nil then
                    if target_val >= pred_val then
                        prec = prec + pred_val
                    else
                        prec = prec + target_val
                    end
                end
            end

            local score 
            if pred:size(1) <= j then
                score = 1
            else
                score = prec / (pred:size(1) - j)
            end

            scores[j + 1] = score
        end

        --add brevity penalty
        local log_bleu = torch.min(0, 1 - (target:size(1) / pred:size(1)))

        for j = 1, opt.max_bleu do
            log_bleu = log_bleu + (1 / opt.max_bleu) * torch.log(scores[j])
        end

        bleu_scores[i] = torch.exp(log_bleu)
    end
    return bleu_scores
end


function beam_error_rate(beam_results, target)
    local error_rates = torch.zeros(opt.beam_k)
    for i = 1, opt.beam_k do 
        local pred = beam_results[i]
        local total_wrong = 0
        for j = 1, torch.min(pred:size(1), target:size(1)) do
            if pred[j] ~= target[j] then
                total_wrong = total_wrong + 1
            end
        end
        total_wrong = total_wrong + torch.abs(pred:size(1) - target:size(1))
        error_rates[i] = total_wrong / target:size(1)
    end
    return error_rates
end

function eval(m, criterion, data)
    m.enc:evaluate()
    m.dec:evaluate()

    local nll = 0
    local total = 0

    local map_bleu_total, best_bleu_total = 0, 0
    local map_error_total, best_error_total = 0, 0

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

        local beam_res = generateBeam(m, opt.beam_k, source)

        local beam_bleu_score = calc_bleu_score(beam_res, target)
        local beam_error_rate = calc_error_rate(beam_res, target)

        --update values
        nll = nll + loss * batch_l
        total = total + nonzeros

        map_bleu_total = map_bleu_total + beam_bleu_score[1]
        map_error_total = map_error_total + beam_error_rate[1]

        best_bleu_total = best_bleu_total + torch.max(beam_bleu_score)
        best_error_total = best_error_total + torch.min(beam_error_rate)
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


    torch.manualSeed(opt.seed)


    if opt.parallel then
        opt.print = parallel.print
    else
        opt.print = print
    end

    if opt.gpuid >= 0 then
        opt.print('Using CUDA on GPU ' .. opt.gpuid .. '...')
        if opt.gpuid2 >= 0 then
            opt.print('Using CUDA on second GPU ' .. opt.gpuid2 .. '...')
        end
        require 'cutorch'
        require 'cunn'
        cutorch.setDevice(opt.gpuid)
        cutorch.manualSeed(opt.seed)
    end
    
    -- Create the data loader classes
    opt.print('Loading data...')
    local train_data = data.new(opt, opt.data_file)
    local valid_data = data.new(opt, opt.val_data_file)
    opt.print('Done!')

    opt.print(string.format('Source vocab size: %d, Target vocab size: %d',
        valid_data.source_size, valid_data.target_size))
    opt.max_sent_l = math.max(valid_data.source:size(2),
        valid_data.target:size(2))
    opt.print(string.format('Source max sent len: %d, Target max sent len: %d',
        valid_data.source:size(2), valid_data.target:size(2)))

    opt.vocab_size_enc = valid_data.source_size
    opt.vocab_size_dec = valid_data.target_size
    opt.seq_length = valid_data.seq_length
    
    -- Build
    local model, criterion = build()

    if opt.parallel then 
        return train_data, valid_data, model, criterion, opt
    else
        -- Train
        train(model, criterion, train_data, valid_data)
    end

    -- TODO: Test
    
end