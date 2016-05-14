require 'rnn'
require 'hdf5'

INF = 1e15
------------
-- Coupling
------------

-- Forward coupling: copy encoder cell and output to decoder RNN
function forward_connect(enc_rnn, dec_rnn, seq_length)
    if opt.layer_type == 'bi' then
        enc_fwd_seqLSTM = enc_rnn['modules'][1]['modules'][2]['modules'][1]
        enc_bwd_seqLSTM = enc_rnn['modules'][1]['modules'][2]['modules'][2]['modules'][2]
        
        dec_rnn.userPrevOutput = enc_rnn.output[{{},seq_length}]
        dec_rnn.userPrevCell = enc_bwd_seqLSTM.cell[seq_length]
    else
        dec_rnn.userPrevOutput = nn.rnn.recursiveCopy(dec_rnn.userPrevOutput, enc_rnn.outputs[seq_length])
        if opt.layer_type ~= 'gru' and opt.layer_type ~= 'rnn' then
            dec_rnn.userPrevCell = nn.rnn.recursiveCopy(dec_rnn.userPrevCell, enc_rnn.cells[seq_length])
        end
    end
end

-- Backward coupling: copy decoder gradients to encoder RNN
function backward_connect(enc_rnn, dec_rnn)
    if opt.layer_type == 'bi' then
        enc_fwd_seqLSTM = enc_rnn['modules'][1]['modules'][2]['modules'][1]
        enc_bwd_seqLSTM = enc_rnn['modules'][1]['modules'][2]['modules'][2]['modules'][2]

        enc_fwd_seqLSTM.userNextGradCell = dec_rnn.userGradPrevCell
        enc_bwd_seqLSTM.userNextGradCell = dec_rnn.userGradPrevCell
        enc_fwd_seqLSTM.gradPrevOutput = dec_rnn.userGradPrevOutput
        enc_bwd_seqLSTM.gradPrevOutput = dec_rnn.userGradPrevOutput
    else
        if opt.layer_type ~= 'gru' and opt.layer_type ~= 'rnn' then
            enc_rnn.userNextGradCell = nn.rnn.recursiveCopy(enc_rnn.userNextGradCell, dec_rnn.userGradPrevCell)
        end
        if opt.layer_type == 'rnn' then
            enc_rnn.gradPrevOutput = nn.rnn.recursiveCopy(enc_rnn.gradPrevOutput, dec_rnn.gradPrevOutput)
        else
            enc_rnn.gradPrevOutput = nn.rnn.recursiveCopy(enc_rnn.gradPrevOutput, dec_rnn.userGradPrevOutput)
        end
    end
end

function rnn_layer(inp, hidden_size)
    rm = nn.Sequential()
        :add(nn.ParallelTable()
        :add(inp == hidden_size and nn.Identity() or nn.Linear(inp, hidden_size)) -- input layer
        :add(nn.Linear(hidden_size, hidden_size))) -- recurrent layer
        :add(nn.CAddTable()) -- merge
        :add(nn.Sigmoid()) -- transfer
    rnn = nn.Recurrence(rm, hidden_size, 1)
    return rnn
end

function seqBRNN_batched(inp, hidden_size)
    return nn.SeqBRNN(inp, hidden_size, true)
end

-- Bidirectional LSTM already includes sequencer
function add_sequencer(layer)
    if opt.layer_type ~= 'bi' then
        return nn.Sequencer(layer)
    else
        return layer
    end
end

------------
-- Structure
------------

function build_encoder_stack(recurrence, embeddings)
    local enc = nn.Sequential()
    if embeddings ~= nil then enc:add(embeddings) end

    if opt.layer_type ~= 'bi' then
        enc:add(nn.SplitTable(1, 2))
    end

    local enc_rnn
    for i = 1, opt.num_layers do
        local inp = opt.hidden_size
        if i == 1 and embeddings ~= nil then inp = opt.word_vec_size end

        local rnn = recurrence(inp, opt.hidden_size)
        enc:add(add_sequencer(rnn))

        if i == opt.num_layers then
            enc_rnn = rnn -- Save final layer of encoder
        elseif opt.dropout > 0 then
            if opt.layer_type == 'bi' then
                enc:add(nn.Unsqueeze(2))
                enc:add(nn.SplitTable(1, 2))
                enc:add(nn.Sequencer(nn.Dropout(opt.dropout)))
                enc:add(nn.JoinTable(1, 1))
            else
                enc:add(nn.Sequencer(nn.Dropout(opt.dropout)))
            end
        end
    end

    if opt.layer_type == 'bi' then
        enc:add(nn.SplitTable(1, 2))
    end
    enc:add(nn.SelectTable(-1))

    return enc, enc_rnn
end

function build_encoder(recurrence)
    local enc_embeddings = nn.LookupTable(opt.vocab_size_enc, opt.word_vec_size)
    local enc, enc_rnn = build_encoder_stack(recurrence, enc_embeddings)
    return enc, enc_rnn, enc_embeddings
end

function build_hred_encoder(recurrence)
    local enc = nn.Sequential()
    local enc_embeddings = nn.LookupTable(opt.vocab_size_enc, opt.word_vec_size)
    local par = nn.ParallelTable()

    utterance_rnns = {}
    -- Build parallel utterance rnns
    for i = 1, opt.utter_context do
        local utterance_rnn = build_encoder_stack(recurrence, enc_embeddings)
        utterance_rnn:add(nn.Unsqueeze(2))
        if opt.load_red then 
            table.insert(utterance_rnns, utterance_rnn)
        end
        par:add(utterance_rnn)
    end

    enc:add(par)
    enc:add(nn.JoinTable(2))

    -- Build context rnn
    local context_rnn, enc_rnn = build_encoder_stack(recurrence, nil)
    enc:add(context_rnn)
    
    return enc, enc_rnn, enc_embeddings, utterance_rnns
end

function build_decoder(recurrence)
    local dec = nn.Sequential()
    local dec_embeddings = nn.LookupTable(opt.vocab_size_dec, opt.word_vec_size)
    dec:add(dec_embeddings)
    
    if opt.layer_type ~= 'bi' then
        dec:add(nn.SplitTable(1, 2))
    else
        -- Decoder is not bidirectional
        recurrence = nn.SeqLSTM
    end

    local dec_rnn
    for i = 1, opt.num_layers do
        local inp = opt.hidden_size
        if i == 1 then inp = opt.word_vec_size end

        local rnn = recurrence(inp, opt.hidden_size)
        rnn.batchfirst = true
        dec:add(add_sequencer(rnn))
        if i == 1 then -- Save initial layer of decoder
            dec_rnn = rnn
        end
        if opt.dropout > 0 and i < opt.num_layers then
            if opt.layer_type == 'bi' then
                dec:add(nn.Unsqueeze(2))
                dec:add(nn.SplitTable(1, 2))
                dec:add(nn.Sequencer(nn.Dropout(opt.dropout)))
                dec:add(nn.JoinTable(1, 1))
            else
                dec:add(nn.Sequencer(nn.Dropout(opt.dropout)))
            end
        end
    end

    if opt.layer_type == 'bi' then
        dec:add(nn.SplitTable(1, 2))
    end

    dec:add(nn.Sequencer(nn.Linear(opt.hidden_size, opt.vocab_size_dec)))
    dec:add(nn.Sequencer(nn.LogSoftMax()))

    return dec, dec_rnn, dec_embeddings
end

function build()
    local recurrence = nn.LSTM
    if opt.layer_type == 'rnn' then
        recurrence = rnn_layer
    elseif opt.layer_type == 'gru' then
        recurrence = nn.GRU
    elseif opt.layer_type == 'fast' then
        recurrence = nn.FastLSTM
    elseif opt.layer_type == 'bi' then
        recurrence = seqBRNN_batched
    end
    if opt.model_type == 'hred' then
    	build_encoder = build_hred_encoder
    end

    opt.print('Building model with specs:')
    opt.print('Layer type: ' .. opt.layer_type)
    opt.print('Model type: ' .. opt.model_type)
    opt.print('Embedding size: ' .. opt.word_vec_size)
    opt.print('Hidden layer size: ' .. opt.hidden_size)
    opt.print('Number of layers: ' .. opt.num_layers)

    -- Criterion
    local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

    local enc, enc_rnn, enc_embeddings, dec, dec_rnn, dec_embeddings
    if opt.train_from:len() == 0 then   
        -- Encoder, enc_rnn is top rnn in vertical enc stack
        enc, enc_rnn, enc_embeddings = build_encoder(recurrence)

        -- Decoder, dec_rnn is lowest rnn in vertical dec stack
        dec, dec_rnn, dec_embeddings = build_decoder(recurrence)
    else
        -- Frequently the models have CudaTensors, need to load and convert to doubles
        require 'cunn'

        -- Load the model
        assert(path.exists(opt.train_from), 'checkpoint path invalid')
        opt.print('Loading ' .. opt.train_from .. '...')
        local checkpoint = torch.load(opt.train_from)
        local model, model_opt = checkpoint[1], checkpoint[2]

        -- Load the different components
        enc = model[1]:double()
        dec = model[2]:double()
        enc_rnn = model[3]:double()
        dec_rnn = model[4]:double() 
        enc_embeddings = enc['modules'][1]:double()
        dec_embeddings = dec['modules'][1]:double()

        if opt.load_red then
            local enc_red_p, _ = enc:getParameters()
            local enc_rnn_red_p, _ = enc_rnn:getParameters()
            local enc_embeddings_red_p, _ = enc_embeddings:getParameters()

            -- Encoder, enc_rnn is top rnn in vertical enc stack
            enc, enc_rnn, enc_embeddings, utterance_rnns = build_encoder(recurrence)
        
            for i = 1, #utterance_rnns do
                p, _ = utterance_rnns[i]:getParameters()
                p:copy(enc_red_p)
            end

            local p, _ = enc_rnn:getParameters()
            p:copy(enc_rnn_red_p)

            local p, _ = enc_embeddings:getParameters()
            p:copy(enc_embeddings_red_p)
        end
    end


    -- Parameter tracking
    local layers = {enc, dec}
    local num_params = 0
    local params = {}
    local grad_params = {}

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
        if opt.gpuid >= 0 then
            params[i] = params[i]:cuda()
            grad_params[i] = grad_params[i]:cuda()
        end
    end

    -- Initialize pre-trained embeddings if necessary
    if opt.train_from:len() == 0 then
        if opt.pre_word_vecs:len() > 0 then
            local f = hdf5.open(opt.pre_word_vecs)
            local pre_word_vecs = f:read('word_vecs'):all()
            opt.print('Using pre-trained word embeddings from ' .. opt.pre_word_vecs)
            for i = 1, pre_word_vecs:size(1) do
                enc_embeddings.weight[i]:copy(pre_word_vecs[i])
                dec_embeddings.weight[i]:copy(pre_word_vecs[i])
            end
        end
    end

    opt.print('Number of parameters: ' .. num_params .. '\n')

    -- Package model for training
    local m = {
        enc = layers[1],
        enc_rnn = enc_rnn,
        enc_embeddings = enc_embeddings,
        dec = layers[2],
        dec_rnn = dec_rnn,
        dec_embeddings = dec_embeddings,
        params = params,
        grad_params = grad_params
    }

    return m, criterion
end

------------
-- Training
------------

function train_ind(ind, m, criterion, data)
    m.enc:zeroGradParameters()
    m.dec:zeroGradParameters()

    local d = data[ind]
    local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
    local batch_l, target_l, source_l = d[5], d[6], d[7]
    if opt.model_type == 'hred' then source_l = opt.utter_context end

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

    -- Fix word embeddings
    if opt.fix_word_vecs == 1 then
        m.enc_embeddings.gradWeight:zero()
        m.dec_embeddings.gradWeight:zero()
    end
    
    if opt.parallel then
        return {gps = m.grad_params, batch_l = batch_l, target_l = target_l, source_l = source_l, nonzeros = nonzeros, loss = loss, param_norm = param_norm, grad_norm = grad_norm}
    else
        return batch_l, target_l, source_l, nonzeros, loss, param_norm, grad_norm
    end

end

function train(m, criterion, train_data, valid_data)
    opt.print('Beginning training...')

    local timer = torch.Timer()
    local start_decay = 0
    opt.train_perf = {}
    opt.val_perf = {}

    function clean_layer(layer)
        if opt.gpuid >= 0 then
            if type(layer.output) ~= 'table' then
                layer.output = torch.CudaTensor()
            end
            if type(layer.gradInput) ~= 'table' then 
                layer.gradInput = torch.CudaTensor()
            end
        else
            if type(layer.output) ~= 'table' then
                layer.output = torch.DoubleTensor()
            end
            if type(layer.gradInput) ~= 'table' then 
                layer.gradInput = torch.DoubleTensor()
            end
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
        print('d')
        
        local i = 1

         for j = 1, skip do
            local pkg = {parameters = m.params, index = batch_order[i]}
            parallel.children[j]:receive("noblock")
            parallel.children[j]:send(pkg)
            i = i + 1
        end


        local rec = 0
        while rec < 3 do --data:size() do
            if opt.parallel then
                if thresh ~= nil and cur_perp < thresh then
                    skip = opt.n_proc
                    for j = 2, skip do
                        local pkg = {parameters = m.params, index = batch_order[i]}
                        parallel.children[j]:send(pkg)
                        i = i + 1
                        -- here should check if i is data:size()
                    end
                    thresh = nil
                end

                -- parallel.children:join()
                local batch_l, target_l, source_l, nonzeros, loss, param_norm, grad_norm
                for j =  1, skip do

                    local reply = parallel.children[j]:receive("noblock")
                    if reply ~= nil then
                        rec = rec + 1
                        for k = 1, #m.params do
                            if opt.ada_grad then
                                historical_grad[k]:add(torch.cmul(reply.gps[k], reply.gps[k]))
                                m.params[k]:add(-1,  torch.cmul(reply.gps[k], torch.cdiv(l_r[k], torch.sqrt(fudge[k] + historical_grad[k]))))
                            else   
                                m.params[k]:add(-opt.learning_rate, reply.gps[k])
                            end
                        end

                        num_words_target = num_words_target + reply.batch_l * reply.target_l
                        num_words_source = num_words_source + reply.batch_l * reply.source_l
                        train_nonzeros = train_nonzeros + reply.nonzeros
                        train_loss = train_loss + reply.loss * reply.batch_l

                        if i <= 3 then --data:size() then
                            local pkg = {parameters = m.params, index = batch_order[i]}
                            parallel.children[j]:join()
                            parallel.children[j]:send(pkg)
                            i = i + 1
                        end

                        batch_l, target_l, source_l, nonzeros, loss, param_norm, grad_norm =  reply.batch_l, reply.target_l, reply.source_l, reply.nonzeros, reply.loss, reply.param_norm, reply.grad_norm
                    end
                    
                end
                local time_taken = timer:time().real - start_time
                if i % opt.print_every == 0  and batch_l ~= nil then
                    local stats = string.format('Epoch: %d, Batch: %d/%d, Batch size: %d, LR: %.4f, ',
                        epoch, i, data:size(), batch_l, opt.learning_rate)
                    cur_perp = math.exp(train_loss / train_nonzeros)
                    stats = stats .. string.format('PPL: %.2f, |Param|: %.2f, |GParam|: %.2f, ',
                        cur_perp, param_norm, grad_norm)
                    stats = stats .. string.format('Training: %d/%d/%d total/source/target tokens/sec',
                        (num_words_target+num_words_source) / time_taken,
                        num_words_source / time_taken, num_words_target / time_taken)
                    stats = stats .. string.format('Time ellapse: %d', timer:time().real - start_time)
                    opt.print(stats)
                end
                sys.sleep(.5)
            else
                local batch_l, target_l, source_l, nonzeros, loss, param_norm, grad_norm
                batch_l, target_l, source_l, nonzeros, loss, param_norm, grad_norm = train_ind(batch_order[i], m, criterion, train_data)

                -- Update params

                for k = 1, #m.params do
                    if opt.ada_grad then
                        historical_grad[k]:add(torch.cmul(m.grad_params[k],m.grad_params[k]))
                   
                        m.params[k]:add(-1,  torch.cmul(m.grad_params[k], torch.cdiv(l_r[k], torch.sqrt(fudge[k] + historical_grad[k]))))
                    else   
                        m.params[k]:add(-opt.learning_rate, m.grad_params[k])
                    end

                end

                -- m.dec:updateParameters(opt.learning_rate)
                -- m.enc:updateParameters(opt.learning_rate)

                -- Bookkeeping
                num_words_target = num_words_target + batch_l * target_l
                num_words_source = num_words_source + batch_l * source_l
                train_nonzeros = train_nonzeros + nonzeros
                train_loss = train_loss + loss * batch_l

                i = i + 1  
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
            end
            -- Friendly reminder
            if i % 200 == 0 then
                collectgarbage()
            end
        end
        return train_loss, train_nonzeros
    end

    local total_loss, total_nonzeros, batch_loss, batch_nonzeros

    cur_perp = INF

    thresh = opt.wait or INF + 1

    skip = 0

    if opt.parallel then
        if  cur_perp > thresh then
            skip = 1
        else
            skip = opt.n_proc
            thresh = nil
        end
    end


    if opt.ada_grad then
        opt.print('Using ada_grad')
        local fudge_fact = .000000001
        historical_grad = {}
        fudge = {}
        l_r = {}
        for k = 1, #m.params do
            historical_grad[k] = torch.zeros(m.params[k]:size(1))
            fudge[k] = torch.zeros(m.params[k]:size(1)):fill(fudge_fact)
            l_r[k] = torch.zeros(m.params[k]:size(1)):fill(opt.learning_rate)
            if opt.gpuid > 0 then
                historical_grad[k] = historical_grad[k]:cuda()
                fudge[k] = fudge[k]:cuda()
                l_r[k] = l_r[k]:cuda()
            end
        end
    end
    parallel.children:join()



    for epoch = opt.start_epoch, opt.num_epochs do
        print('b')
        -- Causing error after 1st epoch (likely because of clean_layer)
        -- TODO: figure out how to fix clean_layer
        m.enc:training()
        m.dec:training()
        print('c')
        local total_loss, total_nonzeros = train_batch(train_data, epoch)

        local train_score = math.exp(total_loss / total_nonzeros)
        opt.print('Train', train_score)

        -- local valid_score = eval(m, criterion, valid_data)
        -- opt.print('Valid', valid_score)
        valid_score = 100000
        opt.train_perf[#opt.train_perf + 1] = train_score
        opt.val_perf[#opt.val_perf + 1] = valid_score

        decay_lr(epoch)

        -- Clean and save model
        local save_file = string.format('%s_epoch%.2f_%.2f.t7', opt.save_file, epoch, valid_score)
        if epoch % opt.save_every == 0 then
            opt.print('Saving checkpoint to ' .. save_file)
            m.enc:clearState(); m.enc_rnn:clearState(); m.dec:clearState(); clean_layer(m.dec_rnn:clearState())
            torch.save(save_file, {{m.enc, m.dec, m.enc_rnn, m.dec_rnn}, opt})
            print('past')
        end
        print('a')
    end
end

function calc_bleu_score(beam_results, target)
    local bleu_scores = torch.zeros(opt.beam_k)

    -- For each of the beam examples
    for i = 1, opt.beam_k do 
        local pred = beam_results[i]

        local scores = torch.zeros(opt.max_bleu)
        -- For each of the n-grams
        for j = 0, opt.max_bleu - 1 do
            local pred_counts = {}

            -- Loop through preds by n-gram
            for k = 1, pred:size(1) - j  do

                -- Generate key
                local key = ""
                for l = 0, j do
                    if l > 0 then
                        key = key + " "
                    end
                    key = key + pred[k + l]
                end

                -- Update pred counts
                if pred_counts[key] == nil then
                    pred_counts[key] = 1
                else
                    pred_counts[key] = 1 + pred_counts[key]
                end
            end

            local target_counts = {}

             -- Loop through target by n-gram
            for k = 1, target:size(1) - j do

                -- Generate key
                local key = ""
                for l = 0, j do
                    if l > 0 then
                        key = key + " "
                    end
                    key = key + target[k + l]
                end

                -- Update target counts
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

        -- Add brevity penalty
        local log_bleu = torch.min(0, 1 - (target:size(1) / pred:size(1)))
        for j = 1, opt.max_bleu do
            log_bleu = log_bleu + (1 / opt.max_bleu) * torch.log(scores[j])
        end

        bleu_scores[i] = torch.exp(log_bleu)
    end
    return bleu_scores
end


function calc_error_rate(beam_results, target)
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
        if opt.model_type == 'hred' then source_l = opt.utter_context end

        -- Forward prop enc
        local enc_out = m.enc:forward(source)
        forward_connect(m.enc_rnn, m.dec_rnn, source_l)

        -- Forward prop dec
        local dec_out = m.dec:forward(target)
        local loss = criterion:forward(dec_out, target_out)

        -- TODO: This does not yet support batches but it could soon!
        -- Worry not younglings
        -- It does however work one example at a time
        -- opt.allow_unk = 0
        -- local sbeam = beam.new(opt, m)
        -- local k_best = sbeam:generate_k(opt.beam_k, source[1])

        -- local beam_bleu_score = calc_bleu_score(k_best, target)
        -- local beam_error_rate = calc_error_rate(k_best, target)

        -- Update values
        nll = nll + loss * batch_l
        total = total + nonzeros

        -- map_bleu_total = map_bleu_total + beam_bleu_score[1]
        -- map_error_total = map_error_total + beam_error_rate[1]

        -- best_bleu_total = best_bleu_total + torch.max(beam_bleu_score)
        -- best_error_total = best_error_total + torch.min(beam_error_rate)
    end

    local valid = math.exp(nll / total)
    return valid
end

-- Loads in data from opt.data_file and opt.val_data_file
-- into the data objects defined in data.lua
function load_data(opt)
    opt.print('Loading data...')
    local train_data = data.new(opt, opt.data_file)
    local valid_data = data.new(opt, opt.val_data_file)
      opt.print(string.format('Source vocab size: %d, Target vocab size: %d',
        valid_data.source_size, valid_data.target_size))
    opt.max_sent_l = math.max(valid_data.source:size(2),
        valid_data.target:size(2))
    opt.print(string.format('Source max sent len: %d, Target max sent len: %d',
        valid_data.source:size(2), valid_data.target:size(2)))

    opt.vocab_size_enc = valid_data.source_size
    opt.vocab_size_dec = valid_data.target_size
    opt.seq_length = valid_data.seq_length

    opt.print('Done loading data!\n')
    return train_data, valid_data, opt
end
