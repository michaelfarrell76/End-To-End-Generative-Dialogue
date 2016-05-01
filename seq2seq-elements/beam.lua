require 'nn'
require 'rnn'
require 'string'
require 'hdf5'

require 'data.lua'

stringx = require('pl.stringx')

------------
-- Options
------------

cmd = torch.CmdLine()

-- file location
cmd:option('-model', 'seq2seq_lstm.t7.', [[Path to model .t7 file]])
cmd:option('-src_file', '', [[Source sequence to decode (one line per sequence)]])
cmd:option('-targ_file', '', [[True target sequence (optional)]])
cmd:option('-output_file', 'pred.txt', [[Path to output the predictions (each line will be the
                                       decoded sequence]])
cmd:option('-src_dict', 'data/demo.src.dict', [[Path to source vocabulary (*.src.dict file)]])
cmd:option('-targ_dict', 'data/demo.targ.dict', [[Path to target vocabulary (*.targ.dict file)]])
cmd:option('-char_dict', 'data/demo.char.dict', [[If using chars, path to character 
                                                vocabulary (*.char.dict file)]])

-- beam search options
cmd:option('-beam', 7, [[Beam size]])
cmd:option('-max_sent_l', 250, [[Maximum sentence length. If any sequences in srcfile are longer
                               than this then it will error out]])
cmd:option('-simple', 0, [[If = 1, output prediction is simply the first time the top of the beam
                         ends with an end-of-sentence token. If = 0, the model considers all 
                         hypotheses that have been generated so far that ends with end-of-sentence 
                         token and takes the highest scoring of all of them.]])
-- cmd:option('-replace_unk', 0, [[Replace the generated UNK tokens with the source token that 
--                               had the highest attention weight. If srctarg_dict is provided, 
--                               it will lookup the identified source token and give the corresponding 
--                               target token. If it is not provided (or the identified source token
--                               does not exist in the table) then it will copy the source token]])
cmd:option('-srctarg_dict', 'data/en-de.dict', [[Path to source-target dictionary to replace UNK 
                             tokens. See README.md for the format this file should be in]])
-- cmd:option('-score_gold', 1, [[If = 1, score the log likelihood of the gold as well]])
cmd:option('-n_best', 1, [[If > 1, it will also output an n_best list of decoded sentences]])
cmd:option('-gpuid',  -1, [[ID of the GPU to use (-1 = use CPU)]])
cmd:option('-gpuid2', -1, [[Second GPU ID]])

opt = cmd:parse(arg)

------------
-- Misc
------------

function copy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in pairs(orig) do
            copy[orig_key] = orig_value
        end
    else
        copy = orig
    end
    return copy
end

-- Convert a flat index to a row-column tuple
function flat_to_rc(v, flat_index)
    local row = math.floor((flat_index - 1) / v:size(2)) + 1
    return row, (flat_index - 1) % v:size(2) + 1
end

function clean_sent(sent)
    local s = stringx.replace(sent, UNK_WORD, '')
    s = stringx.replace(s, START_WORD, '')
    s = stringx.replace(s, END_WORD, '')
    return s
end

function strip(s)
    return s:gsub("^%s+",""):gsub("%s+$","")
end

function flip_table(u)
    local t = {}
    for key, value in pairs(u) do
        t[value] = key
    end
    return t   
end

------------
-- Indexing
------------

function idx2key(file)
    local f = io.open(file,'r')
    local t = {}
    for line in f:lines() do
        local c = {}
        for w in line:gmatch'([^%s]+)' do
            table.insert(c, w)
        end
        t[tonumber(c[2])] = c[1]
    end
    return t
end

function sent2wordidx(sent, word2idx)
    local t = {}
    local u = {}
    
    for word in sent:gmatch'([^%s]+)' do
        local idx = word2idx[word] or UNK 
        table.insert(t, idx)
        table.insert(u, word)
    end
    
    return torch.LongTensor(t), u
end

function wordidx2sent(sent, idx2word, source_str, skip_end)
    local t = {}
    local start_i, end_i
    
    if skip_end then
        end_i = #sent-1
    else
        end_i = #sent
    end

    for i = 2, end_i do -- Skip START and END
        -- TODO: do something clever about UNKs?
        -- if sent[i] == UNK then
        --     if opt.replace_unk == 1 then
        --         local s = source_str[attn[i]]
        --         if phrase_table[s] ~= nil then
        --             print(s .. ':' ..phrase_table[s])
        --         end
        --         local r = phrase_table[s] or s
        --         table.insert(t, r)
        --     else
        --         table.insert(t, idx2word[sent[i]])
        --     end
        -- else
        table.insert(t, idx2word[sent[i]])  
        -- end
    end
    return table.concat(t, ' ')
end

------------
-- State
------------

local StateAll = torch.class("StateAll")

function StateAll.initial(start)
    return {start}
end

function StateAll.advance(state, token)
    local new_state = copy(state)
    table.insert(new_state, token)
    return new_state
end

function StateAll.disallow(out)
    local bad = {1, 3} -- 1 is PAD, 3 is BOS
    for j = 1, #bad do
        out[bad[j]] = -1e9
    end
end

function StateAll.same(state1, state2)
    for i = 2, #state1 do
        if state1[i] ~= state2[i] then
            return false
        end
    end
    return true
end

function StateAll.next(state)
   return state[#state]
end

function StateAll.heuristic(state)
   return 0
end

function StateAll.print(state)
   for i = 1, #state do
      io.write(state[i] .. " ")
   end
   print()
end

------------
-- Scoring
------------

function forward_connect(enc_rnn, dec_rnn, seq_length)
    print(enc_rnn.outputs)
    dec_rnn.userPrevOutput = nn.rnn.recursiveCopy(dec_rnn.userPrevOutput, enc_rnn.outputs[seq_length])
    dec_rnn.userPrevCell = nn.rnn.recursiveCopy(dec_rnn.userPrevCell, enc_rnn.cells[seq_length])
end

function get_scores(m, source, beam)
    local source_l = source:size(1)
    source = source:view(1, -1) -- :expand(beam:size(1), source_l)
    
    -- Forward prop enc
    local enc_out = m.enc:forward(source)
    -- local final_enc_rnn = m.enc.modules[#m.enc.modules - 1]
    -- local initial_dec_rnn = m.dec.modules[3]

    for key,value in pairs(m.enc_rnn) do
        print("found member " .. key);
    end
    forward_connect(m.enc_rnn, m.dec_rnn, source_l)

    -- Forward prop dec
    local dec_out = m.dec:forward(beam)
    return torch.Tensor()
end

------------
-- Beam magic
------------

function generate_beam(m, initial, K, max_sent_l, source, gold)
    -- Reset decoder initial states
    if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
        cutorch.setDevice(opt.gpuid)
    end

    -- Add EOS token to source
    local eos = torch.LongTensor({{END}})
    source = source:cat(eos)

    -- Let's get all fb up in here
    -- scores[i][k] is the log prob of the k'th hyp of i words
    -- hyps[i][k] contains the words in k'th hyp at i word
    local n = max_sent_l
    local result = {}
    local scores = torch.zeros(n + 1, K):float()
    local hyps = torch.zeros(n + 1, K, n + 1):long()
    hyps:fill(START)

    -- Find k-max columns of a matrix
    -- Use 2*k in case some are invalid
    -- NB: only available through fbcunn (which doesn't support os x)
    -- sad face
    -- local pool = nn.TemporalKMaxPooling(2*K)

    -- Beam me up, Scotty!
    for i = 1, n do
        local cur_beam = hyps[i]:narrow(2, i + 1, i)
        -- print(cur_beam)
        local cur_K = K

        -- Score all next words for each context in the beam
        -- log p(y_{i+1} | y_c, x) for all y_c
        local model_scores = get_scores(m, source, cur_beam)
        cur_beam:wtf()
        -- local input = data.make_input(article, cur_beam, cur_K)
        -- local model_scores = self.mlp:forward(input)

        local out = model_scores:clone():double()

        -- If length limit is reached, next word must be end.
        local finalized = (i == n) and self.opt.fixedLength
        if finalized then
            out[{{}, self.END}]:add(FINAL_VAL)
        else
            -- Apply hard constraints.
            out[{{}, self.START}] = -INF
            if not self.opt.allowUNK then
                out[{{}, self.UNK}] = -INF
            end
            if self.opt.fixedLength then
                out[{{}, self.END}] = -INF
            end

            -- Add additional extractive features.
            feat_gen:add_features(out, cur_beam)
        end

      -- Only take first row when starting out.
      if i == 1 then
         cur_K = 1
         out = out:narrow(1, 1, 1)
         model_scores = model_scores:narrow(1, 1, 1)
      end

      -- Prob of summary is log p + log p(y_{i+1} | y_c, x)
      for k = 1, cur_K do
         out[k]:add(scores[i][k])
      end

      -- (2) Retain the K-best words for each hypothesis using GPU.
      -- This leaves a KxK matrix which we flatten to a K^2 vector.
      local max_scores, mat_indices = find_k_max(pool, out:cuda())
      local flat = max_scores:view(max_scores:size(1)
                                      * max_scores:size(2)):float()

      -- 3) Construct the next hypotheses by taking the next k-best.
      local seen_ngram = {}
      for k = 1, K do
         for _ = 1, 100 do

            -- (3a) Pull the score, index, rank, and word of the
            -- current best in the table, and then zero it out.
            local score, index = flat:max(1)
            if finalized then
               score[1] = score[1] - FINAL_VAL
            end
            scores[i+1][k] = score[1]
            local prev_k, y_i1 = flat_to_rc(max_scores, mat_indices, index[1])
            flat[index[1]] = -INF

            -- (3b) Is this a valid next word?
            local blocked = (self.opt.blockRepeatWords and
                                words_used[i][prev_k][y_i1])

            blocked = blocked or
               (self.opt.extractive and not feat_gen:has_ngram({y_i1}))
            blocked = blocked or
               (self.opt.abstractive and feat_gen:has_ngram({y_i1}))

            -- Hypothesis recombination.
            local new_context = {}
            if self.opt.recombine then
               for j = i+2, i+W do
                  table.insert(new_context, hyps[i][prev_k][j])
               end
               table.insert(new_context, y_i1)
               blocked = blocked or util.has(seen_ngram, new_context)
            end

            -- (3c) Add the word, its score, and its features to the
            -- beam.
            if not blocked then
               -- Update tables with new hypothesis.
               for j = 1, i+W do
                  local pword = hyps[i][prev_k][j]
                  hyps[i+1][k][j] = pword
                  words_used[i+1][k][pword] = true
               end
               hyps[i+1][k][i+W+1] = y_i1
               words_used[i+1][k][y_i1] = true

               -- Keep track of hypotheses seen.
               if self.opt.recombine then
                  util.add(seen_ngram, new_context)
               end

               -- Keep track of features used (For MERT)
               feats[i+1][k]:copy(feats[i][prev_k])
               feat_gen:compute(feats[i+1][k], hyps[i+1][k],
                                model_scores[prev_k][y_i1], y_i1, i)

               -- If we have produced an END symbol, push to stack.
               if y_i1 == self.END then
                  table.insert(result, {i+1, scores[i+1][k],
                                        hyps[i+1][k]:clone(),
                                        feats[i+1][k]:clone()})
                  scores[i+1][k] = -INF
               end
               break
            end
         end
      end
   end

   -- Sort by score.
   table.sort(result, function (a, b) return a[2] > b[2] end)

    -- Return the scores and hypotheses at the final stage.
    -- return result

    -- local n = max_sent_l
    -- local prev_ks = torch.LongTensor(n, K):fill(1) -- Backpointer table
    -- local next_ys = torch.LongTensor(n, K):fill(1) -- Current States
    -- local scores = torch.FloatTensor(n, K) -- Current Scores
    -- scores:zero()
    -- local source_l = math.min(source:size(1), opt.max_sent_l)

    -- local states = {} -- Store predicted word idx
    -- states[1] = {}
    -- table.insert(states[1], initial)
    -- next_ys[1][1] = State.next(initial)
    -- for k = 1, 1 do
    --     table.insert(states[1], initial)
    --     next_ys[1][k] = State.next(initial)
    -- end

    -- local source_input = source:view(source_l, 1)

    -- local rnn_state_enc = {}
    -- for i = 1, #init_fwd_enc do
    --     table.insert(rnn_state_enc, init_fwd_enc[i]:zero())
    -- end
    -- local context = context_proto[{{}, {1, source_l}}]:clone() -- 1 x source_l x hidden_size

    -- for t = 1, source_l do
    --     local encoder_input = {source_input[t], table.unpack(rnn_state_enc)}
    --     local out = model[1]:forward(encoder_input)
    --     rnn_state_enc = out
    --     context[{{},t}]:copy(out[#out])
    -- end
    -- context = context:expand(K, source_l, model_opt.hidden_size)
    
    -- if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
    --     cutorch.setDevice(opt.gpuid2)
    --     local context2 = context_proto2[{{1, K}, {1, source_l}}]
    --     context2:copy(context)
    --     context = context2
    -- end

    -- rnn_state_dec = {}
    -- for i = 1, #init_fwd_dec do
    --     table.insert(rnn_state_dec, init_fwd_dec[i]:zero())
    -- end

    -- if model_opt.init_dec == 1 then
    --     for L = 1, model_opt.num_layers do
    --         rnn_state_dec[L*2]:copy(rnn_state_enc[L*2-1]:expand(K, model_opt.hidden_size))
    --         rnn_state_dec[L*2+1]:copy(rnn_state_enc[L*2]:expand(K, model_opt.hidden_size))
    --     end
    -- end
    -- out_float = torch.FloatTensor()
    -- print(context)
    -- context:wtf()

    -- local i = 1
    -- local done = false
    -- local max_score = -1e9
    -- local found_eos = false
    -- while (not done) and (i < n) do
    --     i = i + 1
    --     states[i] = {}
        
    --     local decoder_input1 = next_ys:narrow(1,i-1,1):squeeze()
    --     if opt.beam == 1 then
    --         decoder_input1 = torch.LongTensor({decoder_input1})
    --     end
    --     local decoder_input = {decoder_input1, context, table.unpack(rnn_state_dec)}
    --     local out_decoder = model[2]:forward(decoder_input)
    --     local out = model[3]:forward(out_decoder[#out_decoder]) -- K x vocab_size

    --     rnn_state_dec = {} -- to be modified later
    --     table.insert(rnn_state_dec, out_decoder[#out_decoder])
    --     for j = 1, #out_decoder - 1 do
    --         table.insert(rnn_state_dec, out_decoder[j])
    --     end
    --     out_float:resize(out:size()):copy(out)
    --     for k = 1, K do
    --         State.disallow(out_float:select(1, k))
    --         out_float[k]:add(scores[i-1][k])
    --     end
    --     -- All the scores available

    --     local flat_out = out_float:view(-1)
    --     if i == 2 then
    --         flat_out = out_float[1] -- all outputs same for first batch
    --     end
       
    --     for k = 1, K do
    --         while true do
    --             local score, index = flat_out:max(1)
    --             local score = score[1]
    --             local prev_k, y_i = flat_to_rc(out_float, index[1])
    --             states[i][k] = State.advance(states[i-1][prev_k], y_i)
    --             local diff = true
    --             for k2 = 1, k-1 do
    --                 if State.same(states[i][k2], states[i][k]) then
    --                     diff = false
    --                 end
    --             end

    --             if i < 2 or diff then
    --                 local max_attn, max_index = decoder_softmax.output[prev_k]:max(1)
    --                 attn_argmax[i][k] = State.advance(attn_argmax[i-1][prev_k],max_index[1])
    --                 prev_ks[i][k] = prev_k
    --                 next_ys[i][k] = y_i
    --                 scores[i][k] = score
    --                 flat_out[index[1]] = -1e9
    --                 break -- move on to next k 
    --             end
    --             flat_out[index[1]] = -1e9
    --         end
    --     end

    --     for j = 1, #rnn_state_dec do
    --         rnn_state_dec[j]:copy(rnn_state_dec[j]:index(1, prev_ks[i]))
    --     end
    --     end_hyp = states[i][1]
    --     end_score = scores[i][1]
    --     end_attn_argmax = attn_argmax[i][1]
    --     if end_hyp[#end_hyp] == END then
    --         done = true
    --         found_eos = true
    --     else
    --         for k = 1, K do
    --             local possible_hyp = states[i][k]
    --             if possible_hyp[#possible_hyp] == END then
    --                 found_eos = true
    --                 if scores[i][k] > max_score then
    --                     max_hyp = possible_hyp
    --                     max_score = scores[i][k]
    --                     max_attn_argmax = attn_argmax[i][k]
    --                 end
    --             end
    --         end
    --     end
    -- end

    -- local gold_score = 0
    -- if opt.score_gold == 1 then
    --     rnn_state_dec = {}
    --     for i = 1, #init_fwd_dec do
    --         table.insert(rnn_state_dec, init_fwd_dec[i][{{1}}]:zero())
    --     end
    --     if model_opt.init_dec == 1 then
    --         for L = 1, model_opt.num_layers do
    --             rnn_state_dec[L*2]:copy(rnn_state_enc[L*2-1][{{1}}])
    --             rnn_state_dec[L*2+1]:copy(rnn_state_enc[L*2][{{1}}])
    --         end
    --     end
    --     local target_l = gold:size(1)
    --     for t = 2, target_l do
    --         local decoder_input1
    --         if model_opt.use_chars_dec == 1 then
    --             decoder_input1 = word2charidx_targ:index(1, gold[{{t-1}}])
    --         else
    --             decoder_input1 = gold[{{t-1}}]
    --         end
    --         local decoder_input = {decoder_input1, context[{{1}}], table.unpack(rnn_state_dec)}
    --         local out_decoder = model[2]:forward(decoder_input)
    --         local out = model[3]:forward(out_decoder[#out_decoder]) -- K x vocab_size
    --         rnn_state_dec = {} -- to be modified later
    --         table.insert(rnn_state_dec, out_decoder[#out_decoder])
    --         for j = 1, #out_decoder - 1 do
    --             table.insert(rnn_state_dec, out_decoder[j])
    --         end
    --         gold_score = gold_score + out[1][gold[t]]
    --     end
    -- end
    -- if opt.simple == 1 or end_score > max_score or not found_eos then
    --     max_hyp = end_hyp
    --     max_score = end_score
    --     max_attn_argmax = end_attn_argmax
    -- end

    -- return max_hyp, max_score, max_attn_argmax, gold_score, states[i], scores[i], attn_argmax[i]
end

------------
-- Set up
------------

function main()
    -- Some globals
    PAD = 1; UNK = 2; START = 3; END = 4
    PAD_WORD = '<blank>'; UNK_WORD = '<unk>'; START_WORD = '<s>'; END_WORD = '</s>'
    MAX_SENT_L = opt.max_sent_l
    assert(path.exists(opt.src_file), 'src_file does not exist')
    assert(path.exists(opt.model), 'model does not exist')
   
    -- Parse input params
    opt = cmd:parse(arg)
    if opt.gpuid >= 0 then
        require 'cutorch'
        require 'cunn'
        require 'cudnn'
    end

    print('Loading ' .. opt.model .. '...')
    local checkpoint = torch.load(opt.model)
    print('Done!')

    -- if opt.replace_unk == 1 then
    --     phrase_table = {}
    --     if path.exists(opt.srctarg_dict) then
    --         local f = io.open(opt.srctarg_dict,'r')
    --         for line in f:lines() do
    --             local c = line:split("|||")
    --             phrase_table[strip(c[1])] = c[2]
    --         end
    --     end
    -- end

    -- Load model and word2idx/idx2word dictionaries
    model, model_opt = checkpoint[1], checkpoint[2]
    idx2word_src = idx2key(opt.src_dict)
    word2idx_src = flip_table(idx2word_src)
    idx2word_targ = idx2key(opt.targ_dict)
    word2idx_targ = flip_table(idx2word_targ)

    -- Format model
    local enc = model[1]
    local dec = model[2]

    -- Fragile: relies on final enc rnn being 2nd module from end and initial
    -- dec rnn being 3rd module from start
    local final_enc_rnn = enc.modules[#enc.modules - 1]
    local initial_dec_rnn = dec.modules[3]

    local m = {
        enc = enc,
        enc_rnn = final_enc_rnn,
        dec = dec,
        dec_rnn = initial_dec_rnn
    }
    
    -- Load gold labels if they exist
    if path.exists(opt.targ_file) then
        print('Loading GOLD labels at ' .. opt.targ_file)
        gold = {}
        local file = io.open(opt.targ_file, 'r')
        for line in file:lines() do
            table.insert(gold, line)
        end
    else
        opt.score_gold = 0
    end

    if opt.gpuid >= 0 then
        cutorch.setDevice(opt.gpuid)
        for i = 1, #model do
            if opt.gpuid2 >= 0 then
                if i == 1 then
                    cutorch.setDevice(opt.gpuid)
                else
                    cutorch.setDevice(opt.gpuid2)
                end
            end
            model[i]:double():cuda()
            model[i]:evaluate()
        end
    end

    -- softmax_layers = {}
    -- model[2]:apply(get_layer)
    -- decoder_attn:apply(get_layer)
    -- decoder_softmax = softmax_layers[1]
    -- attn_layer = torch.zeros(opt.beam, MAX_SENT_L)

    context_proto = torch.zeros(1, MAX_SENT_L, model_opt.hidden_size)
    -- local h_init_dec = torch.zeros(opt.beam, model_opt.hidden_size)
    -- local h_init_enc = torch.zeros(1, model_opt.hidden_size) 
    -- if opt.gpuid >= 0 then
    --     h_init_enc = h_init_enc:cuda()
    --     h_init_dec = h_init_dec:cuda()
    --     cutorch.setDevice(opt.gpuid)
    --     if opt.gpuid2 >= 0 then
    --         cutorch.setDevice(opt.gpuid)
    --         context_proto = context_proto:cuda()
    --         cutorch.setDevice(opt.gpuid2)
    --         context_proto2 = torch.zeros(opt.beam, MAX_SENT_L, model_opt.hidden_size):cuda()
    --     else
    --         context_proto = context_proto:cuda()
    --     end
    -- end
    -- init_fwd_enc = {}
    -- init_fwd_dec = {h_init_dec:clone()} -- initial context
    -- for L = 1, model_opt.num_layers do
    --     table.insert(init_fwd_enc, h_init_enc:clone())
    --     table.insert(init_fwd_enc, h_init_enc:clone())
    --     table.insert(init_fwd_dec, h_init_dec:clone()) -- memory cell
    --     table.insert(init_fwd_dec, h_init_dec:clone()) -- hidden state
    -- end
     
    pred_score_total = 0
    gold_score_total = 0
    pred_words_total = 0
    gold_words_total = 0
   
    State = StateAll
    local sent_id = 0
    pred_sents = {}
    local file = io.open(opt.src_file, 'r')
    local out_file = io.open(opt.output_file,'w')
    for line in file:lines() do
        sent_id = sent_id + 1
        line = clean_sent(line)
        print('SENT ' .. sent_id .. ': ' ..line)

        local source, source_str = sent2wordidx(line, word2idx_src)
        if opt.score_gold == 1 then
            target, target_str = sent2wordidx(gold[sent_id], word2idx_targ)
        end

        state = State.initial(START)
        pred, pred_score, gold_score, all_sents, all_scores = generate_beam(m,
            state, opt.beam, MAX_SENT_L, source, target)
        pred_score_total = pred_score_total + pred_score
        pred_words_total = pred_words_total + #pred - 1
        pred_sent = wordidx2sent(pred, idx2word_targ, source_str, true)
        out_file:write(pred_sent .. '\n')

        print('PRED ' .. sent_id .. ': ' .. pred_sent)
        if gold ~= nil then
            print('GOLD ' .. sent_id .. ': ' .. gold[sent_id])
            if opt.score_gold == 1 then
                print(string.format("PRED SCORE: %.4f, GOLD SCORE: %.4f", pred_score, gold_score))
                gold_score_total = gold_score_total + gold_score
                gold_words_total = gold_words_total + target:size(1) - 1
            end
        end

        if opt.n_best > 1 then
            for n = 1, opt.n_best do
                pred_sent_n = wordidx2sent(all_sents[n], idx2word_targ, source_str, false)
                local out_n = string.format("%d ||| %s ||| %.4f", n, pred_sent_n, all_scores[n])
                print(out_n)
                out_file:write(out_n .. '\n')
            end
        end

        print('')
    end

    print(string.format("PRED AVG SCORE: %.4f, PRED PPL: %.4f", pred_score_total / pred_words_total,
        math.exp(-pred_score_total / pred_words_total)))
    if opt.score_gold == 1 then
        print(string.format("GOLD AVG SCORE: %.4f, GOLD PPL: %.4f",
            gold_score_total / gold_words_total,
            math.exp(-gold_score_total / gold_words_total)))
    end
    out_file:close()
end

main()
