require 'nn'
require 'string'
require 'hdf5'

require 'data.lua'

stringx = require('pl.stringx')

------------
-- Options
------------

cmd = torch.CmdLine()

-- file location
cmd:option('-model', 'seq2seq_lstm_attn.t7.', [[Path to model .t7 file]])
cmd:option('-src_file', '', [[Source sequence to decode (one line per sequence)]])
cmd:option('-targ_file', '', [[True target sequence (optional)]])
cmd:option('-output_file', 'pred.txt', [[Path to output the predictions (each line will be the
                                       decoded sequence]])
cmd:option('-src_dict', 'data/demo.src.dict', [[Path to source vocabulary (*.src.dict file)]])
cmd:option('-targ_dict', 'data/demo.targ.dict', [[Path to target vocabulary (*.targ.dict file)]])
cmd:option('-char_dict', 'data/demo.char.dict', [[If using chars, path to character 
                                                vocabulary (*.char.dict file)]])

-- beam search options
cmd:option('-beam', 5, [[Beam size]])
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
-- cmd:option('-n_best', 1, [[If > 1, it will also output an n_best list of decoded sentences]])
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
    s = stringx.replace(s, START_CHAR, '')
    s = stringx.replace(s, END_CHAR, '')
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

function wordidx2sent(sent, idx2word, source_str, attn, skip_end)
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
-- Beam magic
------------



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
    print(model)
    
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

   softmax_layers = {}
   model[2]:apply(get_layer)
   decoder_attn:apply(get_layer)
   decoder_softmax = softmax_layers[1]
   attn_layer = torch.zeros(opt.beam, MAX_SENT_L)
   
   context_proto = torch.zeros(1, MAX_SENT_L, model_opt.rnn_size)
   local h_init_dec = torch.zeros(opt.beam, model_opt.rnn_size)
   local h_init_enc = torch.zeros(1, model_opt.rnn_size) 
   if opt.gpuid >= 0 then
      h_init_enc = h_init_enc:cuda()      
      h_init_dec = h_init_dec:cuda()
      cutorch.setDevice(opt.gpuid)
      if opt.gpuid2 >= 0 then
     cutorch.setDevice(opt.gpuid)
     context_proto = context_proto:cuda()    
     cutorch.setDevice(opt.gpuid2)
     context_proto2 = torch.zeros(opt.beam, MAX_SENT_L, model_opt.rnn_size):cuda()
      else
     context_proto = context_proto:cuda()
      end
      attn_layer = attn_layer:cuda()
   end
   init_fwd_enc = {}
   init_fwd_dec = {h_init_dec:clone()} -- initial context   
   for L = 1, model_opt.num_layers do
      table.insert(init_fwd_enc, h_init_enc:clone())
      table.insert(init_fwd_enc, h_init_enc:clone())
      table.insert(init_fwd_dec, h_init_dec:clone()) -- memory cell
      table.insert(init_fwd_dec, h_init_dec:clone()) -- hidden state      
   end      
     
   pred_score_total = 0
   gold_score_total = 0
   pred_words_total = 0
   gold_words_total = 0
   
   State = StateAll
   local sent_id = 0
   pred_sents = {}
   local file = io.open(opt.src_file, "r")
   local out_file = io.open(opt.output_file,'w')   
   for line in file:lines() do
      sent_id = sent_id + 1
      line = clean_sent(line)      
      print('SENT ' .. sent_id .. ': ' ..line)
      local source, source_str
      if model_opt.use_chars_enc == 0 then
     source, source_str = sent2wordidx(line, word2idx_src)
      else
     source, source_str = sent2charidx(line, char2idx, model_opt.max_word_l)
      end
      if opt.score_gold == 1 then
     target, target_str = sent2wordidx(gold[sent_id], word2idx_targ)
      end
      state = State.initial(START)
      pred, pred_score, attn, gold_score, all_sents, all_scores, all_attn = generate_beam(model,
        state, opt.beam, MAX_SENT_L, source, target)
      pred_score_total = pred_score_total + pred_score
      pred_words_total = pred_words_total + #pred - 1
      pred_sent = wordidx2sent(pred, idx2word_targ, source_str, attn, true)
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
        pred_sent_n = wordidx2sent(all_sents[n], idx2word_targ, source_str, all_attn[n], false)
        local out_n = string.format("%d ||| %s ||| %.4f", n, pred_sent_n, all_scores[n])
        print(out_n)
        out_file:write(out_n .. '\n')
     end     
      end
      
      print('')
   end
   print(string.format("PRED AVG SCORE: %.4f, PRED PPL: %.4f", pred_score_total / pred_words_total,
               math.exp(-pred_score_total/pred_words_total)))
   if opt.score_gold == 1 then      
      print(string.format("GOLD AVG SCORE: %.4f, GOLD PPL: %.4f",
              gold_score_total / gold_words_total,
              math.exp(-gold_score_total/gold_words_total)))
   end
   out_file:close()
end

main()


