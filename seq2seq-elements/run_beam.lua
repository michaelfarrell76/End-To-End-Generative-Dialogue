require 'hdf5'
require 'beam.lua'

------------
-- Options
------------

cmd = torch.CmdLine()

-- File location
cmd:option('-model', 'seq2seq_lstm.t7.', [[Path to model .t7 file]])
cmd:option('-src_file', '', [[Source sequence to decode (one line per sequence)]])
cmd:option('-targ_file', '', [[True target sequence (optional)]])
cmd:option('-output_file', 'pred.txt', [[Path to output the predictions (each line will be the
                                       decoded sequence]])
cmd:option('-src_dict', 'data/demo.src.dict', [[Path to source vocabulary (*.src.dict file)]])
cmd:option('-targ_dict', 'data/demo.targ.dict', [[Path to target vocabulary (*.targ.dict file)]])

-- Beam search options
cmd:option('-k', 5, [[Beam size]])
cmd:option('-max_sent_l', 80, [[Maximum sentence length. If any sequences in srcfile are longer
                               than this then it will error out]])
cmd:option('-simple', 0, [[If = 1, output prediction is simply the first time the top of the beam
                         ends with an end-of-sentence token. If = 0, the model considers all 
                         hypotheses that have been generated so far that ends with end-of-sentence 
                         token and takes the highest scoring of all of them.]])
cmd:option('-allow_unk', 0, [[If = 1, prediction can include UNK tokens.]])
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

local function clean_sent(sent)
    local s = stringx.replace(sent, UNK_WORD, '')
    s = stringx.replace(s, START_WORD, '')
    s = stringx.replace(s, END_WORD, '')
    return s
end

local function strip(s)
    return s:gsub("^%s+",""):gsub("%s+$","")
end

local function flip_table(u)
    local t = {}
    for key, value in pairs(u) do
        t[value] = key
    end
    return t   
end

------------
-- Indexing
------------

local function idx2key(file)
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

local function sent2wordidx(sent, word2idx)
    local t = {}
    local u = {}
    
    for word in sent:gmatch'([^%s]+)' do
        local idx = word2idx[word] or UNK 
        table.insert(t, idx)
        table.insert(u, word)
    end
    
    return torch.LongTensor(t), u
end

local function wordidx2sent(sent, idx2word, source_str, skip_end)
    local t = {}
    local start_i, end_i
    
    if skip_end then
        end_i = sent:size(1) - 1
    else
        end_i = sent:size(1)
    end

    for i = 2, end_i do -- Skip START and END
        table.insert(t, idx2word[sent[i]])
    end
    return table.concat(t, ' ')
end

------------
-- Set up
------------

function main()
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

    -- Load model and word2idx/idx2word dictionaries
    model, model_opt = checkpoint[1], checkpoint[2]
    idx2word_src = idx2key(opt.src_dict)
    word2idx_src = flip_table(idx2word_src)
    idx2word_targ = idx2key(opt.targ_dict)
    word2idx_targ = flip_table(idx2word_targ)

    -- Format model
    local enc = model[1]
    local dec = model[2]
    local enc_rnn = model[3]
    local dec_rnn = model[4]

    local m = {
        enc = enc,
        enc_rnn = enc_rnn,
        dec = dec,
        dec_rnn = dec_rnn
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

    -- Create beam and start making predictions
    local sbeam = beam.new(opt, m)
     
    local pred_score_total = 0
    local gold_score_total = 0
    local pred_words_total = 0
    local gold_words_total = 0
   
    local sent_id = 0
    local pred_sents = {}
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

        local function append(source, token)
            token = torch.LongTensor({token})
            return source:cat(token)
        end

        local eos = torch.LongTensor({END})
        source = source:cat(eos)
        local pred = sbeam:generate_map(source)

        -- pred_score_total = pred_score_total + pred_score
        pred_words_total = pred_words_total + pred:size(1)
        pred_sent = wordidx2sent(pred, idx2word_targ, source_str, true)
        out_file:write(pred_sent .. '\n')

        print('PRED ' .. sent_id .. ': ' .. pred_sent)
        -- if gold ~= nil then
        --     print('GOLD ' .. sent_id .. ': ' .. gold[sent_id])
        --     if opt.score_gold == 1 then
        --         print(string.format("PRED SCORE: %.4f, GOLD SCORE: %.4f", pred_score, gold_score))
        --         gold_score_total = gold_score_total + gold_score
        --         gold_words_total = gold_words_total + target:size(1) - 1
        --     end
        -- end

        -- if opt.n_best > 1 then
        --     for n = 1, opt.n_best do
        --         pred_sent_n = wordidx2sent(all_sents[n], idx2word_targ, source_str, false)
        --         local out_n = string.format("%d ||| %s ||| %.4f", n, pred_sent_n, all_scores[n])
        --         print(out_n)
        --         out_file:write(out_n .. '\n')
        --     end
        -- end
        print('')
    end

    print(string.format("PRED AVG SCORE: %.4f, PRED PPL: %.4f", pred_score_total / pred_words_total,
        math.exp(-pred_score_total / pred_words_total)))
    -- if opt.score_gold == 1 then
    --     print(string.format("GOLD AVG SCORE: %.4f, GOLD PPL: %.4f",
    --         gold_score_total / gold_words_total,
    --         math.exp(-gold_score_total / gold_words_total)))
    -- end
    out_file:close()
end

main()
