require 'hdf5'
require 'beam'
require 'dict'
require 'cunn'

------------
-- Options
------------

cmd = torch.CmdLine()

-- File location
cmd:option('-model',    'seq2seq_lstm.t7.', [[Path to model .t7 file]])
cmd:option('-lm',       'nnlm.t7',       [[Path to language model .t7 file]])
cmd:option('-src_file', '', [[Source sequence to decode (one line per sequence)]])
cmd:option('-targ_file', '', [[True target sequence (optional)]])
cmd:option('-output_file', 'pred.txt', [[Path to output the predictions (each line will be the
                                       decoded sequence]])
cmd:option('-src_dict', 'data/demo.src.dict', [[Path to source vocabulary (*.src.dict file)]])
cmd:option('-targ_dict', 'data/demo.targ.dict', [[Path to target vocabulary (*.targ.dict file)]])

-- Beam search options
-- Beam search options
cmd:option('-k',            50,     [[Beam size]])
cmd:option('-max_sent_l',   20, [[Maximum sentence length. If any sequences in srcfile are longer
                                    than this then it will error out]])
cmd:option('-simple',       0,  [[If = 1, output prediction is simply the first time the top of the beam
                                    ends with an end-of-sentence token. If = 0, the model considers all 
                                    hypotheses that have been generated so far that ends with end-of-sentence 
                                    token and takes the highest scoring of all of them.]])
cmd:option('-allow_unk',    0,  [[If = 1, prediction can include UNK tokens.]])
cmd:option('-antilm',       0,  [[If = 1, prediction limits scoring contribution from earlier input.]])
cmd:option('-gamma',        3,  [[Number of initial word probabilities to discount from sequence probability.]])
cmd:option('-lambda',       0.45,[[Discount on initial word probabilities while using antiLM.]])
cmd:option('-len_reward',       2.5,[[Discount on initial word probabilities while using antiLM.]])
cmd:option('-k2',       40,[[Discount on initial word probabilities while using antiLM.]])

cmd:option('-decay',       0.9,[[Decay rate of lambda]])
-- cmd:option('-replace_unk', 0, [[Replace the generated UNK tokens with the source token that 
--                               had the highest attention weight. If srctarg_dict is provided, 
--                               it will lookup the identified source token and give the corresponding 
--                               target token. If it is not provided (or the identified source token
--                               does not exist in the table) then it will copy the source token]])
-- cmd:option('-score_gold', 1, [[If = 1, score the log likelihood of the gold as well]])

cmd:option('-gpuid',  -1, [[ID of the GPU to use (-1 = use CPU)]])
cmd:option('-gpuid2', -1, [[Second GPU ID]])

opt = cmd:parse(arg)

------------
-- Set up
------------

-- Finds indices for all locations of a specific value
function find_all(t, val)
    local indices = {}
    for i = 1, t:size(1) do
        if t[i] == val then
            table.insert(indices, i)
        end
    end
    return indices
end


function split_utterances(source)
    -- Fix for subtle (no end utterance token)
    if opt.utter_context == 1 then
        local source_split = {}
        table.insert(source_split, source)
        return source_split
    end

    source = source:view(1, source:size(1))


    -- First determine context length
    local std_len = source:size(2)
    local source_split = {}
    for i = 1, 2 do
        table.insert(source_split, torch.zeros(source:size()))
    end

    -- Split each utterance out into entries in table
    for i = 1, source:size(1) do
        local cleaned = remove_pad(source[i])
        local end_utterances = find_all(cleaned, END_UTTERANCE)

        -- For now it's assumed opt.utter_context = 2
        -- TODO: generalize to opt.utter_context = n
        local first = cleaned:sub(1, end_utterances[1])
        if end_utterances[1] + 1 > cleaned:size(1) then
            -- Necessary to handle a few bad cases where second utterance length = 0
            end_utterances[1] = cleaned:size(1) - 1
        end
        local second = cleaned:sub(end_utterances[1] + 1, cleaned:size(1))

        -- End of first is always going to be end utterance token, which we
        -- want to replace
        first[first:size(1)] = END
        first = pad_start(first)
        second = pad_both(second)

        -- Standardize length by prepending pad tokens
        -- TODO: instead of assuming std length is the length of combined tokens,
        -- use the max seq length found in either utterance in batch (should
        -- result in less padding)
        first = pad_blank(first, std_len)
        second = pad_blank(second, std_len)

        source_split[1][{i, {}}] = first
        source_split[2][{i, {}}] = second
    end

    return source_split
end

function main()
    assert(path.exists(opt.src_file), 'src_file does not exist')
    assert(path.exists(opt.model), 'model does not exist')

    -- Parse input params
    opt = cmd:parse(arg)

    print('Loading ' .. opt.model .. '...')
    local checkpoint = torch.load(opt.model)
    local lm
    if path.exists(opt.lm) then
        local lm_checkpoint = torch.load(opt.lm)
        lm = lm_checkpoint[1][2]
    end
    print('Done!')

    -- Load model and word2idx/idx2word dictionaries
    model, model_opt = checkpoint[1], checkpoint[2]
    idx2word_src = idx2key(opt.src_dict)
    word2idx_src = flip_table(idx2word_src)
    idx2word_targ = idx2key(opt.targ_dict)
    word2idx_targ = flip_table(idx2word_targ)
    opt.layer_type = model_opt.layer_type

    -- Format model
    local enc = model[1]:double()
    local dec = model[2]:double()
    local enc_rnn = model[3]:double()
    local dec_rnn = model[4]:double()

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

    -- Initialize beam and start making predictions
    local sbeam = beam.new(opt, m, lm)
     
    local sent_id = 0
    local pred_sents = {}
    local file = io.open(opt.src_file, 'r')
    local out_file = io.open(opt.output_file,'w')

    for line in file:lines() do
        sent_id = sent_id + 1
        line = clean_sent(line)
        print('SENT ' .. sent_id .. ': ' ..line)

        local source, source_str = sent2wordidx(line, word2idx_src)
        source = pad_end(source)
        -- if opt.model_type == 'hred' then
            -- source = split_utterances(source)
        -- end

        if opt.score_gold == 1 then
            target, target_str = sent2wordidx(gold[sent_id], word2idx_targ)
        end

        
        -- local pred = sbeam:generate_map(source)

        local preds, scores = sbeam:generate_k(opt.k, source)

        local min_preds = math.min(opt.k2, #preds)
        for i = 1, min_preds do
        	local pred_sent = wordidx2sent(preds[i], idx2word_targ, true)
        	print('PRED (' .. i .. ') ' .. 'SCORE: '.. scores[i] .. ' ' .. sent_id .. ': ' .. pred_sent)
        end

        print('GOLD ' .. sent_id .. ': ' .. gold[sent_id])
        print('')
    end

    out_file:close()
end

main()
