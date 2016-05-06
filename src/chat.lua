require 'hdf5'
require 'pl'

require 'beam'
require 'dict'

------------
-- Options
------------

cmd = torch.CmdLine()

-- File location
cmd:option('-model', 'seq2seq_lstm.t7.', [[Path to model .t7 file]])
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

opt = cmd:parse(arg)

------------
-- Misc
------------

function split(sent)

end

------------
-- Chat
------------

function chat(sbeam)
	local dialogue = {}
	local chatting = true

    local hello = 'hello!'
    local idx = sent2wordidx(hello, word2idx_targ)
    print(idx)

	-- while chatting then

	-- end

	for line in file:lines() do
	    line = clean_sent(line)
	    print('SENT ' .. sent_id .. ': ' ..line)

	    local source, source_str = sent2wordidx(line, word2idx_src)
	    if opt.score_gold == 1 then
	        target, target_str = sent2wordidx(gold[sent_id], word2idx_targ)
	    end

	    local eos = torch.LongTensor({END})
	    source = source:cat(eos)
	    local pred = sbeam:generate_map(source)
	    -- local preds = sbeam:generate_k(opt.k, source)

	    local pred_sent = wordidx2sent(preds[i], idx2word_targ, source_str, true)
	end
end

------------
-- Set up
------------

function main()
    error('Chat not yet implemented.')
    assert(path.exists(opt.model), 'model does not exist')

    -- Parse input params
    opt = cmd:parse(arg)

    print('Loading ' .. opt.model .. '...')
    local checkpoint = torch.load(opt.model)
    print('Done!')

    -- Load model and word2idx/idx2word dictionaries
    model, model_opt = checkpoint[1], checkpoint[2]
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

    -- Initialize beam and start making chit-chat
    local sbeam = beam.new(opt, m)
    chat(sbeam)
end

main()
