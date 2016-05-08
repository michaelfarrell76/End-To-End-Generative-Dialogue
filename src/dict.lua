-- Utility methods for working with vocabulary dictionaries

-- Some cheeky globals
PAD = 1; UNK = 2; START = 3; END = 4; END_UTTERANCE = 10005
PAD_WORD = '<blank>'; UNK_WORD = '<unk>'; START_WORD = '<s>'; END_WORD = '</s>'; END_UTTERANCE_WORD = '<t>'

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

function wordidx2sent(sent, idx2word, skip_pad)
	local skip = 0
	if skip_pad then skip = 1 end

    local t = {}
    local start_i = 1 + skip
    local end_i = sent:size(1) - skip
    
    for i = start_i, end_i do
        table.insert(t, idx2word[sent[i]])
    end
    return table.concat(t, ' ')
end

------------
-- Misc
------------

function pad_start(t)
	local bos = torch.LongTensor({START})
    return bos:cat(t)
end

function pad_end(t)
	local eos = torch.LongTensor({END})
    return t:cat(eos)
end

function pad_both(t)
    return pad_start(pad_end(t))
end

function remove_pad(t)
	return t[{{2, t:size(1) - 1}}]
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
