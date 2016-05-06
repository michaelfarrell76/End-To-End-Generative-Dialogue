-- Utility methods for working with vocabulary dictionaries

PAD = 1; UNK = 2; START = 3; END = 4
PAD_WORD = '<blank>'; UNK_WORD = '<unk>'; START_WORD = '<s>'; END_WORD = '</s>'

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
-- Misc
------------

function clean_sent(sent)
    local s = stringx.replace(sent, UNK_WORD, '')
    s = stringx.replace(s, START_WORD, '')
    s = stringx.replace(s, END_WORD, '')
    return s
end

local function strip(s)
    return s:gsub("^%s+",""):gsub("%s+$","")
end

function flip_table(u)
    local t = {}
    for key, value in pairs(u) do
        t[value] = key
    end
    return t   
end
